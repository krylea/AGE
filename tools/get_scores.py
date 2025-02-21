import torch
import os
from argparse import Namespace, ArgumentParser

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import sys
import random

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from utils.common import tensor2im
from options.test_options import TestOptions
from models.age import AGE

from tools.scores import METRICS, dataset_to_tensor, clamp

#
#   Dataset stuff
#


import torch
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import Dataset, ConcatDataset, Subset, IterableDataset, random_split
import torch



import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import Lambda, Resize, Compose

import numpy as np
import cv2
import math
import os
import datetime


"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images



class ImagesDataset(Dataset):
    @classmethod
    def from_folder(cls, source_root, opts, transforms=None):
        source_paths = sorted(make_dataset(source_root))
        return cls(source_paths, opts, transforms)

    @classmethod
    def from_folder_by_category(cls, source_root, opts, transforms=None):
        source_paths = sorted(make_dataset(source_root))
        all_category_paths = {}
        for path in source_paths:
            cate = path.split('/')[-1].split('_')[0]
            if cate not in all_category_paths:
                all_category_paths[cate] = []
            all_category_paths[cate].append(path)
        return [cls(category_paths, opts, transforms) for category_paths in all_category_paths.values()]
    
    def __init__(self, source_paths, opts, transforms=None):
        super().__init__()
        self.source_paths = source_paths
        self.transforms = transforms
        #self.average_codes = torch.load(opts.class_embedding_path, map_location=torch.device("cpu"))
        self.opts = opts

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        path = self.source_paths[index]
        im = Image.open(path).convert('RGB')
        if self.transforms:
            im = self.transforms(im)

        return im
    

#
#   AGE stuff
#


def get_n_distribution(net, transform, class_embeddings, opts):
    samples=os.listdir(opts.train_data_path)
    xs=[]
    for s in tqdm(samples):
        cate=s.split('_')[0]
        av_codes=class_embeddings[cate].cuda()
        from_im = Image.open(os.path.join(opts.train_data_path,s))
        from_im = from_im.convert('RGB')
        from_im = transform(from_im)
        with torch.no_grad():
            x=net.get_code(from_im.unsqueeze(0).to("cuda").float(), av_codes.unsqueeze(0))['x']
            x=torch.stack(x)
            xs.append(x)
    codes=torch.stack(xs).squeeze(2).squeeze(2).permute(1,0,2).cpu().numpy()
    mean=np.mean(codes,axis=1)
    mean_abs=np.mean(np.abs(codes),axis=1)
    cov=[]
    for i in range(codes.shape[0]):
        cov.append(np.cov(codes[i].T))
    os.makedirs(opts.n_distribution_path, exist_ok=True)
    np.save(os.path.join(opts.n_distribution_path, 'n_distribution.npy'),{'mean':mean, 'mean_abs':mean_abs, 'cov':cov})


class Sampler():
    def __init__(self, dist, opts):
        self.dist = dist
        self.opts = opts

    def __call__(self, outputs):
        means=self.dist['mean']
        means_abs=self.dist['mean_abs']
        covs=self.dist['cov']
        one = torch.ones_like(torch.from_numpy(means[0]))
        zero = torch.zeros_like(torch.from_numpy(means[0]))
        dws=[]
        groups=[[0,1,2],[3,4,5]]
        for i in range(means.shape[0]):
            x=torch.from_numpy(np.random.multivariate_normal(mean=means[i], cov=covs[i], size=1)).float().cuda()
            mask = torch.where(torch.from_numpy(means_abs[i])>self.opts.beta, one, zero).cuda()
            x=x*mask
            for g in groups[i]:
                dw=torch.matmul(outputs['A'][g], x.transpose(0,1)).squeeze(-1)
                dws.append(dw)
        dws=torch.stack(dws)
        codes = torch.cat(((self.opts.alpha*dws.unsqueeze(0)+ outputs['ocodes'][:, :6]), outputs['ocodes'][:, 6:]), dim=1)
        return codes



def apply_transforms(images, image_size=-1):
    var = ((images + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1

    if image_size > 0:
        transform = TF.Resize(image_size)
        var = transform(var)

    return var


def get_class_generations(net, reference_images, num_generations, sampler, class_ids=None, transforms=None, seed=None):
    def _generate_image(from_im):
        outputs = net.get_test_code(from_im.unsqueeze(0).to("cuda").float())
        codes=sampler(outputs)
        with torch.no_grad():
            res0 = net.decode(codes, randomize_noise=False, resize=sampler.opts.resize_outputs)
        return res0.cpu().detach()
    
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    if class_ids is None:
        class_ids = range(len(reference_images))

    generated_images = []
    for class_id in tqdm(class_ids):
        class_generations = []
        for i in range(num_generations):
            source_image = random.choice(reference_images[class_id])
            generated_image = _generate_image(source_image)
            if transforms is not None:
                generated_image = transforms(generated_image)
            class_generations.append(generated_image)
        class_generations = torch.cat(class_generations, dim=0)
        generated_images.append(class_generations)
    generated_images = torch.stack(generated_images, dim=0)

    return generated_images

def split_datasets(datasets, num_reference, eval_size=-1, seed=None):
    if seed is not None:
        rng=torch.Generator()
        rng.manual_seed(seed)
    ref_datasets = []
    eval_datasets = []
    for dataset in tqdm(datasets):
        #ref_inds = random.sample(range(len(dataset)), num_reference)
        indices = torch.randperm(len(dataset), generator=rng)
        ref_inds = indices[:num_reference]
        eval_inds = indices[num_reference:num_reference+eval_size] if eval_size > 0 else indices[num_reference:]
        ref_datasets.append(torch.stack([dataset[i] for i in ref_inds], dim=0))
        eval_datasets.append(torch.stack([dataset[i] for i in eval_inds], dim=0))
    return ref_datasets, eval_datasets
    
def evaluate_scores(generator, ref_datasets, eval_datasets, num_images, sampler, metrics=('fid', 'lpips'), device=torch.device("cuda"), 
        image_size=-1, seed=None, **kwargs): 
    scores = {}

    transforms = [Lambda(clamp)]
    if image_size > 0:
        transforms.append(Resize((image_size, image_size)))
    transforms = Compose(transforms)
    
    #ref_datasets, eval_datasets = split_datasets(dataset, reference_size)
    
    all_class_generations = get_class_generations(generator, ref_datasets, num_images, sampler, transforms=transforms, seed=seed)
    datasets = [transforms(dataset_to_tensor(dataset_i)) for dataset_i in eval_datasets]

    for metric in metrics:
        metric_fct = METRICS[metric](device=device)

        scores[metric] = metric_fct(all_class_generations, datasets, **kwargs)
    
    return scores


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--n_distribution_path', type=str)
    parser.add_argument('--n_images', type=int, default=128)
    parser.add_argument('--n_ref', type=int, default=30)
    parser.add_argument('--image_size', type=int, default=128)

    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=0.005)

    return parser.parse_args()


if __name__=='__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    args = parse_args()

    #load model
    #test_opts = TestOptions().parse()
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(args))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    #opts['output_size'] = 256
    opts = Namespace(**opts)
    #opts.resize_outputs=True
    #opts.output_size=256
    net = AGE(opts)
    net.eval()
    net.cuda()
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    transform=transforms_dict['transform_inference']

    # get n distribution (only needs to be executed once)
    #if not os.path.exists(os.path.join(opts.n_distribution_path, 'n_distribution.npy')):
    #    class_embeddings=torch.load(os.path.join(test_opts.class_embedding_path, 'class_embeddings.pt'))
    #    get_n_distribution(net, transform, class_embeddings, test_opts)


    dist=np.load(opts.n_distribution_path, allow_pickle=True).item()
    sampler = Sampler(dist, opts)


    #train_datasets = ImagesDataset.from_folder_by_category(source_root=dataset_args['train_source_root'], opts=opts, transforms=transform)
    test_datasets = ImagesDataset.from_folder_by_category(source_root=dataset_args['test_source_root'], opts=opts, transforms=transform)
    
    time = datetime.datetime.now()
    outname = opts.name + ".txt" if len(opts.name) > 0 else "%d_%d_%d%d.txt" % (time.month, time.day, time.hour, time.minute)
    outfile = os.path.join(args.output_path, outname)

    #evaluate_scores = evaluate_scores_by_class if not opts.combine_fid else evaluate_scores_all
    num_reference=30
    ref_datasets, eval_datasets = split_datasets(test_datasets, num_reference, seed=0)

    test_scores = evaluate_scores(net, ref_datasets, eval_datasets, 128, sampler)

    with open(outfile, 'w') as writer:
        writer.write("Test:\n")
        for metric, metric_scores in test_scores.items():
            writer.write('%s:\t%f\n' % (metric, metric_scores))

    #train_scores = evaluate_scores(train_datasets, net, opts.kshot, sampler, num_images=opts.n_images)

    #with open(outfile, 'a') as writer:
    #    writer.write("Train:\n")
    #    for metric, metric_scores in train_scores.items():
    #        writer.write('%s:\t%f\n' % (metric, metric_scores))




    