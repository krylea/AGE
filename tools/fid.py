import torch
import os
from argparse import Namespace, ArgumentParser

import torchvision.transforms as tf

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import sys
import random
import shutil

import lpips
import cv2

sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from utils.common import tensor2im
from options.test_options import TestOptions
from models.age import AGE

from torch.utils.data import Dataset

from pytorch_fid.fid_score import calculate_fid_given_paths


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


def sampler(outputs, dist, opts):
    means=dist['mean']
    means_abs=dist['mean_abs']
    covs=dist['cov']
    one = torch.ones_like(torch.from_numpy(means[0]))
    zero = torch.zeros_like(torch.from_numpy(means[0]))
    dws=[]
    groups=[[0,1,2],[3,4,5]]
    for i in range(means.shape[0]):
        x=torch.from_numpy(np.random.multivariate_normal(mean=means[i], cov=covs[i], size=1)).float().cuda()
        mask = torch.where(torch.from_numpy(means_abs[i])>opts.beta, one, zero).cuda()
        x=x*mask
        for g in groups[i]:
            dw=torch.matmul(outputs['A'][g], x.transpose(0,1)).squeeze(-1)
            dws.append(dw)
    dws=torch.stack(dws)
    codes = torch.cat(((opts.alpha*dws.unsqueeze(0)+ outputs['ocodes'][:, :6]), outputs['ocodes'][:, 6:]), dim=1)
    return codes


def fid(real, fake, gpu):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    command = 'python -m pytorch_fid {} {} --device cuda:{}'.format(real, fake, gpu)
    #command = 'python -m pytorch_fid {} {}'.format(real, fake)
    os.system(command)

def LPIPS(root):
    print('Calculating LPIPS...')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    model = loss_fn_vgg
    model.cuda()

    files = os.listdir(root)
    data = {}
    for file in tqdm(files, desc='loading data'):
        cls = file.split('_')[0]
        idx = int(file.split('_')[1][:-4])
        img = lpips.im2tensor(cv2.resize(lpips.load_image(os.path.join(root, file)), (32, 32)))
        data.setdefault(cls, {})[idx] = img

    classes = set([file.split('_')[0] for file in files])
    res = []
    for cls in tqdm(classes):
        temp = []
        files_cls = [file for file in files if file.startswith(cls + '_')]
        for i in range(0, len(files_cls) - 1, 1):
            # print(i, end='\r')
            for j in range(i + 1, len(files_cls), 1):
                img1 = data[cls][i].cuda()
                img2 = data[cls][j].cuda()

                d = model(img1, img2, normalize=True)
                temp.append(d.detach().cpu().numpy())
        res.append(np.mean(temp))
    return np.mean(res)


def eval(args, opts, net, dist, datasets):
    transform = tf.Compose([
                    tf.Resize((256, 256)),
                    tf.ToTensor(),
                    tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

    transform2 = tf.Resize((args.image_size, args.image_size))

    if os.path.exists(args.real_dir):
        shutil.rmtree(args.real_dir)
    if os.path.exists(args.fake_dir):
        shutil.rmtree(args.fake_dir)

    os.makedirs(args.real_dir, exist_ok=True)
    os.makedirs(args.fake_dir, exist_ok=True)
    
    for i, class_dataset in tqdm(enumerate(datasets)):
        all_class_images = [x for x in class_dataset]
        indices = np.random.permutation(len(all_class_images))
        ref_inds = indices[:args.n_ref]
        fid_inds = indices[args.n_ref:args.n_ref+args.n_eval] if args.n_eval > 0 else indices[args.n_ref:]
        cond_images = [all_class_images[idx] for idx in ref_inds]
        fid_images = [all_class_images[idx] for idx in fid_inds]
        for j in range(args.n_images):
            idx = random.choice(range(len(cond_images)))
            from_im = transform(cond_images[idx])
            outputs = net.get_test_code(from_im.unsqueeze(0).to("cuda").float())
            codes=sampler(outputs, dist, opts)
            with torch.no_grad():
                res0 = net.decode(codes, randomize_noise=args.randomize_noise, resize=args.resize_outputs)
            res0 = tensor2im(transform2(res0[0]))
            im_save_path = os.path.join(args.fake_dir, "%d_%d.jpg" % (i, j))
            Image.fromarray(np.array(res0)).save(im_save_path)

        
        for j, image in enumerate(fid_images):
            im_save_path = os.path.join(args.real_dir, "%d_%d.jpg" % (i, j))
            image = transform2(image)
            image.save(im_save_path)

    fid_score = calculate_fid_given_paths((args.real_dir, args.fake_dir), 50, torch.device("cuda"), 2048).item()
    print("FID:", str(fid_score))
    lpips_score = LPIPS(args.fake_dir).item()
    print("LPIPS:", str(lpips_score))

    return fid_score, lpips_score


parser = ArgumentParser()
parser.add_argument('--name', type=str,default="results/flower_wavegan_base_index")
parser.add_argument('--dataset', type=str, default="animalfaces")
parser.add_argument('--eval_path', type=str, default="eval_results2.txt")
parser.add_argument('--real_dir', type=str, default="results/flower_wavegan_base_index/reals")
parser.add_argument('--fake_dir', type=str,default="results/flower_wavegan_base_index/tests")
parser.add_argument('--test_data_path', type=str)
parser.add_argument('--n_sample_test', type=int, default=1)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--n_distribution_path', type=str)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--n_images', type=int, default=128)
parser.add_argument('--n_eval', type=int, default=-1)
parser.add_argument('--n_ref', type=int, default=30)
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--n_exps', type=int, default=1)
parser.add_argument('--randomize_noise', action='store_true')
parser.add_argument('--resize_outputs', action='store_true')
#parser.add_argument('--resize_outputs', type=int, default=30)
args = parser.parse_args()

if __name__=='__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    #load model
    #test_opts = TestOptions().parse()
    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(args))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)
    opts.A_length=100
    net = AGE(opts)
    net.eval()
    net.cuda()

    datasets = ImagesDataset.from_folder_by_category(args.test_data_path, opts, transforms=None)

    dist=np.load(os.path.join(opts.n_distribution_path, 'n_distribution.npy'), allow_pickle=True).item()

    fid_scores = []
    lpips_scores=[]
    for i in range(args.n_exps):
        fid_score, lpips_score = eval(args, opts, net, dist, datasets)
        fid_scores.append(fid_score)
        lpips_scores.append(lpips_score)
    fid_out = sum(fid_scores) / len(fid_scores)
    lpips_out = sum(lpips_scores) / len(lpips_scores)

    with open(args.eval_path, 'a') as writer:
        fid_scores_str = ", ".join(["%.2f" % (x,) for x in fid_scores])
        lpips_scores_str = ", ".join(["%.2f" % (x,) for x in lpips_scores])
        writer.write("%s:\tFID: %.2f (%s)\tLPIPS: %.2f (%s)\n" % (args.name, fid_out, fid_scores_str, lpips_out, lpips_scores_str)) 






    