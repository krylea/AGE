import os
from argparse import Namespace

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import sys
import json

import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from utils.common import tensor2im, log_input_image
from options.test_options import TestOptions
from models.age import AGE


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

def run():
    test_opts = TestOptions().parse()
    ckpt = torch.load(test_opts.psp_checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    net = AGE(opts)
    net.eval()
    net.cuda()

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    transform = transforms_dict['transform_inference']
    data_path=test_opts.train_data_path
    class_embedding_path=test_opts.class_embedding_path
    os.makedirs(class_embedding_path, exist_ok=True)
    dataset = InferenceDataset(root=data_path,
                            transform=transform,
                            opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    codes={}
    counts={}
    for input_batch, cate_batch in tqdm(dataloader):
        with torch.no_grad():
            input_batch = input_batch.cuda()
            for image_idx, input in enumerate(input_batch):
                input_image = input
                cate = cate_batch[image_idx]
                outputs = net.get_test_code(input_image.unsqueeze(0).float())
                # save codes
                if cate not in codes.keys():
                    codes[cate]=outputs['ocodes'][0]
                    counts[cate]=1
                else:
                    codes[cate]+=outputs['ocodes'][0]
                    counts[cate]+=1
    means={}
    for cate in codes.keys():
        means[cate]=codes[cate]/counts[cate]
    torch.save(means, os.path.join(class_embedding_path, 'class_embeddings.pt'))
    
    get_n_distribution(net, transform, means, opts)

if __name__ == '__main__':
    run()
