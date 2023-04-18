import os
import random
import shutil

import cv2
import lpips
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torch.utils.data
import torchvision.transforms as transforms

import torch
import os
from argparse import Namespace

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

from torch.utils.data import Dataset


def fid(real, fake):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    #command = 'python -m pytorch_fid {} {} --gpu {}'.format(real, fake, gpu)
    command = 'python -m pytorch_fid {} {} --device cuda:0'.format(real, fake)
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
    print(np.mean(res))

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


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,default="results/flower_wavegan_base_index")
parser.add_argument('--dataset', type=str, default="animalfaces")
parser.add_argument('--real_dir', type=str, default="results/flower_wavegan_base_index/reals")
parser.add_argument('--fake_dir', type=str,default="results/flower_wavegan_base_index/tests")
parser.add_argument('--n_sample_test', type=int, default=1)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--n_distribution_path', type=str)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=0.005)
parser.add_argument('--n_images', type=int, default=128)
args = parser.parse_args()


transform_list = [ transforms.ToTensor(), transforms.Resize((256, 256)),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)

transform2 = transforms.Resize((128,128))

if __name__ == '__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

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
    net = AGE(opts)
    net.eval()
    net.cuda()

    real_dir = args.real_dir
    fake_dir = args.fake_dir#os.path.join(args.name, args.fake_dir)
    print('real dir: ', real_dir)
    print('fake dir: ', fake_dir)

    if os.path.exists(fake_dir):
        shutil.rmtree(fake_dir)
    os.makedirs(fake_dir, exist_ok=True)

    data = np.load("animal_128.npy")
    if args.dataset == 'flower':
        data = data[85:]
        num = 10
    elif args.dataset == 'animalfaces':
        data = data[119:]
        num = 10
    elif args.dataset == 'vggface':
        data = data[1802:]
        num = 30

    #num=30
    data_for_gen = data[:, :num, :, :, :]
    data_for_fid = data[:, num:, :, :, :]

    if not os.path.exists(real_dir):
        os.makedirs(real_dir, exist_ok=True)
        
        for cls in tqdm(range(data_for_fid.shape[0]), desc='preparing real images'):
            for i in range(128):
                idx = np.random.choice(data_for_fid.shape[1], 1)
                real_img = data_for_fid[cls, idx, :, :, :][0]
                if args.dataset == 'vggface':
                    real_img *= 255
                real_img = Image.fromarray(np.uint8(real_img))
                real_img.save(os.path.join(real_dir, '{}_{}_{}.png'.format(cls, str(i).zfill(3), idx.item())), 'png')

    if os.path.exists(fake_dir):
        dist=np.load(os.path.join(opts.n_distribution_path, 'n_distribution.npy'), allow_pickle=True).item()
        for cls in tqdm(range(data_for_gen.shape[0]), desc='generating fake images'):
            for i in range(128):
                idx = np.random.choice(data_for_gen.shape[1], args.n_sample_test)
                imgs = data_for_gen[cls, idx, :, :, :]
                imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], dim=0).cuda()
                outputs = net.get_test_code(imgs.float())
                codes=sampler(outputs, dist, opts)
                with torch.no_grad():
                    res0 = net.decode(codes, randomize_noise=False, resize=True)
                res0 = tensor2im(transform2(res0[0]))
                im_save_path = os.path.join(fake_dir, "image_%d_%d_%d.jpg" % (cls, i, idx.item()))
                Image.fromarray(np.array(res0)).save(im_save_path)

    fid(real_dir, fake_dir)