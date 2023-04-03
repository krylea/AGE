
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

if __name__=='__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    #load model
    test_opts = TestOptions().parse()
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    if 'learn_in_w' not in opts:
        opts['learn_in_w'] = False
    if 'output_size' not in opts:
        opts['output_size'] = 1024
    opts = Namespace(**opts)

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    transform=transforms_dict['transform_inference']

    test_data_path=test_opts.test_data_path
    output_path=test_opts.output_path
    os.makedirs(output_path, exist_ok=True)

    imagepaths = os.listdir(test_data_path)
    for imagepath in imagepaths:
        image = Image.open(os.path.join(test_data_path, imagepath))
        image = image.convert('RGB')
        image = transform(image)
        image = tensor2im(image)
        im_save_path = os.path.join(output_path, imagepath)
        Image.fromarray(np.array(image)).save(im_save_path)


