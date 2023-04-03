
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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--test_data_path', type=str)


    return parser.parse_args()

if __name__=='__main__':
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)

    #load model
    opts = parse_args()

    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    transform=transforms_dict['transform_inference']

    test_data_path=opts.test_data_path
    output_path=opts.output_path
    os.makedirs(output_path, exist_ok=True)

    imagepaths = os.listdir(test_data_path)
    for imagepath in tqdm(imagepaths):
        image = Image.open(os.path.join(test_data_path, imagepath))
        image = image.convert('RGB')
        image = transform(image)
        image = tensor2im(image)
        im_save_path = os.path.join(output_path, imagepath)
        Image.fromarray(np.array(image)).save(im_save_path)


