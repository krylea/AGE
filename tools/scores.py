"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
from torchvision.transforms import Lambda, Resize, Compose
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d

from torch.utils.data import Dataset, IterableDataset, TensorDataset, Subset

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from pytorch_fid.inception import InceptionV3

import lpips

import math

from torchmetrics.image.fid import FrechetInceptionDistance


def dataset_to_tensor(dataset):
    if isinstance(dataset, IterableDataset):
        tensors = [x for x in dataset]
    else:
        tensors = [dataset[i] for i in range(len(dataset))]
    return torch.stack(tensors, dim=0)

class FIDMetric2():
    def __init__(self, normalize=True, dims=2048, device=torch.device("cuda")):
        self.fid = FrechetInceptionDistance(dims=dims, normalize=normalize)

    def __call__(self, fake_inputs, real_inputs):
        if isinstance(real_inputs, Dataset):
            real_inputs = dataset_to_tensor(real_inputs)
        self.fid.reset()
        self.fid.update(real_inputs, real=True)
        self.fid.update(fake_inputs, real=False)
        return self.fid.compute()

class FID():
    def __init__(self, dims=2048, device=torch.device("cuda"), metrics_path=None, eps=1e-6):
        self.dims = dims
        self.device = device
        self.block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([self.block_idx]).to(self.device)
        self.metrics_path = metrics_path
        self.eps = eps


    def get_activations(self, dataset, batch_size=50, num_workers=1, max_batches=-1):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                        Make sure that the number of samples is a multiple of
                        the batch size, otherwise some samples are ignored. This
                        behavior is retained to match the original FID score
                        implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
        """
        self.model.eval()

        if batch_size > len(dataset):
            print(('Warning: batch size is bigger than the data size. '
                'Setting batch size to data size'))
            batch_size = len(dataset)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=False,
                                                num_workers=num_workers)

        pred_arr = np.empty((len(dataset), self.dims))

        start_idx = 0

        for batch in tqdm(dataloader):
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(self.device)

            with torch.no_grad():
                pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

            if max_batches > 0 and start_idx // batch_size >= max_batches:
                break

        return pred_arr

    def calculate_activation_statistics(self, dataset, batch_size=50, num_workers=1):
        """Calculation of the statistics used by the FID.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                        batch size batch_size. A reasonable batch size
                        depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                the inception model.
        """
        act = self.get_activations(dataset, batch_size, num_workers)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % self.eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * self.eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    def get_dataset_metrics(self, dataset, **kwargs):
        if isinstance(dataset, str):
            assert self.metrics_path is not None
            dataset_metrics_path = os.path.join(self.metrics_path, dataset+".npz")
            with np.load(dataset_metrics_path) as f:
                m, s = f['mu'][:], f['sigma'][:]
        else:
            if isinstance(dataset, torch.Tensor):
                dataset = TensorDataset(dataset)
            m, s = self.calculate_activation_statistics(dataset, **kwargs)

        return m, s

    def save_dataset_metrics(self, dataset, label, **kwargs):
        assert self.metrics_path is not None
        m, s = self.calculate_activation_statistics(dataset, **kwargs)
        dataset_metrics_path = os.path.join(self.metrics_path, label+".npz")
        np.savez(dataset_metrics_path, mu=m, sigma=s)

    def __call__(self, dataset1, dataset2, batch_size=50, num_workers=1):
        m1, s1 = self.get_dataset_metrics(dataset1, batch_size=batch_size, num_workers=num_workers)
        m2, s2 = self.get_dataset_metrics(dataset2, batch_size=batch_size, num_workers=num_workers)

        return self.calculate_frechet_distance(m1, s1, m2, s2)


class FIDMetric():
    def __init__(self, device=torch.device("cuda"), dims=2048):
        self.fid = FID(dims=dims, device=device)

    def _preprocess(self, dataset, renorm=True):
        if type(dataset) is list:
            if renorm:#convert from [-1,1] to [0,1]
                for dataset_i in dataset:
                    dataset_i = (dataset_i + 1)/2
            dataset = torch.cat(dataset, dim=0)
        elif len(dataset.size()) == 5:
            if renorm:#convert from [-1,1] to [0,1]
                dataset = (dataset+1)/2
            dataset = dataset.view(-1, *dataset.size()[-3:])
            

        return dataset
    
    def __call__(self, fake_images, real_images, batch_size=50, num_workers=1, renorm=True):
        fake_images = self._preprocess(fake_images, renorm=renorm)
        real_images = self._preprocess(real_images, renorm=renorm)

        return self.fid(fake_images, real_images, batch_size=batch_size, num_workers=num_workers)

class LPIPSMetric():
    def __init__(self, device=torch.device('cuda'), net='alex'):
        self.lpips = lpips.LPIPS(net=net).to(device)
        self.device = device

    def _evaluate_class(self, images):
        N = images.size(0)
        pair_scores = torch.zeros(N, N)
        for j in range(N):
            with torch.no_grad():
                pair_scores[j] = self.lpips(images[j:j+1].expand(N, -1, -1, -1).to(self.device), images.to(self.device)).view(N).cpu()
        pair_scores.fill_diagonal_(0)
        return pair_scores.sum() / (pair_scores.nelement() - N)

    def __call__(self, images, *args):
        if type(images) is list or len(images.size()) == 5:
            N = images.size(0)
            scores = torch.zeros(N)
            for i in range(N):
                scores[i] = self._evaluate_class(images[i])
            return scores.mean().item()
        elif len(images.size()) == 4:
            return self._evaluate_class(images)
        else:
            raise NotImplementedError()


METRICS = {
    'fid': FIDMetric,
    'lpips': LPIPSMetric
}


def clamp(images):
    images[images < -1] = -1
    images[images > 1] = 1
    return images

    
