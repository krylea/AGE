# Copyright (C) 2021 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, check out LICENSE.md

#
#   From Imaginaire
#


import os

import requests
import torch.distributed as dist
import torchvision.utils

from imaginaire.utils.distributed import is_master



def download_file_from_google_drive(URL, destination):
    r"""Download a file from google drive.

    Args:
        URL: GDrive file ID.
        destination: Path to save the file.

    Returns:

    """
    download_file(f"https://docs.google.com/uc?export=download&id={URL}", destination)


def download_file(URL, destination):
    r"""Download a file from google drive or pbss by using the url.

    Args:
        URL: GDrive URL or PBSS pre-signed URL for the checkpoint.
        destination: Path to save the file.

    Returns:

    """
    session = requests.Session()
    response = session.get(URL, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    r"""Get confirm token

    Args:
        response: Check if the file exists.

    Returns:

    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    r"""Save response content

    Args:
        response:
        destination: Path to save the file.

    Returns:

    """
    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


