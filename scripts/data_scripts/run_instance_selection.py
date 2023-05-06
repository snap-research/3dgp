"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.

Filters a given dataset with InceptionV3 to remove out-of-distribution images
from the "Instance Selection for GANs" paper: https://arxiv.org/abs/2007.15255
"""

import os
import argparse
import sys; sys.path.extend(['.', 'src']) # Otherwise, detector will not load
from typing import List

import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from src.metrics.metric_utils import get_feature_detector
from scripts.utils import ImagePathsDataset, find_images_in_dir, copy_files

#----------------------------------------------------------------------------

def compute_gassian_ll_scores(embeddings, reg_covar: float=0.0):
    gmm = GaussianMixture(n_components=1, reg_covar=reg_covar)
    gmm.fit(embeddings)
    log_likelihood = gmm.score_samples(embeddings)
    return log_likelihood

#----------------------------------------------------------------------------

def run_instance_selection(source_dir, output_dir: os.PathLike, num_imgs_to_keep: int=None, keep_ratio: float=None, subdir_wise: bool=False, reg_covar: float=0.0, verbose: bool=True):
    assert num_imgs_to_keep is not None or keep_ratio is not None, f"Either num_imgs_to_keep or keep_ratio must be specified: {num_imgs_to_keep}, {keep_ratio}."
    assert num_imgs_to_keep is None or keep_ratio is None, f"Only one of num_imgs_to_keep or keep_ratio must be specified: {num_imgs_to_keep}, {keep_ratio}."
    assert num_imgs_to_keep is None or num_imgs_to_keep > 0, f"num_imgs_to_keep must be positive: {num_imgs_to_keep}."
    assert keep_ratio is None or keep_ratio > 0 and keep_ratio <= 1, f"keep_ratio must be in (0, 1]: {keep_ratio}."

    if subdir_wise:
        for subdir in tqdm([d for d in sorted(os.listdir(source_dir)) if os.path.isdir(os.path.join(source_dir, d))]):
            run_instance_selection(
                source_dir=os.path.join(source_dir, subdir),
                output_dir=os.path.join(output_dir, subdir),
                num_imgs_to_keep=num_imgs_to_keep,
                keep_ratio=keep_ratio,
                subdir_wise=False,
                reg_covar=reg_covar,
                verbose=False,
            )
        return

    # Extracting features
    src_paths = find_images_in_dir(source_dir)
    num_imgs_total = len(src_paths)
    num_imgs_to_keep = int(num_imgs_total * keep_ratio) if num_imgs_to_keep is None else num_imgs_to_keep
    assert num_imgs_to_keep > 0 and num_imgs_to_keep <= num_imgs_total, f"num_imgs_to_keep must be in [1, {num_imgs_total}]: {num_imgs_to_keep}."
    if verbose: print(f"Going to keep {num_imgs_to_keep}/{num_imgs_total} images.")
    embs = extract_embs(src_paths, verbose=verbose) # [num_images, d]
    if verbose: print('Computing densities...', end='')
    ll_scores = compute_gassian_ll_scores(embs, reg_covar=reg_covar) # [num_images]
    if verbose: print('Done!')
    order = np.argsort(ll_scores) # [num_images]

    imgs_to_save = [src_paths[i] for i in order[-num_imgs_to_keep:]] # [num_imgs_to_keep]
    imgs_to_save_subpaths = [os.path.relpath(p, start=source_dir) for p in imgs_to_save] # [num_imgs_to_keep]
    copy_files(
        source_dir=source_dir,
        files_to_copy=imgs_to_save_subpaths,
        output_dir=output_dir,
        num_jobs=16,
        verbose=verbose,
    )

#----------------------------------------------------------------------------

@torch.no_grad()
def extract_embs(img_paths: List[os.PathLike], device='cuda', batch_size: int=256, verbose: bool=True):
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector = get_feature_detector(detector_url, device='cuda', verbose=verbose)
    img_dataset = ImagePathsDataset(img_paths, transform=transforms.ToTensor())
    dataloader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=5)
    all_feats = []

    for batch in tqdm(dataloader, desc='Extracting features', disable=not verbose):
        imgs = batch["image"].to(device)
        assert imgs.min() >= 0
        assert imgs.max() <= 1
        imgs = (imgs * 255).to(torch.uint8) # [batch_size, c, h, w]
        feats = detector(imgs, return_features=True) # [batch_size, d]
        all_feats.extend(feats.cpu().numpy())

    return np.stack(all_feats).astype(np.float32)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_dir', required=True, type=str, help='Source directory')
    parser.add_argument('-o', '--output_dir', required=True, type=str, default=None, help='Target directory')
    parser.add_argument('-n', '--num_imgs_to_keep', required=False, default=None, type=int, help='Keep ratio for the dataset in terms of the amount of images.')
    parser.add_argument('-r', '--keep_ratio', required=False, default=None, type=float, help='Keep ratio for the dataset in terms of the ratio.')
    parser.add_argument('--subdir_wise', action='store_true', help='Should we process the features for each subdirectory separately? (Should)')
    parser.add_argument('--reg_covar', type=float, default=0.0, help='Covariance regularization strength — should be used when num_images < 2048. For ImageNet, we used 1e-05.')
    args = parser.parse_args()

    run_instance_selection(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        num_imgs_to_keep=args.num_imgs_to_keep,
        keep_ratio=args.keep_ratio,
        subdir_wise=args.subdir_wise,
        reg_covar=args.reg_covar,
    )

#----------------------------------------------------------------------------
