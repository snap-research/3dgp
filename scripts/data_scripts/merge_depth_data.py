"""
Merges the directory of depth maps with the directory of images
We coulnd't extract the depth for all the images. Also, they are extracted under a weird name.
"""

import os
import shutil
import argparse

import numpy as np
from tqdm import tqdm

from scripts.utils import find_images_in_dir

#----------------------------------------------------------------------------

def merge_depth_data(images_dir: os.PathLike, depths_dir: os.PathLike, target_dir: os.PathLike):
    img_fnames = set(find_images_in_dir(images_dir, full_path=False))
    depth_fnames = set(find_images_in_dir(depths_dir, full_path=False))
    depth_img_fnames = set([f.replace('.png', '.jpg') for f in depth_fnames])
    imgs_to_save = img_fnames.intersection(depth_img_fnames)

    print(f'Going to save {len(imgs_to_save)} images...')

    # Get the intersection of the names
    for img_fname in tqdm(imgs_to_save):
        img_src_path = os.path.join(images_dir, img_fname)
        depth_src_path = os.path.join(depths_dir, img_fname.replace('.jpg', '.png'))
        img_trg_path = os.path.join(target_dir, img_fname)
        depth_trg_path = os.path.join(target_dir, img_fname.replace('.jpg', '_depth.png'))
        os.makedirs(os.path.dirname(img_trg_path), exist_ok=True)
        shutil.copy(img_src_path, img_trg_path)
        shutil.copy(depth_src_path, depth_trg_path)

    print('Done!')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images_dir', required=True, type=str, help='Source directory of images')
    parser.add_argument('-d', '--depths_dir', required=True, type=str, help='Source directory of depths')
    parser.add_argument('-t', '--target_dir', required=True, type=str, default=None, help='Target directory')
    args = parser.parse_args()

    merge_depth_data(
        images_dir=args.images_dir,
        depths_dir=args.depths_dir,
        target_dir=args.target_dir,
    )

#----------------------------------------------------------------------------
