import os
import re
import json
import shutil
import random
import contextlib
import zipfile
import pickle
from typing import List, Dict, Tuple

import click
import joblib
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TVF
from torchvision.utils import make_grid
from src import dnnlib
from typing import Optional, Callable
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader


#----------------------------------------------------------------------------

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

#----------------------------------------------------------------------------

def display_dir(dir_path: os.PathLike, num_imgs: int=25, selection_strategy: str="order", n_skip_imgs: int=0, ignore_regex=None, **kwargs) -> "Image":
    Image.init()
    if selection_strategy in ('order', 'random'):
        img_paths = find_images_in_dir(dir_path, ignore_regex)
        img_paths = img_paths[n_skip_imgs:]

    if selection_strategy == 'order':
        img_paths = img_paths[:num_imgs]
    elif selection_strategy == 'random':
        img_paths = random.sample(img_paths, k=num_imgs)
    elif selection_strategy == 'random_imgs_from_subdirs':
        img_paths = [p for d in [d for d in listdir_full_paths(dir_path) if os.path.isdir(d)] for p in random.sample(listdir_full_paths(d), k=num_imgs)]
    else:
        raise NotImplementedError(f'Unknown selection strategy: {selection_strategy}')

    return display_imgs(img_paths, **kwargs)

#----------------------------------------------------------------------------

def display_imgs(img_paths: List[os.PathLike], nrow: bool=None, resize: int=None, crop: Tuple=None, padding: int=2) -> "Image":
    imgs = [Image.open(p) for p in img_paths]
    imgs = [(x.convert('RGB') if np.array(x).dtype != np.int32 else x) for x in imgs]
    if not crop is None:
        imgs = [img.crop(crop) for img in imgs]
    if not resize is None:
        imgs = [TVF.resize(x, size=resize, interpolation=TVF.InterpolationMode.LANCZOS) for x in imgs]
    imgs = [TVF.to_tensor(TVF.center_crop(x, output_size=min(x.size))) for x in imgs] # [num_imgs, c, h, w]
    imgs = [((x.float() / 2 ** 16) if x.dtype == torch.int32 else x) for x in imgs] # [num_imgs, c, h, w]
    imgs = [x.repeat(3, 1, 1) if x.shape[0] == 1 else x for x in imgs] # [num_imgs, c, h, w]
    imgs = torch.stack(imgs) # [num_imgs, c, h, w]
    grid = make_grid(imgs, nrow=(int(np.sqrt(imgs.shape[0])) if nrow is None else nrow), padding=padding) # [c, grid_h, grid_w]
    grid = TVF.to_pil_image(grid)

    return grid

#----------------------------------------------------------------------------

def resize_and_save_image(src_path: str, trg_path: str, size: int, ignore_grayscale: bool=False, ignore_broken: bool=False, ignore_existing: bool=False):
    Image.init()
    assert file_ext(src_path) in Image.EXTENSION, f"Unknown image extension: {src_path}"
    assert file_ext(trg_path) in Image.EXTENSION, f"Unknown image extension: {trg_path}"

    if ignore_existing and os.path.isfile(trg_path):
        return

    try:
        img = Image.open(src_path)
        if img.mode == 'L' and ignore_grayscale:
            return
        img.load() # required for png.split()
    except:
        if ignore_broken:
            return
        else:
            raise

    img = center_resize_crop(img, size)
    jpg_kwargs = {'quality': 95} if file_ext(trg_path) == '.jpg' else {}

    if file_ext(trg_path) in ('.jpg', '.jpeg') and len(img.split()) == 4:
        jpg = Image.new("RGB", img.size, (255, 255, 255))
        jpg.paste(img, mask=img.split()[3]) # 3 is the alpha channel
        jpg.save(trg_path, **jpg_kwargs)
    else:
        if img.mode == "CMYK":
            img = img.convert("RGB")
        img.save(trg_path, **jpg_kwargs)

#----------------------------------------------------------------------------

def center_resize_crop(img: Image, size: int) -> Image:
    img = TVF.center_crop(img, min(img.size)) # First, make it square
    img = TVF.resize(img, size, interpolation=TVF.InterpolationMode.LANCZOS) # Now, resize it

    return img

#----------------------------------------------------------------------------

def file_ext(path: os.PathLike) -> str:
    return os.path.splitext(path)[1].lower()

#----------------------------------------------------------------------------

# Extract the zip file for simplicity...
def extract_zip(zip_path: os.PathLike, overwrite: bool=False):
    assert file_ext(zip_path) == '.zip', f'Not a zip archive: {zip_path}'

    if os.path.exists(zip_path[:-4]):
        if overwrite or click.confirm(f'Dir {zip_path[:-4]} already exists. Delete it?', default=False):
            shutil.rmtree(zip_path[:-4])

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(zip_path[:-4]))

#----------------------------------------------------------------------------

def compress_to_zip(dir_to_compress: os.PathLike, delete: bool=False):
    shutil.make_archive(dir_to_compress, 'zip', root_dir=os.path.dirname(dir_to_compress), base_dir=os.path.basename(dir_to_compress))

    if delete:
        shutil.rmtree(dir_to_compress)

#----------------------------------------------------------------------------

def load_generator(cfg: DictConfig, verbose: bool=True) -> Tuple[torch.nn.Module, Dict, str]:
    """
    args:
    - cfg.random_init --- use a randomly initialized generator?
    """
    if not cfg.networks_dir is None:
        assert cfg.network_pkl is None, "Cant have both parameters: network_pkl and cfg.networks_dir"
        if not cfg.selection_metric is None:
            metrics_file = os.path.join(cfg.networks_dir, f'metric-{cfg.selection_metric}.jsonl')
            with open(metrics_file, 'r') as f:
                snapshot_metrics_vals = [json.loads(line) for line in f.read().splitlines()]
            snapshot = sorted(snapshot_metrics_vals, key=lambda m: m['results'][cfg.selection_metric])[0]
            network_pkl = os.path.join(cfg.networks_dir, snapshot['snapshot_pkl'])
            if verbose:
                print(f'Using checkpoint: {network_pkl} with {cfg.selection_metric} of', snapshot['results'][cfg.selection_metric])
        else:
            output_regex = "^network-snapshot-\d{6}.pkl$"
            ckpt_regex = re.compile(output_regex)
            ckpts = sorted([f for f in os.listdir(cfg.networks_dir) if ckpt_regex.match(f)])
            network_pkl = os.path.join(cfg.networks_dir, ckpts[-1])
            if verbose:
                print(f"Using the latest found checkpoint: {network_pkl}")
    elif not cfg.network_pkl is None:
        assert cfg.networks_dir is None, "Cant have both parameters: network_pkl and cfg.networks_dir"
        network_pkl = cfg.network_pkl
    else:
        raise NotImplementedError(f'Must provide either networks_dir or network_pkl.')

    # Load network.
    if not dnnlib.util.is_url(network_pkl, allow_file_urls=True) and not os.path.isfile(network_pkl):
        raise ValueError(f'--network must point to a file or URL, but got {network_pkl}')
    if verbose:
        print(f'Loading networks from {network_pkl}')
    with dnnlib.util.open_url(network_pkl) as f:
        snapshot = pickle.Unpickler(f).load()
        G = snapshot['G_ema'] # type: ignore

    G.cfg = dnnlib.EasyDict.init_recursively(G.cfg)

    if cfg.reload_code or cfg.random_init:
        from src.training.networks_epigraf import Generator
        G_new = Generator(
            G.cfg,
            mapping_kwargs=dnnlib.EasyDict(
                mean_camera_params=(G.mapping.mean_camera_params.cpu() if hasattr(G.mapping, 'mean_camera_params') else None),
                camera_cond=G.cfg.camera_cond if 'camera_cond' in G.cfg else False,
            ),
            img_resolution=G.img_resolution,
            img_channels=G.img_channels,
        )
        if not cfg.random_init:
            G_new.load_state_dict(G.state_dict())
        G = G_new

    return G, snapshot, network_pkl

#----------------------------------------------------------------------------

def find_images_in_dir(dir_path: os.PathLike, ignore_regex=None, full_path: bool=True):
    Image.init()
    imgs = [os.path.relpath(os.path.join(root, fname), start=dir_path) for root, _dirs, files in os.walk(dir_path) for fname in files]
    imgs = sorted([f for f in imgs if file_ext(f) in Image.EXTENSION])
    if full_path:
        imgs = [os.path.join(dir_path, f) for f in imgs]
    if not ignore_regex is None:
        imgs = [f for f in imgs if not re.fullmatch(ignore_regex, f)]
    return imgs

#----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

#----------------------------------------------------------------------------

def listdir_full_paths(d: os.PathLike) -> List[os.PathLike]:
    return [os.path.join(d, o) for o in sorted(os.listdir(d))]

#----------------------------------------------------------------------------

def get_all_files(dir_path: os.PathLike, full_path: bool=True, ext_white_list: List[str]=None) -> List[os.PathLike]:
    all_files = [os.path.join(root, fname) for root, _dirs, files in os.walk(dir_path) for fname in files]
    if not ext_white_list is None:
        ext_white_list = set(list(ext_white_list))
        all_files = [f for f in all_files if file_ext(f) in ext_white_list]
    if full_path:
        all_files = [os.path.join(dir_path, f) for f in all_files]
    return all_files

#----------------------------------------------------------------------------

def lanczos_resize_tensors(x: torch.Tensor, size):
    x = [TVF.to_pil_image(img) for img in x]
    x = [TVF.resize(img, size=size, interpolation=TVF.InterpolationMode.LANCZOS) for img in x]
    x = [TVF.to_tensor(img) for img in x]

    return torch.stack(x)

#----------------------------------------------------------------------------

def maybe_makedirs(d: os.PathLike):
    # TODO: what the hell is this function name?
    if d != '':
        os.makedirs(d, exist_ok=True)

#----------------------------------------------------------------------------

class ImagePathsDataset(VisionDataset):
    def __init__(self, img_paths: List[os.PathLike], transform: Optional[Callable]=None):
        self.imgs_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx: int):
        image = pil_loader(self.imgs_paths[idx])

        if not self.transform is None:
            image = self.transform(image)

        return dict(image=image, path=self.imgs_paths[idx])

#----------------------------------------------------------------------------

def remove_empty_files(dir_path: os.PathLike, verbose: bool=False):
    for f in tqdm(get_all_files(dir_path)):
        if os.stat(f).st_size == 0:
            if verbose:
                print(f'Removing empty file: {f}')
            os.remove(f)

#----------------------------------------------------------------------------

def copy_files(source_dir: os.PathLike, files_to_copy: List[os.PathLike], output_dir: os.PathLike, num_jobs: int=8, verbose: bool=True):
    """
    source_dir --- main dataset directory
    files_to_copy --- filepaths inside the `source_dir` directory
    output_dir --- where to save the files
    """
    jobs = []
    dirs_to_create = []

    for filepath in tqdm(files_to_copy, desc='Collecting jobs', disable=not verbose):
        src_file_path = os.path.join(source_dir, filepath)
        dst_file_path = os.path.join(output_dir, filepath)
        dirs_to_create.append(os.path.dirname(dst_file_path))
        jobs.append(joblib.delayed(shutil.copy)(src=src_file_path, dst=dst_file_path))

    for d in tqdm(list(set(dirs_to_create)), desc='Creating necessary directories', disable=not verbose):
        if d != '':
            os.makedirs(d, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Executing jobs", total=len(jobs), disable=not verbose)) as _progress_bar:
        joblib.Parallel(n_jobs=num_jobs)(jobs)

#----------------------------------------------------------------------------
