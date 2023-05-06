"""
Generates a multi-view video of a sample from pi-GAN
Adapted from Adapted from https://github.com/marcoamonteiro/pi-GAN
"""

import sys; sys.path.extend(['.', 'src'])
import os
import re
from typing import List, Optional, Union

import hydra
import torch
import numpy as np
from omegaconf import DictConfig
import torchvision as tv
from torchvision.utils import make_grid
import torchvision.transforms.functional as TVF
from tqdm import tqdm
from PIL import Image

from src.dnnlib import EasyDict
from src.training.inference_utils import generate_trajectory, generate_camera_params
from scripts.utils import load_generator, set_seed, maybe_makedirs, lanczos_resize_tensors

#----------------------------------------------------------------------------

@hydra.main(config_path="../configs/scripts", config_name="inference.yaml")
def generate_vis(cfg: DictConfig):
    torch.set_grad_enabled(False)

    device = torch.device('cuda')
    save_dir = cfg.output_dir
    set_seed(cfg.seed) # To fix non-z randomization

    # Step 0. Preparing the generator
    G = load_generator(cfg.ckpt, verbose=cfg.verbose)[0]
    G = (G.pair_upsampler() if cfg.pair_upsampler else G).to(device).eval()
    G.cfg = EasyDict.init_recursively(G.cfg)
    G.synthesis.cfg = G.cfg
    G.cfg.use_full_box = G.cfg.get('use_full_box', False)
    G.cfg.use_inf_depth = G.cfg.get('use_inf_depth', True)
    G.cfg.camera = G.cfg.dataset.camera
    if hasattr(G.cfg, 'num_ray_steps'):
        G.synthesis.img_resolution = G.synthesis.test_resolution = cfg.img_resolution
        G.cfg.num_ray_steps = G.cfg.num_ray_steps * cfg.ray_step_multiplier
        G.cfg.dataset.white_back = True if cfg.force_whiteback else G.cfg.dataset.white_back
    G.nerf_noise_std = 0
    G.cfg.camera.ray.end = G.cfg.camera.ray.end + cfg.get('far_plane_offset', 0)
    maybe_makedirs(save_dir)

    # Step 1. Sampling ws codes
    assert not (cfg.seeds is None and cfg.num_seeds is None), "You must specify either `num_seeds` or `seeds`"
    assert cfg.seeds is None or cfg.num_seeds is None, "You cannot specify both `num_seeds` and `seeds`"
    seeds = cfg.seeds if cfg.num_seeds is None else (cfg.seed + np.arange(cfg.num_seeds))
    classes = None if cfg.classes is None else parse_range(cfg.classes)
    sample_names = [f'{s:04d}' for s in seeds] if classes is None else [f'c{c:04d}-s{s:04d}' for c in classes for s in seeds]
    ws, z, c = sample_ws_from_seeds(G, seeds, cfg, device, classes=classes, num_interp_steps=cfg.num_interp_steps) # [num_grids, num_ws, w_dim]

    # Step 2. Sampling camera trajectories
    camera_params = generate_camera_params(G, z, c, cfg.trajectory)

    # Step 3. Sampling images.
    if cfg.vis.name == 'image_grid':
        images = generate_trajectory(cfg, G, ws, camera_params=camera_params, **cfg.synthesis_kwargs) # [num_frames, num_c * num_seeds, c, h, w]
        # images = images.permute(1, 0, 2, 3, 4).reshape(-1, *images.shape[2:]) # [num_c * num_seeds * num_frames, c, h, w]
        images = torch.cat(list(images), dim=3) # [num_c * num_seeds, c, h, w * num_frames]
        save_images_as_grids(images, cfg.vis.num_images_per_grid, sample_names, save_dir, save_ext=cfg.vis.save_ext, nrow=cfg.vis.nrow)
    elif cfg.vis.name == 'video_grid':
        for grid_idx in tqdm(list(range(0, (len(ws) + cfg.vis.num_videos_per_grid - 1) // cfg.vis.num_videos_per_grid)), desc='Generating'):
            grid_slice = slice(grid_idx * cfg.vis.num_videos_per_grid, (grid_idx + 1) * cfg.vis.num_videos_per_grid, 1)
            camera_params = generate_camera_params(G, z[grid_slice], c[grid_slice], cfg.trajectory)
            grid_videos = generate_trajectory(cfg, G, ws[grid_slice], camera_params=camera_params, verbose=False, **cfg.synthesis_kwargs) # [num_frames, num_videos, c, h, w]
            curr_names = sample_names[grid_slice] # [num_videos_per_grid]
            video_frames = torch.stack([make_grid(g, nrow=int(np.ceil(min(cfg.vis.num_videos_per_grid, len(sample_names)) ** 0.5)) if cfg.vis.nrow == 'auto' else cfg.vis.nrow) for g in grid_videos]) # [t, c, gh, gw]
            video = (video_frames * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
            save_path = os.path.join(save_dir, f'{curr_names[0]}-{curr_names[-1]}.{"gif" if cfg.vis.as_gif else "mp4"}')
            if cfg.vis.as_gif:
                frames = [Image.fromarray(x, 'RGB') for x in video.numpy()]
                frames[0].save(save_path, quality=75, save_all=True, append_images=frames[1:], duration=1000/cfg.vis.fps, loop=0)
            else:
                tv.io.write_video(save_path, video, fps=cfg.vis.fps, video_codec='h264', options={'crf': '10'})
    else:
        raise NotImplementedError(f"Unknown vis_type: {cfg.vis.name}")

#----------------------------------------------------------------------------

def sample_z_from_seeds(seeds: List[int], z_dim: int) -> torch.Tensor:
    zs = [np.random.RandomState(s).randn(1, z_dim) for s in seeds] # [num_samples, z_dim]
    return torch.from_numpy(np.concatenate(zs, axis=0)).float() # [num_samples, z_dim]

#----------------------------------------------------------------------------

def sample_c_from_seeds(seeds: List[int], c_dim: int, device: str='cpu') -> torch.Tensor:
    if c_dim == 0:
        return torch.empty(len(seeds), 0)
    c_idx = [np.random.RandomState(s).choice(np.arange(c_dim), size=1).item() for s in seeds] # [num_samples]
    return c_idx_to_c(c_idx, c_dim, device)

#----------------------------------------------------------------------------

def c_idx_to_c(c_idx: List[int], c_dim: int, device: str) -> torch.Tensor:
    c_idx = np.array(c_idx) # [num_samples, 1]
    c = np.zeros((len(c_idx), c_dim)) # [num_samples, c_dim]
    c[np.arange(len(c_idx)), c_idx] = 1.0

    return torch.from_numpy(c).float().to(device) # [num_samples, c_dim]

#----------------------------------------------------------------------------

def sample_ws_from_seeds(G, seeds: List[int], cfg: DictConfig, device: str, num_interp_steps: int=0,  classes: Optional[List[int]]=None):
    if num_interp_steps == 0:
        z = sample_z_from_seeds(seeds, G.z_dim).to(device) # [num_samples, z_dim]
        if classes is None:
            c = sample_c_from_seeds(seeds, G.c_dim, device=device) # [num_samples, c_dim]
        else:
            c = c_idx_to_c(classes, G.c_dim, device) # [num_samples, c_dim]

        if cfg.truncation_psi < 1.0 and G.c_dim > 0:
            num_samples_to_avg = 256
            z_for_avg = torch.randn(len(c) * num_samples_to_avg, G.z_dim, device=z.device) # [num_c * num_samples_to_avg, z_dim]
            c_for_avg = c.repeat_interleave(num_samples_to_avg, dim=0) # [num_c * num_samples_to_avg, c_dim]
            ws_for_avg = G.mapping(z_for_avg, c_for_avg) # [num_c * num_samples_to_avg, num_ws, w_dim]
            ws_for_avg = ws_for_avg.view(len(c), num_samples_to_avg, G.num_ws, G.cfg.w_dim) # [num_c, num_samples_to_avg, w_dim]
            ws_avg = ws_for_avg.mean(dim=1) # [num_c, num_ws, w_dim]
            if not classes is None:
                ws_avg = ws_avg.repeat_interleave(len(seeds), dim=0) # [num_classes * num_seeds, num_ws, w_dim]

        if not classes is None:
            z = z.repeat(len(c), 1) # [num_classes * num_seeds, z_dim]
            c = c.repeat_interleave(len(seeds), dim=0) # [num_classes * num_seeds, c_dim]

        if cfg.truncation_psi < 1.0 and G.c_dim > 0:
            ws = G.mapping(z, c=c) # [num_samples, num_ws, w_dim]
            ws = ws * cfg.truncation_psi + ws_avg * (1 - cfg.truncation_psi) # [num_samples, num_ws, w_dim]
        else:
            ws = G.mapping(z, c=c, truncation_psi=cfg.truncation_psi) # [num_samples, num_ws, w_dim]

        return ws, z, c
    else:
        assert classes is None
        z_from = sample_z_from_seeds(seeds[0::2], G.z_dim).to(device) # [num_samples, z_dim]
        z_to = sample_z_from_seeds(seeds[1::2], G.z_dim).to(device) # [num_samples, z_dim]
        c_from = sample_c_from_seeds(seeds[0::2], G.c_dim, device=device) # [num_samples, c_dim]
        c_to = sample_c_from_seeds(seeds[1::2], G.c_dim, device=device) # [num_samples, c_dim]
        ws_from = G.mapping(z_from, c=c_from, truncation_psi=cfg.truncation_psi) # [num_samples, num_ws, w_dim]
        ws_to = G.mapping(z_to, c=c_to, truncation_psi=cfg.truncation_psi) # [num_samples, num_ws, w_dim]
        alpha = torch.linspace(0, 1, num_interp_steps, device=device).view(num_interp_steps, 1, 1, 1) # [num_interp_steps]
        ws = ws_from.unsqueeze(0) * (1 - alpha) + ws_to.unsqueeze(0) * alpha # [num_interp_steps, num_samples, num_ws, w_dim]

        return ws, (z_from, z_to), (c_from, c_to)

#----------------------------------------------------------------------------

def save_images_as_grids(images: torch.Tensor, num_images_per_grid: int, img_names: List[str], save_dir: os.PathLike, nrow=None, save_ext: bool='.jpg'):
    save_kwargs = {'q': 95} if save_ext == '.jpg' else {}
    for grid_idx in tqdm(list(range(0, (len(images) + num_images_per_grid - 1) // num_images_per_grid)), desc='Saving'):
        grid_slice = slice(grid_idx * num_images_per_grid, (grid_idx + 1) * num_images_per_grid, 1)
        grid_images = images[grid_slice] # [num_images_per_grid, c, h, w]
        grid_img_names = img_names[grid_slice] # [num_images_per_grid]
        nrow = min(len(grid_images), int(np.sqrt(num_images_per_grid))) if nrow is None else nrow
        grid = make_grid(grid_images, nrow=nrow) # [c, grid_h, grid_w]
        TVF.to_pil_image(grid).save(os.path.join(save_dir, f'{grid_img_names[0]}-{grid_img_names[-1]}{save_ext}'), **save_kwargs)

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''
    Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    Copy-pasted from EG3D
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_vis() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------

