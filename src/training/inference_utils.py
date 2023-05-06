import os
import textwrap
from typing import List, Optional

import torch
import torchvision as tv
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TVF

from src import dnnlib
from src.dnnlib import TensorGroup
from src.training.rendering_utils import sample_camera_params
from src.training.training_utils import sample_random_c

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, cfg, random_seed=0):
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    all_indices = list(range(len(training_set)))
    np.random.RandomState(random_seed).shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    # Load data.
    batch = [training_set[i] for i in grid_indices]
    images = [b['image'] for b in batch]
    depth = np.stack([b['depth'] for b in batch]) if cfg.training.use_depth else None
    labels = [b['label'] for b in batch]
    if cfg.camera.origin.angles.dist == 'custom':
        camera_angles = torch.from_numpy(np.array([b['camera_angles'] for b in batch]))
    else:
        camera_angles = None
    camera_params = sample_camera_params(cfg.camera, len(images), 'cpu', origin_angles=camera_angles) # [batch_size, ...]

    return (gw, gh), np.stack(images), depth, np.stack(labels), camera_params

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        Image.fromarray(img[:, :, 0], 'L').save(fname, q=95)
    if C == 3:
        Image.fromarray(img, 'RGB').save(fname, q=95)

#----------------------------------------------------------------------------

def generate_videos(G: torch.nn.Module, z: torch.Tensor, c: torch.Tensor, gen_depths: bool=False) -> torch.Tensor:
    num_videos = 9 if G.img_resolution >= 1024 else 16
    z, c = z[:num_videos], c[:num_videos], # [num_videos, z_dim], [num_videos, c_dim]
    traj_cfg = dnnlib.EasyDict({'name': 'front_circle', 'num_frames': 32, 'fov_diff': 1.0, 'yaw_diff': 0.5, 'pitch_diff': 0.3, 'use_mean_camera': True})
    vis_cfg = dnnlib.EasyDict({'batch_size': 4})
    camera_params = generate_camera_params(G, z, c, traj_cfg) # [num_samples * num_frames, ...]
    ws = G.mapping(z=z, c=c) # [num_videos, num_ws, w_dim]
    render_opts = dict(return_depth=gen_depths, return_depth_adapted=gen_depths)
    images = generate_trajectory(vis_cfg, G, ws, camera_params, verbose=False, render_opts=render_opts) # [num_frames, num_videos, c, h, w]
    images = images.permute(1, 0, 2, 3, 4) # [num_videos, num_frames, c, h, w]

    if not isinstance(images, TensorGroup):
        images = TensorGroup(img=images)

    return images

#----------------------------------------------------------------------------

def save_videos(videos: torch.Tensor, save_path: os.PathLike):
    grids = [tv.utils.make_grid(vs, nrow=int(np.sqrt(videos.shape[0]))) for vs in videos.permute(1, 0, 2, 3, 4)] # [num_frames, c, gh, gw]
    video = (torch.stack(grids) * 255).to(torch.uint8).permute(0, 2, 3, 1) # [T, H, W, C]
    tv.io.write_video(save_path, video, fps=25, video_codec='h264', options={'crf': '20'})

#----------------------------------------------------------------------------

def generate_trajectory(cfg, G, ws: torch.Tensor, camera_params: TensorGroup, **generate_kwargs):
    """Produces frames for all `ws` for each trajectory step"""
    num_cameras = len(camera_params) // len(ws) # [1]
    num_samples = len(camera_params) // num_cameras # [1]
    camera_params = camera_params.to(dtype=torch.float32, device=ws.device) # [num_samples * num_frames, ...]
    ws = ws.repeat_interleave(num_cameras, dim=0) # [num_samples * num_cameras, num_ws, w_dim]
    images = generate(cfg, G, ws=ws, camera_params=camera_params, **generate_kwargs) # [num_samples * num_cameras, c, h, w]
    if isinstance(images, TensorGroup):
        images = images.reshape_each(lambda x: [num_samples, num_cameras, *x.shape[1:]]) # [num_samples, num_cameras, c, h, w]
    else:
        images = images.reshape(num_samples, num_cameras, *images.shape[1:]) # [num_samples, num_cameras, c, h, w]
    images = images.permute(1, 0, 2, 3, 4) # [num_cameras, num_samples, c, h, w]

    return images

#----------------------------------------------------------------------------

def generate(cfg: DictConfig, G, ws: torch.Tensor, camera_params: TensorGroup, verbose: bool=True, **synthesis_kwargs):
    frames = []
    batch_indices = range(0, (len(ws) + cfg.batch_size - 1) // cfg.batch_size)
    batch_indices = tqdm(batch_indices, desc='Generating') if verbose else batch_indices
    for batch_idx in batch_indices:
        curr_slice = slice(batch_idx * cfg.batch_size, (batch_idx + 1) * cfg.batch_size)
        curr_ws, curr_camera_params = ws[curr_slice], camera_params[curr_slice] # [batch_size, num_ws, w_dim], [batch_size, 3]
        frame = G.synthesis(curr_ws, camera_params=curr_camera_params, noise_mode='const', **synthesis_kwargs) # [batch_size, c, h, w]
        if isinstance(frame, TensorGroup) and 'depth' in frame:
            depth_range = G.cfg.camera.ray.end - G.cfg.camera.ray.start
            depth_mid = (G.cfg.camera.ray.start + G.cfg.camera.ray.end) * 0.5
            frame.depth = (frame.depth - depth_mid) / depth_range * 2.0 # [batch_size, 1, h, w]
        frame = frame.clamp(-1, 1).cpu() * 0.5 + 0.5 # [batch_size, c, h, w]
        frames.append(frame)

    if isinstance(frames[0], TensorGroup):
        return TensorGroup.cat(frames, dim=0)
    else:
        return torch.cat(frames, dim=0) # [num_frames, c, h, w]

#----------------------------------------------------------------------------

def generate_camera_params(G, z: torch.Tensor, c: torch.Tensor, trajectory_cfg) -> TensorGroup:
    if trajectory_cfg.use_mean_camera:
        mean_camera_params = get_mean_camera_params(G, device=z.device) # [3]
        canonical_camera_params = mean_camera_params.repeat_interleave(len(z), dim=0) # [num_seeds, ...]
    else:
        canonical_camera_params = sample_posterior_camera_params(G, z, c) # [num_seeds, ...]
    return generate_camera_trajectory(trajectory_cfg, canonical_camera_params=canonical_camera_params) # [num_samples * num_frames, ...]

#----------------------------------------------------------------------------

def generate_camera_trajectory(trajectory, canonical_camera_params: TensorGroup) -> TensorGroup:
    """
    Generates camera trajectories for each canonical camera position
    """
    num_samples = len(canonical_camera_params)
    num_frames = len(trajectory.yaw_offsets) if trajectory.name == 'points' else trajectory.num_frames
    camera_params = canonical_camera_params.repeat_interleave(num_frames, dim=0) # [num_samples * num_frames, ...]

    if trajectory.name == 'point':
        assert num_frames == 1
        angles = camera_params.angles.cpu() + torch.tensor([trajectory.yaw_offset, trajectory.pitch_offset, 0.0]).unsqueeze(0) # [num_samples * num_frames, 3]
        fov = camera_params.fov.cpu() + trajectory.fov_offset # [1]
    elif trajectory.name == 'front_circle':
        steps = torch.linspace(0, 1, num_frames).repeat(num_samples) # [num_samples * num_frames]
        yaw = camera_params.angles[:, 0].cpu() + trajectory.yaw_diff * torch.sin(steps * 2 * np.pi) # [num_samples * num_frames]
        pitch = camera_params.angles[:, 1].cpu() + trajectory.pitch_diff * torch.cos(steps * 2 * np.pi) # [num_samples * num_frames]
        angles = torch.stack([yaw, pitch, camera_params.angles[:, 2].cpu()], dim=1) # [num_samples * num_frames, 3]
        fov = (camera_params.fov.cpu() + trajectory.fov_diff * torch.sin(steps * 2 * np.pi)) # [num_samples * num_frames]
    elif trajectory.name == 'points':
        yaw = camera_params.angles[:, 0].cpu() + torch.tensor(trajectory.yaw_offsets).repeat(num_samples) # [num_samples * num_frames]
        pitch = camera_params.angles[:, 1].cpu() + trajectory.pitch_offset # [num_samples * num_frames]
        angles = torch.stack([yaw, pitch, camera_params.angles[:, 2].cpu()], dim=1) # [num_samples * num_frames, 3]
        fov = camera_params.fov.cpu() # [num_samples * num_frames]
    elif trajectory.name == 'wiggle':
        yaws = np.linspace(trajectory.yaw_left, trajectory.yaw_right, num_frames) # [num_frames]
        pitches = trajectory.pitch_diff * np.cos(np.linspace(0, 1, num_frames) * 2 * np.pi) + np.pi/2 # [num_frames]
        angles = np.stack([yaws, pitches, np.zeros(yaws.shape)], axis=1) # [num_frames, 3]
        fov = camera_params.fov.cpu() # [num_frames]
    elif trajectory.name == 'line':
        yaws = torch.linspace(trajectory.yaw_start, trajectory.yaw_end, num_frames).repeat(num_samples) # [num_samples * num_frames]
        pitches = torch.linspace(trajectory.pitch_start, trajectory.pitch_end, num_frames).repeat(num_samples) # [num_samples * num_frames]
        angles = torch.stack([yaws, pitches, torch.zeros_like(yaws)], axis=1) # [num_samples * num_frames, 3]
        fov = camera_params.fov.cpu() if trajectory.fov is None else (torch.ones_like(camera_params.fov.cpu()) * trajectory.fov)  # [num_samples * num_frames]
    else:
        raise NotImplementedError(f'Unknown trajectory: {trajectory.name}')

    return TensorGroup(
        angles=angles,
        fov=fov + trajectory.get('fov_offset', 0.0),
        radius=camera_params.radius.cpu(),
        look_at=camera_params.look_at.cpu(),
    ) # [num_samples * num_frames, ...]

#----------------------------------------------------------------------------

def get_mean_camera_params(G: torch.nn.Module, device: torch.device):
    if G.cfg.camera.origin.angles.dist == 'custom':
        return TensorGroup(
            angles=G.mapping.mean_camera_params[[0,1,2]].unsqueeze(0),
            fov=G.mapping.mean_camera_params[[3]],
            radius=G.mapping.mean_camera_params[[4]],
            look_at=torch.zeros(1, 3, device=device),
        ).float()
    else:
        return approximate_mean_camera_params(G, num_samples=1024, device=device) # [1, ...]

#----------------------------------------------------------------------------

def approximate_mean_camera_params(G: "Generator", num_samples: int=1024, device: str='cpu') -> TensorGroup:
    camera_params_prior = sample_camera_params(G.cfg.camera, num_samples, device=device) # [num_samples, ...]
    if not G.cfg.camera_adaptor.enabled:
        camera_params_posterior = camera_params_prior # [num_samples, ...]
    else:
        z = torch.randn(num_samples, G.z_dim, device=device) # [num_samples, z_dim]
        c = sample_random_c(num_samples, G.c_dim, device=device) # [num_samples, c_dim]
        camera_params_posterior = G.synthesis.camera_adaptor(camera_params_prior, z, c) # [num_samples, ...]
    return camera_params_posterior.mean(dim=0, keepdim=True) # [1, ...]

#----------------------------------------------------------------------------

def sample_posterior_camera_params(G: "Generator", z: torch.Tensor, c: torch.Tensor) -> TensorGroup:
    camera_params_prior = sample_camera_params(G.cfg.camera, len(z), device=z.device) # [num_samples, ...]
    if not G.cfg.camera_adaptor.enabled:
        camera_params_posterior = camera_params_prior # [num_samples, ...]
    else:
        camera_params_posterior = G.synthesis.camera_adaptor(camera_params_prior, z, c) # [num_samples, ...]
    return camera_params_posterior

#----------------------------------------------------------------------------
