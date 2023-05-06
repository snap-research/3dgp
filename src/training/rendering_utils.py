"""
Volumetric rendering utils from pi-GAN generator
Adapted from https://github.com/marcoamonteiro/pi-GAN
"""

import random
from typing import List, Dict, Union, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.stats import truncnorm

from src.dnnlib import TensorGroup, EasyDict
from src.torch_utils import misc

#----------------------------------------------------------------------------

def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res

#----------------------------------------------------------------------------

def normalize(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return x / (torch.norm(x, dim=dim, keepdim=True))

#----------------------------------------------------------------------------

def perturb_points(z_vals):
    mids = 0.5 * (z_vals[:, :, 1:, :] + z_vals[:, :, :-1, :]) # [batch_size, h * w, num_steps - 1, 1]
    upper = torch.cat([mids, z_vals[:, :, -1:, :]], dim=2) # [batch_size, h * w, num_steps, 1]
    lower = torch.cat([z_vals[:, :, :1, :], mids], dim=2) # [batch_size, h * w, num_steps, 1]
    noise = torch.rand_like(z_vals) # [batch_size, h * w, num_steps, 1]
    z_vals = lower + (upper - lower) * noise # [batch_size, h * w, num_steps, 1]

    return z_vals

#----------------------------------------------------------------------------

@misc.profiled_function
def transform_points(z_vals, ray_directions, c2w: torch.Tensor, perturb: bool=True):
    """
    Samples a camera position and maps points in the camera space to the world space.
    points: [batch_size, h * w, num_steps, ?]
    c2w: camera-to-world matrix
    """
    batch_size, num_rays, num_steps, _ = z_vals.shape
    if perturb:
        z_vals = perturb_points(z_vals)
    points_homogeneous = torch.ones((batch_size, num_rays, num_steps, 4), device=z_vals.device)
    points_homogeneous[:, :, :, :3] = z_vals * ray_directions.unsqueeze(2) # [batch_size, h * w, num_steps, 3]

    # should be batch_size x 4 x 4 , batch_size x r^2 x num_steps x 4
    points_world = torch.bmm(c2w, points_homogeneous.reshape(batch_size, -1, 4).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, num_rays, num_steps, 4)
    ray_d_world = torch.bmm(c2w[..., :3, :3], ray_directions.reshape(batch_size, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, num_rays, 3)

    homogeneous_origins = torch.zeros((batch_size, 4, num_rays), device=z_vals.device)
    homogeneous_origins[:, 3, :] = 1
    ray_o_world = torch.bmm(c2w, homogeneous_origins).permute(0, 2, 1).reshape(batch_size, num_rays, 4)[..., :3]

    return points_world[..., :3], z_vals, ray_d_world, ray_o_world

#----------------------------------------------------------------------------

def sample_camera_angles(cfg: DictConfig, batch_size: int, device: str):
    """
    Samples batch_size random locations along a sphere. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """
    if cfg.dist == 'uniform':
        yaw = torch.rand((batch_size, 1), device=device) * (cfg.yaw.max - cfg.yaw.min) + cfg.yaw.min # [batch_size, 1]
        pitch = torch.rand((batch_size, 1), device=device) * (cfg.pitch.max - cfg.pitch.min) + cfg.pitch.min # [batch_size, 1]
    elif cfg.dist == 'normal':
        yaw = torch.randn((batch_size, 1), device=device) * cfg.yaw.std + cfg.yaw.mean
        pitch = torch.randn((batch_size, 1), device=device) * cfg.pitch.std + cfg.pitch.mean
    elif cfg.dist == 'truncnorm':
        yaw = sample_truncnorm((cfg.yaw.max + cfg.yaw.min) * 0.5, cfg.yaw.std, cfg.yaw.min, cfg.yaw.max, batch_size, device).unsqueeze(1) # [batch_size, 1]
        pitch = sample_truncnorm((cfg.pitch.max + cfg.pitch.min) * 0.5, cfg.pitch.std, cfg.pitch.min, cfg.pitch.max, batch_size, device).unsqueeze(1) # [batch_size, 1]
    elif cfg.dist == 'hybrid':
        if random.random() < 0.5:
            yaw = (torch.rand((batch_size, 1), device=device) - 0.5) * 2 * cfg.yaw.std * 2 + cfg.yaw.mean
            pitch = (torch.rand((batch_size, 1), device=device) - 0.5) * 2 * cfg.pitch.std * 2 + cfg.pitch.mean
        else:
            yaw = torch.randn((batch_size, 1), device=device) * cfg.yaw.std + cfg.yaw.mean
            pitch = torch.randn((batch_size, 1), device=device) * cfg.pitch.std + cfg.pitch.mean
    elif cfg.dist == 'spherical_uniform':
        yaw_range, yaw_center = cfg.yaw.max - cfg.yaw.min, 0.5 * (cfg.yaw.max + cfg.yaw.min) # [1], [1]
        pitch_range, pitch_center = cfg.pitch.max - cfg.pitch.min, 0.5 * (cfg.pitch.max + cfg.pitch.min) # [1], [1]
        yaw = (torch.rand((batch_size, 1), device=device) - 0.5) * yaw_range + yaw_center # [batch_size, 1]
        v = (torch.rand((batch_size, 1), device=device) - 0.5) * pitch_range + pitch_center # [batch_size, 1]
        v = torch.clamp(v / np.pi, 1e-5, 1 - 1e-5) # [batch_size, 1]
        pitch = torch.arccos(1 - 2 * v) # [batch_size, 1]
    else:
        raise NotImplementedError(f'Unknown distribution: {cfg.dist}')
        # Just use the mean.
        yaw = torch.ones((batch_size, 1), device=device, dtype=torch.float) * cfg.yaw.mean
        pitch = torch.ones((batch_size, 1), device=device, dtype=torch.float) * cfg.pitch.mean

    pitch = torch.clamp(pitch, 1e-5, np.pi - 1e-5)
    angles = torch.cat([yaw, pitch, torch.zeros_like(yaw)], dim=1) # [batch_size, 3]

    return angles

#----------------------------------------------------------------------------

def sample_in_ball(cfg: DictConfig, batch_size: int, device: str):
    """
    Samples points in a ball, parametrized with rotation/elevation and radius
    """
    angles = sample_camera_angles(cfg.angles, batch_size, device) # [batch_size, 3]
    radius = sample_bounded_scalar(cfg.radius, batch_size, device) # [batch_size]

    return torch.cat([angles[:, [0, 1]], radius.unsqueeze(1)], dim=1) # [batch_size, 3]

#----------------------------------------------------------------------------

def sample_bounded_scalar(cfg: DictConfig, batch_size: int, device: str):
    if cfg.dist == 'normal':
        assert cfg.std == 0.0, f"Scalar must be bounded"
        x = torch.empty(batch_size, device=device, dtype=torch.float32).fill_(cfg.mean) # [batch_size]
    elif cfg.dist == 'truncnorm':
        x = sample_truncnorm(cfg.mean, cfg.std, cfg.min, cfg.max, batch_size, device) # [batch_size]
    elif cfg.dist == 'uniform':
        x = torch.rand(batch_size, device=device) * (cfg.max - cfg.min) + cfg.min # [batch_size]
    else:
        raise NotImplementedError(f'Uknown distribution: {cfg.dist}')

    return x

#----------------------------------------------------------------------------

def sample_truncnorm(mean: float, std: float, min: float, max: float, batch_size: int, device: str) -> torch.Tensor:
    x_min_norm = (min - mean) / std # [1]
    x_max_norm = (max - mean) / std # [1]
    x = truncnorm.rvs(a=x_min_norm, b=x_max_norm, loc=mean, scale=std, size=(batch_size,)) # [batch_size]
    x = torch.from_numpy(x).float().to(device) # [batch_size]

    return x

#----------------------------------------------------------------------------

def sample_camera_params(cfg: DictConfig, batch_size: int, device: str='cpu', origin_angles: Optional[torch.Tensor]=None) -> TensorGroup:
    origin_angles = sample_camera_angles(cfg.origin.angles, batch_size, device) if origin_angles is None else origin_angles # [batch_size, 3]
    fov = sample_bounded_scalar(cfg.fov, batch_size, device) # [batch_size]
    radius = sample_bounded_scalar(cfg.origin.radius, batch_size, device) # [batch_size]
    look_at = sample_in_ball(cfg.look_at, batch_size, device) # [batch_size, 3]

    return TensorGroup(angles=origin_angles, fov=fov, radius=radius, look_at=look_at) # [batch_size, ...]

#----------------------------------------------------------------------------

def get_max_sampling_value(cfg: DictConfig) -> float:
    if cfg.dist == 'normal':
        return cfg.mean if cfg.std <= 1e-8 else float('inf')
    elif cfg.dist in ('truncnorm', 'uniform'):
        return cfg.max
    else:
        raise NotImplementedError(f'Uknown distribution: `{cfg.dist}`')

#----------------------------------------------------------------------------

def get_mean_sampling_value(cfg: DictConfig) -> float:
    if cfg.dist in ('normal', 'truncnorm'):
        return cfg.mean
    elif cfg.dist == 'uniform':
        return (cfg.max + cfg.min) / 2
    else:
        raise NotImplementedError(f'Uknown distribution: {cfg.dist}')

#----------------------------------------------------------------------------

def get_mean_angles_values(angles_cfg: Dict) -> List[float]:
    if angles_cfg.dist == 'spherical_uniform':
        return [(angles_cfg.yaw.max + angles_cfg.yaw.min) * 0.5, (angles_cfg.pitch.max + angles_cfg.pitch.min) * 0.5, 0.0]
    elif angles_cfg.dist == 'normal':
        return [angles_cfg.yaw.mean, angles_cfg.pitch.mean, 0.0]
    elif angles_cfg.dist in ('truncnorm', 'uniform'):
        return [(angles_cfg.yaw.max + angles_cfg.yaw.min) * 0.5, (angles_cfg.pitch.max + angles_cfg.pitch.min) * 0.5, 0.0]
    elif angles_cfg.dist == 'custom':
        raise ValueError('Cannot compute the mean value analytically for a custom angles distribution.')
    else:
        raise NotImplementedError(f'Uknown distribution: `{angles_cfg.dist}`')

#----------------------------------------------------------------------------

def compute_cam2world_matrix(camera_params: TensorGroup) -> torch.Tensor:
    """
    Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix.
    camera_params:
        - angles: [batch_size, 3] — yaw/pitch/roll angles
        - radius: [batch_size]
        - look_at: [batch_size, 3] — rotation/elevation/radius of the look-at point.
    """
    origins = spherical2cartesian(camera_params.angles[:, 0], camera_params.angles[:, 1], camera_params.radius) # [batch_size, 3]
    look_at = spherical2cartesian(camera_params.look_at[:, 0], camera_params.look_at[:, 1], camera_params.look_at[:, 2]) # [batch_size, 3]
    forward_vector = normalize(look_at - origins) # [batch_size, 3]
    batch_size = forward_vector.shape[0]
    forward_vector = normalize(forward_vector) # [batch_size, 3]
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=forward_vector.device).expand_as(forward_vector) # [batch_size, 3]
    left_vector = normalize(torch.cross(up_vector, forward_vector, dim=-1))
    up_vector = normalize(torch.cross(forward_vector, left_vector, dim=-1))
    rotation_matrix = torch.eye(4, device=forward_vector.device).unsqueeze(0).repeat(batch_size, 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=forward_vector.device).unsqueeze(0).repeat(batch_size, 1, 1)
    translation_matrix[:, :3, 3] = origins

    cam2world = translation_matrix @ rotation_matrix

    return cam2world

#----------------------------------------------------------------------------

@misc.profiled_function
def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) --- padded to [0, 1] inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[..., 1] - cdf_g[..., 0]

    # denom equals 0 means a bin has weight 0, in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)
    denom[denom < eps] = 1

    samples = bins_g[..., 0] + (u - cdf_g[...,0]) / denom * (bins_g[..., 1] - bins_g[..., 0])

    return samples

#----------------------------------------------------------------------------

def spherical2cartesian(rotation: torch.Tensor, pitch: torch.Tensor, radius: Union[torch.Tensor, float]=1.0) -> torch.Tensor:
    """
    Converts spherical coordinates to cartesian: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    Rotation must be in [0, 2*pi]
    Pitch must be in [0, pi]
    """
    assert rotation.ndim == pitch.ndim, f"Wrong shapes: {rotation.shape}, {pitch.shape}"
    assert len(rotation) == len(pitch), f"Wrong shapes: {rotation.shape}, {pitch.shape}"

    # These equations reflect our camera conventions. Change with care.
    x = radius * torch.sin(pitch) * torch.sin(-rotation) # [..., batch_size]
    y = radius * torch.cos(pitch) # [..., batch_size]
    z = radius * torch.sin(pitch) * torch.cos(rotation) # [..., batch_size]
    coords = torch.stack([x, y, z], dim=-1) # [..., batch_size, 3]

    return coords

#----------------------------------------------------------------------------

def validate_frustum(fov: float, near: float, far: float, radius: float, scale: float=1.0, step: float=1e-2, device: str='cpu', verbose: bool=False) -> bool:
    """
    Generates a lot of points on a hemisphere of radius `radius`,
    computes the corners of the viewing frustum
    and checks that all these corners are inside the [-1, 1]^3 cube
    """
    # Step 1: sample the angles
    num_angles = int((np.pi / 2) / step) # [1]
    yaw = torch.linspace(0, np.pi * 2, steps=num_angles, device=device) # [num_angles]
    pitch = torch.linspace(0, np.pi, steps=num_angles, device=device) # [num_angles]
    yaw, pitch = torch.meshgrid(yaw, pitch, indexing='ij') # [num_angles, num_angles], [num_angles, num_angles]
    pitch = torch.clamp(pitch, 1e-7, np.pi - 1e-7)
    roll = torch.zeros(yaw.shape, device=device) # [num_angles, num_angles]
    angles = torch.stack([yaw.reshape(-1), pitch.reshape(-1), roll.reshape(-1)], dim=1) # [num_angles * num_angles, 3]
    batch_size = angles.shape[0]

    # Step 3: generating rays
    h = w = 2
    num_steps = 2
    x, y = torch.meshgrid(torch.linspace(-1, 1, w, device=device), torch.linspace(1, -1, h, device=device), indexing='ij')
    x = x.T.flatten().unsqueeze(0).repeat(batch_size, 1) # [batch_size, h * w]
    y = y.T.flatten().unsqueeze(0).repeat(batch_size, 1) # [batch_size, h * w]

    fov_rad = fov / 360 * 2 * np.pi # [1]
    z = -torch.ones((batch_size, h * w), device=device) / np.tan(fov_rad * 0.5) # [batch_size, h * w]
    rays_d_cam = normalize(torch.stack([x, y, z], dim=2), dim=2) # [batch_size, h * w, 3]

    z_vals = torch.linspace(near, far, num_steps, device=device).reshape(1, 1, num_steps, 1).repeat(batch_size, h * w, 1, 1) # [batch_size, h * w, num_steps, 1]
    camera_params = TensorGroup(
        angles=angles,
        radius=torch.empty(len(angles), device=device).fill_(radius),
        fov=torch.empty(len(angles), device=device).fill_(fov),
        look_at=torch.zeros_like(angles),
    )
    c2w = compute_cam2world_matrix(camera_params) # [batch_size, 4, 4]
    points_world, _, _, _ = transform_points(z_vals=z_vals, ray_directions=rays_d_cam, c2w=c2w)

    if verbose:
        print('min/max coordinates for the near plane', points_world[:, :, 0].min().item(), points_world[:, :, 0].max().item())
        print('min/max coordinates for the far plane', points_world[:, :, 1].min().item(), points_world[:, :, 1].max().item())
        print('min/max coordinates total', points_world.min().item(), points_world.max().item())

    return points_world.min().item() >= -scale and points_world.max().item() <= scale

#----------------------------------------------------------------------------

def compute_viewing_frustum_sizes(ray_start: float, ray_end: float, fov: float) -> EasyDict:
    """
    Computes the information about the viewing frustum dimensions. Might be useful for debugging.
    Assumes FoV in degrees.
    """
    return EasyDict(
        altitute=ray_end - ray_start,
        bottom_base=ray_end * np.deg2rad(fov),
        top_base=ray_start * np.deg2rad(fov),
    )

#----------------------------------------------------------------------------

