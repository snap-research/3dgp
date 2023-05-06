"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.dnnlib import TensorGroup
from src.dnnlib.util import EasyDict
from src.torch_utils import persistence
from src.training.layers import FullyConnectedLayer, normalize_2nd_moment
from src.training.rendering_utils import sample_camera_params
from src.training.training_utils import linear_schedule


@persistence.persistent_class
class ParamsAdaptor(nn.Module):
    def __init__(self, cfg, in_channels, out_channels, use_z: bool=True):
        super().__init__()
        self.cfg = cfg
        self.project_params = FullyConnectedLayer(in_channels, self.cfg.hid_dim, activation='softplus', lr_multiplier=self.cfg.lr_multiplier)
        if use_z:
            self.project_z = FullyConnectedLayer(self.cfg.z_dim, self.cfg.embed_dim, activation='softplus', lr_multiplier=self.cfg.lr_multiplier)
        else:
            self.project_z = None
        if self.cfg.c_dim > 0:
            self.project_c = FullyConnectedLayer(self.cfg.c_dim, self.cfg.embed_dim, activation='softplus', lr_multiplier=self.cfg.lr_multiplier)
        else:
            self.project_c = None
        main_in_channels = self.cfg.hid_dim + (self.cfg.embed_dim if use_z else 0) + (self.cfg.embed_dim if self.cfg.c_dim > 0 else 0)
        self.main = nn.Sequential(
            FullyConnectedLayer(main_in_channels, self.cfg.hid_dim, activation='softplus', lr_multiplier=self.cfg.lr_multiplier),
            FullyConnectedLayer(self.cfg.hid_dim, out_channels, activation='linear', lr_multiplier=self.cfg.lr_multiplier),
        )

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor]=None, c: Optional[torch.Tensor]=None):
        x = self.project_params(x)
        if not self.project_z is None:
            z = normalize_2nd_moment(self.project_z(z)) # [batch_size, hid_dim]
            x = torch.cat([x, z], dim=1) + 0.0 * z.max() # [batch_size, num_camera_params + hid_dim]
        if not self.project_c is None:
            c = normalize_2nd_moment(self.project_c(c)) # [batch_size, hid_dim]
            x = torch.cat([x, c], dim=1) + 0.0 * c.max() # [batch_size, num_camera_params + 2 * hid_dim]
        return self.main(x)

@persistence.persistent_class
class CameraAdaptor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_origin_cam_params = 4 # (yaw, pitch, roll, radius)
        self.num_look_at_cam_params = 4 # (fov, lookat_yaw, lookat_pitch, lookat_radius)
        self.num_cam_params = self.num_origin_cam_params + self.num_look_at_cam_params
        self.origin_adaptor = ParamsAdaptor(self.cfg, self.num_origin_cam_params, self.num_origin_cam_params, use_z=False)
        self.look_at_adaptor = ParamsAdaptor(self.cfg, self.num_cam_params, self.num_look_at_cam_params)

    def sample_from_prior(self, *args, **kwargs):
        return sample_camera_params(self.cfg.camera, *args, **kwargs)

    @staticmethod
    def unroll_camera_params(cp: TensorGroup) -> torch.Tensor:
        return torch.cat([cp.angles, cp.fov.unsqueeze(1), cp.radius.unsqueeze(1), cp.look_at], dim=1) # [N, 8]

    @staticmethod
    def roll_camera_params(cp: torch.Tensor) -> TensorGroup:
        return TensorGroup(angles=cp[:, [0, 1, 2]], fov=cp[:, 3], radius=cp[:, 4], look_at=cp[:, [5, 6, 7]]) # [N, ...]

    @staticmethod
    def normalize_camera_params(camera_cfg: EasyDict, cp: TensorGroup, eps: float=1e-8) -> TensorGroup:
        yaw, pitch, roll, fov, radius, la_yaw, la_pitch, la_radius = CameraAdaptor.unroll_camera_params(cp).split(1, dim=1)
        yaw = (yaw - camera_cfg.origin.angles.yaw.min) / (camera_cfg.origin.angles.yaw.max - camera_cfg.origin.angles.yaw.min + eps) # [batch_size, 1]
        pitch = (pitch - camera_cfg.origin.angles.pitch.min) / (camera_cfg.origin.angles.pitch.max - camera_cfg.origin.angles.pitch.min + eps) # [batch_size, 1]
        fov = (fov - camera_cfg.fov.min) / (camera_cfg.fov.max - camera_cfg.fov.min + eps) # [batch_size, 1]
        la_yaw = (la_yaw - camera_cfg.look_at.angles.yaw.min) / (camera_cfg.look_at.angles.yaw.max - camera_cfg.look_at.angles.yaw.min + eps) # [batch_size, 1]
        la_pitch = (la_pitch - camera_cfg.look_at.angles.pitch.min) / (camera_cfg.look_at.angles.pitch.max - camera_cfg.look_at.angles.pitch.min + eps) # [batch_size, 1]
        la_radius = (la_radius - camera_cfg.look_at.radius.min) / (camera_cfg.look_at.radius.max - camera_cfg.look_at.radius.min + eps) # [batch_size, 1]
        cp_norm = torch.cat([yaw, pitch, roll, fov, radius, la_yaw, la_pitch, la_radius], dim=1) # [batch_size, 8]
        return CameraAdaptor.roll_camera_params(cp_norm)

    @staticmethod
    def denormalize_camera_params(camera_cfg: EasyDict, cp: TensorGroup) -> TensorGroup:
        yaw, pitch, roll, fov, radius, la_yaw, la_pitch, la_radius = CameraAdaptor.unroll_camera_params(cp).split(1, dim=1)
        yaw = yaw.sigmoid() * (camera_cfg.origin.angles.yaw.max - camera_cfg.origin.angles.yaw.min) + camera_cfg.origin.angles.yaw.min # [batch_size, 1]
        pitch = pitch.sigmoid() * (camera_cfg.origin.angles.pitch.max - camera_cfg.origin.angles.pitch.min - 2e-5) + camera_cfg.origin.angles.pitch.min + 1e-5 # [batch_size, 1]
        roll = roll * 0.0 # Set roll to zero
        fov = fov.sigmoid() * (camera_cfg.fov.max - camera_cfg.fov.min) + camera_cfg.fov.min # [batch_size, 1]
        la_yaw = la_yaw.sigmoid() * (camera_cfg.look_at.angles.yaw.max - camera_cfg.look_at.angles.yaw.min) + camera_cfg.look_at.angles.yaw.min # [batch_size, 1]
        la_pitch = la_pitch.sigmoid() * (camera_cfg.look_at.angles.pitch.max - camera_cfg.look_at.angles.pitch.min) + camera_cfg.look_at.angles.pitch.min # [batch_size, 1]
        la_radius = la_radius.sigmoid() * (camera_cfg.look_at.radius.max - camera_cfg.look_at.angles.pitch.min) + camera_cfg.look_at.angles.pitch.min # [batch_size, 1]
        cp_denorm = torch.cat([yaw, pitch, roll, fov, radius, la_yaw, la_pitch, la_radius], dim=1) # [batch_size, 8]
        return CameraAdaptor.roll_camera_params(cp_denorm)

    def adjust_for_prior(self, camera_params_old: TensorGroup, camera_params_new: TensorGroup) -> TensorGroup:
        if not self.cfg.adjust.angles:
            camera_params_new.angles = camera_params_old.angles + 0.0 * camera_params_new.angles # [batch_size, 3]
        if not self.cfg.adjust.radius:
            camera_params_new.radius = camera_params_old.radius + 0.0 * camera_params_new.radius # [batch_size]
        if not self.cfg.adjust.fov:
            camera_params_new.fov = camera_params_old.fov + 0.0 * camera_params_new.fov # [batch_size]
        if not self.cfg.adjust.look_at:
            camera_params_new.look_at = camera_params_old.look_at + 0.0 * camera_params_new.look_at # [batch_size, 3]

        return camera_params_new

    def compute_new_camera_params(self, camera_params_old_norm, z, c):
        origin_params = torch.cat([camera_params_old_norm.angles, camera_params_old_norm.radius.unsqueeze(1)], dim=1) # [batch_size, 4]
        origin_params_new = self.origin_adaptor(origin_params, c=c) # [batch_size, 4]
        look_at_input_params = torch.cat([origin_params_new[:, :3], camera_params_old_norm.fov.unsqueeze(1), origin_params_new[:, [3]], camera_params_old_norm.look_at], dim=1) # [batch_size, 8]
        look_at_params_new = self.look_at_adaptor(look_at_input_params, z, c) # [batch_size, num_look_at_params]
        camera_params_new_norm = self.roll_camera_params(torch.cat([
            origin_params_new[:, :3],
            look_at_params_new[:, [0]],
            origin_params_new[:, [3]],
            look_at_params_new[:, [1, 2, 3]],
        ], dim=1))
        if self.cfg.get('residual', False):
            camera_params_new_norm = camera_params_old_norm + camera_params_new_norm # [batch_size, ...]
        return camera_params_new_norm

    def forward(self, camera_params_old: TensorGroup, z: torch.Tensor, c: Optional[torch.Tensor]=None):
        # We'll be storing the results here
        camera_params_old_norm = self.normalize_camera_params(self.cfg.camera, camera_params_old)
        camera_params_new_norm = self.compute_new_camera_params(camera_params_old_norm, z, c) # [batch_size, ...]
        camera_params_new = self.denormalize_camera_params(self.cfg.camera, camera_params_new_norm) # [batch_size, ...]
        camera_params_new = self.adjust_for_prior(camera_params_old, camera_params_new)

        return camera_params_new
