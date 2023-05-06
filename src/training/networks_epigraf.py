import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from src.torch_utils import misc
from src.torch_utils import persistence
from src.dnnlib import EasyDict, TensorGroup
from omegaconf import DictConfig

from src.training.networks_stylegan2 import SynthesisBlock
from src.training.networks_camera_adaptor import CameraAdaptor
from src.training.networks_depth_adaptor import DepthAdaptor
from src.training.layers import (
    FullyConnectedLayer,
    MappingNetwork,
    ScalarEncoder1d,
)
from src.training.rendering_utils import compute_cam2world_matrix
from src.training.training_utils import linear_schedule, run_batchwise
from src.training.tri_plane_renderer import sample_rays, ImportanceRenderer, simple_tri_plane_renderer

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriPlaneMLP(nn.Module):
    def __init__(self, cfg: DictConfig, out_dim: int):
        super().__init__()
        self.cfg = cfg
        self.out_dim = out_dim

        if self.cfg.tri_plane.mlp.n_layers == 0:
            assert self.cfg.tri_plane.feat_dim == (self.out_dim + 1), f"Wrong dims: {self.cfg.tri_plane.feat_dim}, {self.out_dim}"
            self.model = nn.Identity()
        else:
            self.backbone_out_dim = 1 + (self.cfg.tri_plane.mlp.hid_dim if self.cfg.has_view_cond else self.out_dim)
            self.dims = [self.cfg.tri_plane.feat_dim] + [self.cfg.tri_plane.mlp.hid_dim] * (self.cfg.tri_plane.mlp.n_layers - 1) + [self.backbone_out_dim] # (n_hid_layers + 2)
            activations = ['lrelu'] * (len(self.dims) - 2) + ['linear']
            assert len(self.dims) > 2, f"We cant have just a linear layer here: nothing to modulate. Dims: {self.dims}"
            layers = [FullyConnectedLayer(self.dims[i], self.dims[i+1], activation=a) for i, a in enumerate(activations)]
            self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: [batch_size, 3, num_points, feat_dim] --- volumetric features from tri-planes
            coords: [batch_size, num_points, 3] --- coordinates, assumed to be in [-1, 1]
            ray_d_world: [batch_size, num_points, 3] --- camera ray's view directions
        """
        batch_size, _, num_points, feat_dim = x.shape
        x = x.mean(dim=1).reshape(batch_size * num_points, feat_dim) # [batch_size * num_points, feat_dim]
        x = self.model(x) # [batch_size * num_points, out_dim]
        x = x.view(batch_size, num_points, self.backbone_out_dim) # [batch_size, num_points, backbone_out_dim]

        misc.assert_shape(x, [batch_size, num_points, self.out_dim + 1])

        # Uses sigmoid clamping from MipNeRF (copied from EG3D)
        if self.cfg.ray_marcher_type == 'mip':
            rgb = torch.sigmoid(x[..., :-1]) * (1 + 2 * 0.001) - 0.001 # [batch_size, num_points, out_dim]
        elif self.cfg.ray_marcher_type == 'classical':
            rgb = x[..., :-1] # [batch_size, num_points, out_dim]
        else:
            raise NotImplementedError(f"Unknown ray marcher: {self.cfg.ray_marcher_type}")

        return {'rgb': rgb, 'sigma': x[:, :, [-1]]}

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisBlocksSequence(torch.nn.Module):
    # A simpler verion of the SG2 SynthesisNetwork, which can also take some 2d tensor as input.
    # This is useful to build a 2D upsampler.
    def __init__(self,
        cfg: DictConfig,            # Hyperparameters config.
        in_resolution,              # Which resolution do we start with?
        out_resolution,             # Output image resolution.
        in_channels,                # Number of input channels.
        out_channels,               # Number of input channels.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert in_resolution == 0 or (in_resolution >= 4 and math.log2(in_resolution).is_integer())
        assert out_resolution >= 4 and math.log2(out_resolution).is_integer()
        assert in_resolution < out_resolution
        super().__init__()
        self.cfg = cfg
        self.out_resolution = out_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_fp16_res = num_fp16_res

        in_resolution_log2 = 2 if in_resolution == 0 else (int(np.log2(in_resolution)) + 1)
        out_resolution_log2 = int(np.log2(out_resolution))
        self.block_resolutions = [2 ** i for i in range(in_resolution_log2, out_resolution_log2 + 1)]
        out_channels_dict = {res: min(int(self.cfg.cbase * self.cfg.fmaps) // res, self.cfg.cmax) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (out_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for block_idx, res in enumerate(self.block_resolutions):
            cur_in_channels = out_channels_dict[res // 2] if block_idx > 0 else in_channels
            cur_out_channels = out_channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.out_resolution)
            block = SynthesisBlock(cur_in_channels, cur_out_channels, w_dim=self.cfg.w_dim, resolution=res,
                img_channels=self.out_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, x: torch.Tensor=None, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.cfg.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,            # Main config
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        **synthesis_seq_kwargs,     # Arguments of SynthesisBlocksSequence
    ):
        super().__init__()
        self.cfg = cfg
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        decoder_out_channels = self.cfg.tri_plane.feat_dim * 3

        self.tri_plane_decoder = SynthesisBlocksSequence(
            cfg=cfg,
            in_resolution=0,
            out_resolution=self.cfg.tri_plane.res,
            in_channels=0,
            out_channels=decoder_out_channels,
            architecture='skip',
            use_noise=self.cfg.use_noise,
            **synthesis_seq_kwargs,
        )
        self.tri_plane_mlp = TriPlaneMLP(self.cfg, out_dim=self.img_channels)
        self.num_ws = self.tri_plane_decoder.num_ws
        self.nerf_noise_std = 0.0
        self.train_resolution = self.cfg.patch.resolution if self.cfg.patch.enabled else self.img_resolution
        self.test_resolution = self.img_resolution
        self.renderer = ImportanceRenderer(ray_marcher_type=self.cfg.ray_marcher_type)

        if self.cfg.depth_adaptor.enabled:
            self.depth_adaptor = DepthAdaptor(
                self.cfg.depth_adaptor,
                min_depth=self.cfg.camera.ray.start,
                max_depth=self.cfg.camera.ray.end
            )
        else:
            self.depth_adaptor = None

        if self.cfg.camera_adaptor.enabled:
            self.camera_adaptor = CameraAdaptor(self.cfg.camera_adaptor)
        else:
            self.camera_adaptor = None

        # Rendering options used at test time.
        # We overwrite them when we need to compute some additional losses
        self._default_render_options = EasyDict(
            max_batch_res=self.cfg.max_batch_res,
            return_depth=False,
            return_depth_adapted=False,
            return_weights=False,
            concat_depth=False,
            cut_quantile=0.0,
            density_bias=self.cfg.density_bias,
        )

    def progressive_update(self, cur_kimg: float):
        self.nerf_noise_std = linear_schedule(cur_kimg, self.cfg.nerf_noise_std_init, 0.0, self.cfg.nerf_noise_kimg_growth)
        if not self.depth_adaptor is None:
            self.depth_adaptor.progressive_update(cur_kimg)

    @torch.no_grad()
    def compute_densities(self, ws: torch.Tensor, coords: torch.Tensor, max_batch_res: int=32, **block_kwargs) -> torch.Tensor:
        plane_feats = self.tri_plane_decoder(ws[:, :self.tri_plane_decoder.num_ws], **block_kwargs) # [batch_size, 3 * feat_dim, tp_h, tp_w]
        ray_d_world = torch.zeros_like(coords) # [batch_size, num_points, 3]
        rgb_sigma = run_batchwise(
            fn=simple_tri_plane_renderer,
            data=dict(coords=coords),
            batch_size=max_batch_res ** 3,
            dim=1,
            mlp=self.tri_plane_mlp, x=plane_feats, scale=self.cfg.camera.cube_scale,
        ) # [batch_size, num_coords, num_feats]

        return rgb_sigma['sigma'] # [batch_size, num_coords, 1]

    def forward(self, ws, camera_params: EasyDict[str, torch.Tensor], patch_params: Dict=None, render_opts: Dict={}, **block_kwargs):
        """
        ws: [batch_size, num_ws, w_dim] --- latent codes
        camera_params: EasyDict {angles: [batch_size, 3], fov: [batch_size], radius: [batch_size], look_at: [batch_size, 3]} --- camera parameters
        patch_params: Dict {scales: [batch_size, 2], offsets: [batch_size, 2]} --- patch parameters (when we do patchwise training)
        """
        render_opts = EasyDict(**{**self._default_render_options, **render_opts})
        batch_size, num_steps = ws.shape[0], self.cfg.num_ray_steps
        decoder_out = self.tri_plane_decoder(ws[:, :self.tri_plane_decoder.num_ws], **block_kwargs) # [batch_size, 3 * feat_dim, tp_h, tp_w]
        plane_feats = decoder_out[:, :3 * self.cfg.tri_plane.feat_dim].view(batch_size, 3, self.cfg.tri_plane.feat_dim, self.cfg.tri_plane.res, self.cfg.tri_plane.res) # [batch_size, 3, feat_dim, tp_h, tp_w]
        h = w = (self.train_resolution if self.training else self.test_resolution)
        tri_plane_out_dim = self.img_channels + 1
        nerf_noise_std = self.nerf_noise_std if self.training else 0.0

        c2w = compute_cam2world_matrix(camera_params) # [batch_size, 4, 4]
        ray_o_world, ray_d_world = sample_rays(c2w, fov=camera_params.fov, resolution=(h, w), patch_params=patch_params, device=ws.device)
        rendering_bounds_kwargs = dict(ray_start='auto', ray_end='auto') if self.cfg.use_full_box else dict(ray_start=self.cfg.camera.ray.start, ray_end=self.cfg.camera.ray.end)
        rendering_options = EasyDict(
            box_size=self.cfg.camera.cube_scale * 2,  num_proposal_steps=num_steps, clamp_mode='softplus',
            use_inf_depth=self.cfg.use_inf_depth, **rendering_bounds_kwargs, num_fine_steps=num_steps, density_noise=nerf_noise_std,
            last_back=self.cfg.dataset.last_back, white_back=self.cfg.dataset.white_back, max_batch_res=render_opts.max_batch_res,
            cut_quantile=render_opts.cut_quantile, density_bias=render_opts.density_bias)
        if self.training or (h <= render_opts.max_batch_res and w <= render_opts.max_batch_res):
            fg_feats, fg_depths, _fg_weights, _fg_final_transmittance = self.renderer(plane_feats, self.tri_plane_mlp, ray_o_world, ray_d_world, rendering_options)
        else:
            fg_feats, fg_depths, _fg_weights, _fg_final_transmittance = run_batchwise(
                fn=self.renderer, data=dict(ray_origins=ray_o_world, ray_directions=ray_d_world),
                # For some reason, cut_quantile fails with error on large tensors
                batch_size=2 ** 24 // (batch_size * self.cfg.num_ray_steps * 3) if render_opts.cut_quantile > 0 else (self.cfg.num_ray_steps * rendering_options.max_batch_res ** 2),
                dim=1, planes=plane_feats, decoder=self.tri_plane_mlp, rendering_options=rendering_options,
            )

        rendered_feats = fg_feats.reshape(batch_size, h, w, tri_plane_out_dim - 1).permute(0, 3, 1, 2).contiguous() # [batch_size, render_out_dim, h, w]
        img = rendered_feats[:, :self.img_channels] # [batch_size, img_channels, h, w]
        depth = fg_depths.reshape(batch_size, 1, h, w) # [batch_size, 1, h, w]

        if not self.depth_adaptor is None:
            depth_adapted = self.depth_adaptor(depth, ws[:, 0]) # [batch_size, 1, h, w]

            if render_opts.concat_depth:
                img = torch.cat([img, depth_adapted], dim=1) # [batch_size, c + 1, h, w]
            else:
                # To avoid potential DataParallel issues
                img = img + 0.0 * depth_adapted.max() # [batch_size, c, h, w]

        if render_opts.return_depth or render_opts.return_depth_adapted:
            out = TensorGroup(img=img)
            if render_opts.return_depth: out.depth = depth
            if render_opts.return_depth_adapted: out.depth_adapted = depth_adapted
            return out
        else:
            return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,            # Main config
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.cfg = cfg
        self.z_dim = self.cfg.z_dim
        self.c_dim = self.cfg.c_dim
        self.w_dim = self.cfg.w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(cfg=cfg, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=self.cfg.z_dim, c_dim=self.cfg.c_dim, w_dim=self.cfg.w_dim, num_ws=self.num_ws, camera_raw_scalars=True, num_layers=self.cfg.map_depth, **mapping_kwargs)

    def progressive_update(self, cur_kimg: float):
        self.synthesis.progressive_update(cur_kimg)

    def forward(self, z, c, camera_params, camera_angles_cond=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, camera_angles=camera_angles_cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        out = self.synthesis(ws, camera_params=camera_params, update_emas=update_emas, **synthesis_kwargs)
        return out

#----------------------------------------------------------------------------
