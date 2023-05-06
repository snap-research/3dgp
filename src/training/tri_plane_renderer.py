"""
All the stuff below is copy-pasted (with light changes) from https://github.com/NVlabs/eg3d
"""
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""
from typing import Dict, Union, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.dnnlib import TensorGroup, EasyDict
from src.torch_utils import persistence
from src.torch_utils import misc
from src.training.rendering_utils import normalize, compute_cam2world_matrix
from src.training.training_utils import run_batchwise

#----------------------------------------------------------------------------

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],

        # Original
        # [[1, 0, 0],
        #  [0, 0, 1],
        #  [0, 1, 0]],
        # [[0, 0, 1],
        #  [1, 0, 0],
        #  [0, 1, 0]]

        # Fixed (https://github.com/NVlabs/eg3d/issues/53):
        # [[0, 1, 0],
        #  [0, 0, 1],
        #  [1, 0, 0]],
        # [[0, 0, 1],
        #  [1, 0, 0],
        #  [0, 1, 0]]

        # Fixed (ours):
        [[1, 0, 0],
         [0, 0, 1],
         [0, 1, 0]],
        [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]]
    ], dtype=torch.float32)

#----------------------------------------------------------------------------

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

#----------------------------------------------------------------------------

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_size=None):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N * n_planes, C, H, W)
    coordinates = (2 / box_size) * coordinates # TODO: add specific box bounds
    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = F.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=True).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

#----------------------------------------------------------------------------

def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = F.grid_sample(grid.expand(batch_size, -1, -1, -1, -1),
                                                       coordinates.reshape(batch_size, 1, 1, -1, n_dims),
                                                       mode='bilinear', padding_mode='zeros', align_corners=False)
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H*W*D, C)
    return sampled_features

#----------------------------------------------------------------------------

@persistence.persistent_class
class ImportanceRenderer(torch.nn.Module):
    def __init__(self, ray_marcher_type: str):
        super().__init__()
        assert ray_marcher_type in ['classical', 'mip']
        self.ray_marcher_type = ray_marcher_type
        self.ray_marcher = ClassicalRayMarcher() if ray_marcher_type == 'classical' else MipRayMarcher2()
        self.plane_axes = generate_planes()

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options):
        # Notations: `t` space is the metric (euclidian) space,
        # `s` space is the *potentially* normalized (disparity) space.
        # We use `s` space = `t` space for normal ray marching.
        # And we use the disparity space for the `s` space if we use Mip-NeRF-360.
        self.plane_axes = self.plane_axes.to(ray_origins.device)
        s_to_t = lambda s: (s * rendering_options['ray_end'] + (1 - s) * rendering_options['ray_start']) # In Mip-NeRF-360 terminology
        s_near, s_far = 0.0, 1.0

        # Create stratified depth samples
        sdist_coarse = self.sample_stratified(ray_origins, ray_start=s_near, ray_end=s_far, num_proposal_steps=rendering_options['num_proposal_steps'])
        batch_size, num_rays, num_proposal_steps, _ = sdist_coarse.shape

        # Coarse Pass
        tdist_coarse = s_to_t(sdist_coarse) # [batch_size, num_rays, num_proposal_steps, 1]
        sample_coordinates = (ray_origins.unsqueeze(-2) + tdist_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, num_proposal_steps, -1).reshape(batch_size, -1, 3)

        out = self.run_model(planes, decoder, sample_coordinates, rendering_options)
        colors_coarse = out['rgb']
        densities_coarse = out['sigma']
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, num_proposal_steps, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, num_proposal_steps, 1)

        # Fine Pass
        if rendering_options['num_fine_steps'] > 0:
            _, _, weights, _ = self.ray_marcher(colors_coarse, densities_coarse, sdist_coarse, rendering_options)
            sdist_fine = self.sample_importance(sdist_coarse, weights, rendering_options['num_fine_steps'])
            tdist_fine = s_to_t(sdist_fine) # [batch_size, num_rays, num_fine_steps, 1]
            sample_coordinates = (ray_origins.unsqueeze(-2) + tdist_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, rendering_options)
            colors_fine = out['rgb']
            densities_fine = out['sigma']
            colors_fine = colors_fine.reshape(batch_size, num_rays, rendering_options['num_fine_steps'], colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, rendering_options['num_fine_steps'], 1)

            all_depths, all_colors, all_densities = self.unify_samples(tdist_coarse, colors_coarse, densities_coarse, tdist_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights, final_transmittance = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights, final_transmittance = self.ray_marcher(colors_coarse, densities_coarse, sdist_coarse, rendering_options)

        return rgb_final, depth_final, weights.sum(2), final_transmittance

    def run_model(self, planes, decoder, sample_coordinates, rendering_options):
        if self.training:
            out = simple_tri_plane_renderer(planes.view(len(planes), 3 * planes.shape[2], planes.shape[3], planes.shape[4]), sample_coordinates, decoder, scale=rendering_options['box_size'] / 2)
        else:
            out = run_batchwise(
                simple_tri_plane_renderer,
                data=dict(coords=sample_coordinates),
                dim=1,
                batch_size=rendering_options['num_proposal_steps'] * rendering_options['max_batch_res'] ** 2, # Split into MxM rays
                x=planes.view(len(planes), 3 * planes.shape[2], planes.shape[3], planes.shape[4]),
                mlp=decoder,
                scale=rendering_options['box_size'] / 2,
            )
        if rendering_options.get('density_noise', 0.0) > 0.0:
            out['sigma'] += torch.randn_like(out['sigma']) * rendering_options['density_noise']
        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_densities = torch.cat([densities1, densities2], dim = -2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, num_proposal_steps, disparity_sampling=False):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_sampling:
            sdist_coarse = torch.linspace(0, 1, num_proposal_steps, device=ray_origins.device).reshape(1, 1, num_proposal_steps, 1).repeat(N, M, 1, 1) # [batch_size, num_rays, num_steps, 1]
            sdist_delta = 1.0 / (num_proposal_steps - 1) # [1]
            sdist_coarse += torch.rand_like(sdist_coarse) * sdist_delta # [batch_size, num_rays, num_steps, 1]
            sdist_coarse = 1.0 / (1.0 / ray_start * (1.0 - sdist_coarse) + 1.0 / ray_end * sdist_coarse) # [batch_size, num_rays, num_steps, 1]
        else:
            if type(ray_start) == torch.Tensor:
                sdist_coarse = linspace(ray_start, ray_end, num_proposal_steps).permute(1,2,0,3)
                sdist_delta = (ray_end - ray_start) / (num_proposal_steps - 1)
                sdist_coarse += torch.rand_like(sdist_coarse) * sdist_delta[..., None]
            else:
                sdist_coarse = torch.linspace(ray_start, ray_end, num_proposal_steps, device=ray_origins.device).reshape(1, 1, num_proposal_steps, 1).repeat(N, M, 1, 1)
                noise = torch.rand_like(sdist_coarse) # [batch_size, h * w, num_steps, 1]
                if self.ray_marcher_type == 'classical':
                    mids = 0.5 * (sdist_coarse[:, :, 1:, :] + sdist_coarse[:, :, :-1, :]) # [batch_size, h * w, num_steps - 1, 1]
                    upper = torch.cat([mids, sdist_coarse[:, :, -1:, :]], dim=2) # [batch_size, h * w, num_steps, 1]
                    lower = torch.cat([sdist_coarse[:, :, :1, :], mids], dim=2) # [batch_size, h * w, num_steps, 1]
                    sdist_coarse = lower + (upper - lower) * noise # [batch_size, h * w, num_steps, 1]
                else:
                    sdist_delta = (ray_end - ray_start) / (num_proposal_steps - 1)
                    sdist_coarse = sdist_coarse + noise * sdist_delta

        return sdist_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape
            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1) # -1 to account for loss of 1 sample in MipRayMarcher

            if self.ray_marcher_type == 'mip':
                # smooth weights
                weights = F.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
                weights = F.avg_pool1d(weights, 2, 1).squeeze()
                weights = weights + 0.01
            else:
                weights = weights + 1e-5
            z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
            importance_z_vals = self.sample_pdf(z_vals_mid, weights[:, 1:-1], N_importance).detach().reshape(batch_size, num_rays, N_importance, 1)
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
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
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[: , :1]), cdf], -1)  # (N_rays, N_samples_+1), padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds-1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[...,1]-cdf_g[...,0]
        denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                             # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
        return samples

#----------------------------------------------------------------------------

@persistence.persistent_class
class MipRayMarcher2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1] # [batch_size, h * w, num_steps - 1, 1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2 # [batch_size, h * w, num_steps - 1, 3]
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2 # [batch_size, h * w, num_steps - 1, 1]
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2 # [batch_size, h * w, num_steps - 1, 1]

        if rendering_options.use_inf_depth:
            deltas_last = 1e10 * torch.ones_like(deltas[:, :, [0]]) # [batch_size, h * w, 1, 1]
            deltas = torch.cat([deltas, deltas_last], dim=2) # [batch_size, h * w, num_steps, 1]

            # Using the last color values for this infinite depth
            colors_mid = torch.cat([colors_mid, colors[:, :, [-1]]], dim=2) # [batch_size, h * w, num_steps, 3]
            densities_mid = torch.cat([densities_mid, densities[:, :, [-1]]], dim=2) # [batch_size, h * w, num_steps, 3]
            depths_mid = torch.cat([depths_mid, depths[:, :, [-1]]], dim=2) # [batch_size, h * w, num_steps, 3]

        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid + rendering_options['density_bias'])
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        if rendering_options['cut_quantile'] > 0.0:
            assert rendering_options['cut_quantile'] <= 1.0, f"Wrong cut_quantile argument: {rendering_options['cut_quantile']}"
            densities_mid[densities_mid < torch.quantile(densities_mid, rendering_options['cut_quantile'])] = 0.0

        density_delta = densities_mid * deltas
        alpha = 1 - torch.exp(-density_delta)

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1 - alpha + 1e-10], -2)
        transmittance = torch.cumprod(alpha_shifted, -2)
        final_transmittance = transmittance[:, :, -1].squeeze(2) # [batch_size, h * w]
        weights = alpha * transmittance[:, :, :-1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        depth = torch.sum(weights * depths_mid, -2)
        # if not rendering_options.use_inf_depth:
        #     # Sometimes, we have too little density and the depth map is smaller than ray_start.
        #     depth = depth / (weight_total + 1e-12)

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2.0 - 1.0 # Scale to (-1, 1) to account for the sigmoid

        return composite_rgb, depth, weights, final_transmittance

#----------------------------------------------------------------------------

@persistence.persistent_class
class ClassicalRayMarcher(nn.Module):
    def forward(self, colors, densities, depths, rendering_options: EasyDict):
        deltas = depths[:, :, 1:] - depths[:, :, :-1] # [batch_size, h * w, num_steps - 1, 1]
        deltas_last = (1e10 if rendering_options.get('use_inf_depth', True) else 1e-3) * torch.ones_like(deltas[:, :, [0]]) # [batch_size, h * w, 1, 1]
        deltas = torch.cat([deltas, deltas_last], dim=2) # [batch_size, h * w, num_steps, 1]

        if rendering_options.clamp_mode == 'softplus':
            densities = F.softplus(densities, beta=rendering_options.get('sp_beta', 1.0)) # [batch_size, h * w, num_steps, 1]
        elif rendering_options.clamp_mode == 'relu':
            densities = F.relu(densities) # [batch_size, h * w, num_steps, 1]
        else:
            raise NotImplementedError(f"Uknown clamp mode: {rendering_options.clamp_mode}")

        if rendering_options['cut_quantile'] > 0.0:
            assert rendering_options['cut_quantile'] <= 1.0, f"Wrong cut_quantile argument: {rendering_options['cut_quantile']}"
            densities[densities < torch.quantile(densities, rendering_options['cut_quantile'])] = 0.0

        # Alpha is in the [0,1] range and represents opacity, i.e. "0" = "air" and "1" = "wall"
        alphas = 1.0 - torch.exp(-deltas * densities) # [batch_size, h * w, num_steps, 1]

        # Compute the accumulated transmittance (also in [0, 1]), i.e. how transparent the space is along the ray till the current point
        transmittance = torch.cumprod(1.0 - alphas + 1e-10, dim=2) # [batch_size, h * w, num_steps, 1]

        # Assume that the first point has nothing in front of it. We also do not need the transmittance
        # after the last point since we are not going to use any color after the last point
        transmittance = torch.cat([torch.ones_like(transmittance[:, :, [0], :]), transmittance], dim=2) # [batch_size, h * w, num_steps + 1, 1]
        final_transmittance = transmittance[:, :, -1].squeeze(2) # [batch_size, h * w]

        # Now we are ready to compute the weight for each color, which is equal to the opacity in the current point,
        # multiplied by the accumulated transmittence of the space in front of it. It is in the [0, 1] range as well
        weights = alphas * transmittance[:, :, :-1] # [batch_size, h * w, num_steps, 1]
        weights_agg = weights.sum(dim=2) # [batch_size, h * w, 1]

        if rendering_options.get('last_back', False):
            weights[:, :, -1] += (1.0 - weights_agg) # [batch_size, h * w, num_steps, 1]

        rgb_final = (weights * colors).sum(dim=2) # [batch_size, h * w, num_feats]
        depth = (weights * depths).sum(dim=2) # [batch_size, h * w, 1]

        if rendering_options.get('white_back_end_idx', 0) > 0:
            # Using white back for the first `white_back_end_idx` channels
            # (it's typically set to the number of image channels)
            rgb_final[:, :, :rendering_options.white_back_end_idx] = rgb_final[:, :, :rendering_options.white_back_end_idx] + 1 - weights_agg

        if rendering_options.get('fill_mode') == 'debug':
            num_colors = colors.shape[-1]
            red_color = torch.zeros(num_colors, device=colors.device)
            red_color[0] = 1.0
            rgb_final[weights_agg.squeeze(-1) < 0.9] = red_color
        elif rendering_options.get('fill_mode') == 'weight':
            rgb_final = weights_agg.expand_as(rgb_final)

        return rgb_final, depth, weights, final_transmittance

#----------------------------------------------------------------------------

def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_size):
    """
    Author: Petr Kellnhofer
    Intersects rays with the [-1, 1] NDC volume.
    Returns min and max distance of entry.
    Returns -1 for no intersection.
    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    Copy-pasted from https://github.com/NVlabs/eg3d
    """
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)

    bb_min = [-1*(box_size/2), -1*(box_size/2), -1*(box_size/2)]
    bb_max = [1*(box_size/2), 1*(box_size/2), 1*(box_size/2)]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)

    # Precompute inverse for stability.
    invdir = 1 / rays_d
    sign = (invdir < 0).long()

    # Intersect with YZ plane.
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]

    # Intersect with XZ plane.
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)

    # Intersect with XY plane.
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]

    # Resolve parallel rays.
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False

    # Use the shortest intersection.
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)

    # Mark invalid.
    tmin[torch.logical_not(is_valid)] = -1
    tmax[torch.logical_not(is_valid)] = -2

    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)

#----------------------------------------------------------------------------

def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    Copy-pasted from https://github.com/NVlabs/eg3d
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out

#----------------------------------------------------------------------------

def sample_rays(c2w: torch.Tensor, fov: Union[float, torch.Tensor], resolution: Tuple[int, int], patch_params: Dict=None, device: str=None):
    """
    Returns sample points, z_vals, and ray directions in camera space.

    If patch_scales/patch_offsets (of shape [batch_size, 2] each, for [0, 1] range) are provided,
    then will rescale the x/y plane accordingly to shoot rays into the desired region
    """
    batch_size = len(c2w)
    compute_batch_size = 1 if (patch_params is None and type(fov) is float) else batch_size # Batch size used for computations
    w, h = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    # Y is flipped to follow image memory layouts.
    x, y = torch.meshgrid(torch.linspace(-1, 1, w, device=device), torch.linspace(1, -1, h, device=device), indexing='ij')
    x = x.T.flatten().unsqueeze(0).repeat(compute_batch_size, 1) # [compute_batch_size, h * w]
    y = y.T.flatten().unsqueeze(0).repeat(compute_batch_size, 1) # [compute_batch_size, h * w]

    if not patch_params is None:
        patch_scales, patch_offsets = patch_params['scales'], patch_params['offsets']
        misc.assert_shape(patch_scales, [batch_size, 2])
        misc.assert_shape(patch_offsets, [batch_size, 2])
        # First, shift [-1, 1] range into [0, 2]
        # Then, multiply by the patch size
        # After that, shift back to [-1, 1]
        # Finally, apply the offset (converted from [0, 1] to [0, 2])
        x = (x + 1.0) * patch_scales[:, 0].view(batch_size, 1) - 1.0 + patch_offsets[:, 0].view(batch_size, 1) * 2.0 # [compute_batch_size, h * w]
        y = (y + 1.0) * patch_scales[:, 1].view(batch_size, 1) - 1.0 + patch_offsets[:, 1].view(batch_size, 1) * 2.0 # [compute_batch_size, h * w]

    fov = fov if isinstance(fov, torch.Tensor) else torch.tensor([fov], device=device) # [compute_batch_size]
    fov_rad = fov.unsqueeze(1).expand(compute_batch_size, 1) / 360 * 2 * np.pi # [compute_batch_size, 1]
    z = -torch.ones((compute_batch_size, h * w), device=device) / torch.tan(fov_rad * 0.5) # [compute_batch_size, h * w]
    ray_d_cam = normalize(torch.stack([x, y, z], dim=2), dim=2) # [compute_batch_size, h * w, 3]

    if compute_batch_size == 1:
        ray_d_cam = ray_d_cam.repeat(batch_size, 1, 1) # [batch_size, h * w, 3]

    ray_d_world = torch.bmm(c2w[..., :3, :3], ray_d_cam.reshape(batch_size, -1, 3).permute(0,2,1)).permute(0, 2, 1).reshape(batch_size, h * w, 3) # [batch_size, h * w, 3]
    homogeneous_origins = torch.zeros((batch_size, 4, h * w), device=device) # [batch_size, 4, h * w]
    homogeneous_origins[:, 3, :] = 1
    ray_o_world = torch.bmm(c2w, homogeneous_origins).permute(0, 2, 1).reshape(batch_size, h * w, 4)[..., :3] # [batch_size, h * w, 3]

    return ray_o_world, ray_d_world

#----------------------------------------------------------------------------

def validate_image_plane(fov: float, radius: float, scale: float=1.0, step: float=1e-2, device: str='cpu') -> bool:
    """
    Generates a lot of points on a hemisphere of radius `radius`,
    computes the corners of the viewing frustum
    and checks that all these corners are inside the [-1, 1]^3 cube
    """
    num_angles = int((np.pi / 2) / step) # [1]
    yaw = torch.linspace(0, np.pi * 2, steps=num_angles, device=device) # [num_angles]
    pitch = torch.linspace(0, np.pi, steps=num_angles, device=device) # [num_angles]
    yaw, pitch = torch.meshgrid(yaw, pitch, indexing='ij') # [num_angles, num_angles], [num_angles, num_angles]
    pitch = torch.clamp(pitch, 1e-7, np.pi - 1e-7)
    roll = torch.zeros(yaw.shape, device=device) # [num_angles, num_angles]
    angles = torch.stack([yaw.reshape(-1), pitch.reshape(-1), roll.reshape(-1)], dim=1) # [num_angles * num_angles, 3]

    h = w = 2
    camera_params = TensorGroup(
        angles=angles,
        radius=torch.empty(len(angles), device=device).fill_(radius),
        fov=torch.empty(len(angles), device=device).fill_(fov),
        look_at=torch.zeros_like(angles),
    )
    c2w = compute_cam2world_matrix(camera_params) # [batch_size, 4, 4]
    ray_o_world, ray_d_world = sample_rays(c2w, fov=camera_params.fov, resolution=(h, w), patch_params=None, device=device)
    ray_start, ray_end = get_ray_limits_box(ray_o_world, ray_d_world, box_size=scale * 2)
    is_ray_valid = ray_end > ray_start
    return torch.all(is_ray_valid).item()

#----------------------------------------------------------------------------

def simple_tri_plane_renderer(x: torch.Tensor, coords: torch.Tensor, mlp: Callable, scale: float=1.0) -> torch.Tensor:
    """
    Computes RGB\sigma values from a tri-plane representation + MLP
    x: [batch_size, feat_dim * 3, h, w]
    coords: [batch_size, h * w * num_steps, 3]
    ray_d_world: [batch_size, h * w, 3] --- ray directions in the world coordinate system
    mlp: additional transform to apply on top of features
    scale: additional scaling of the coordinates
    """
    assert x.shape[1] % 3 == 0, f"We use 3 planes: {x.shape}"
    batch_size, raw_feat_dim, h, w = x.shape
    num_points = coords.shape[1]
    feat_dim = raw_feat_dim // 3
    misc.assert_shape(coords, [batch_size, None, 3])

    x = x.reshape(batch_size * 3, feat_dim, h, w) # [batch_size * 3, feat_dim, h, w]
    coords = coords / scale # [batch_size, num_points, 3]
    coords_2d = torch.stack([
        coords[..., [0, 1]], # x/y plane
        coords[..., [0, 2]], # x/z plane
        coords[..., [1, 2]], # y/z plane
    ], dim=1) # [batch_size, 3, num_points, 2]
    coords_2d = coords_2d.view(batch_size * 3, 1, num_points, 2) # [batch_size * 3, 1, num_points, 2]
    # assert ((coords_2d.min().item() >= -1.0 - 1e-8) and (coords_2d.max().item() <= 1.0 + 1e-8))
    x = F.grid_sample(x, grid=coords_2d, mode='bilinear', align_corners=True).view(batch_size, 3, feat_dim, num_points) # [batch_size, 3, feat_dim, num_points]
    x = x.permute(0, 1, 3, 2) # [batch_size, 3, num_points, feat_dim]
    x = mlp(x) # [batch_size, num_points, out_dim]

    return x

#----------------------------------------------------------------------------
