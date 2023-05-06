"""
Copyright Snap Inc. 2023. This sample code is made available by Snap Inc. for informational purposes only.
No license, whether implied or otherwise, is granted in or to such code (including any rights to copy, modify,
publish, distribute and/or commercialize such code), unless you have entered into a separate agreement for such rights.
Such code is provided as-is, without warranty of any kind, express or implied, including any warranties of merchantability,
title, fitness for a particular purpose, non-infringement, or that such code is free of defects, errors or viruses.
In no event will Snap Inc. be liable for any damages or losses of any kind arising from the sample code or your use thereof.
"""

import numpy as np
import torch
import torch.nn as nn
from src.torch_utils import persistence
from omegaconf import DictConfig

from src.training.layers import Conv2dLayer
from src.training.training_utils import linear_schedule

#----------------------------------------------------------------------------

@persistence.persistent_class
class DepthAdaptor(torch.nn.Module):
    def __init__(self, cfg: DictConfig, min_depth: float, max_depth: float):
        super().__init__()
        self.cfg = cfg
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_range = max_depth - min_depth
        dims = [1] + [self.cfg.hid_dim] * self.cfg.num_hid_layers # [num_hid_layers + 1]
        assert len(dims) >= 1, f"Too few dimensionalities: {dims}"
        self.layers = torch.nn.ModuleList([])
        for in_channels, out_channels in zip(dims[:-1], dims[1:]):
            self.layers.append(Conv2dLayer(in_channels, out_channels, self.cfg.kernel_size, activation='lrelu'))
        if len(self.layers) > 0:
            self.head = Conv2dLayer(dims[-1], 1, 1, activation='linear')
        else:
            self.head = None
        self.register_buffer('progress_coef', torch.tensor([0.0]))

        self.near_plane_offset_raw = torch.nn.Parameter(torch.tensor([self.cfg.near_plane_offset_bias]).float()) # [1]

    def get_near_plane_offset(self, w: torch.Tensor) -> torch.Tensor:
        near_plane_offset_raw = self.near_plane_offset_raw.repeat(len(w)) # [batch_size]
        near_plane_offset = near_plane_offset_raw.sigmoid() * self.cfg.near_plane_offset_max_fraction * self.depth_range # [batch_size]

        return near_plane_offset

    def normalize(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Normalizing the depth map to [-1, 1]
        # Re-positioning the near plane into the [near, near + max_frac * (far - near)] range using the offset
        near_shifted = self.min_depth + self.get_near_plane_offset(w) # [batch_size]
        near_shifted = near_shifted.view(len(x), 1, 1, 1) # [batch_size, 1, 1, 1]

        # Now we assume that all our depth values falls into the [near_shifted, far] range
        mid_depth_shifted = 0.5 * (self.max_depth + near_shifted) # [batch_size, 1, 1, 1]
        depth_range_contracted = self.max_depth - near_shifted # [batch_size, 1, 1, 1]
        x = (x - mid_depth_shifted) / (depth_range_contracted + 1e-12) * 2.0  # [batch_size, 1, h, w]

        return x

    def progressive_update(self, cur_kimg: float):
        self.progress_coef.data = torch.tensor(linear_schedule(cur_kimg, 0.0, 1.0, self.cfg.anneal_kimg)).to(self.progress_coef.device)

    @property
    def start_p(self):
        return (1.0 / (self.cfg.num_hid_layers + 1) * (1 - self.progress_coef) + self.cfg.selection_start_p * self.progress_coef).item()

    def forward(self, depth_map: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Params:
            - `depth_map` --- depth map of size [batch_size, 1, h, w]
        """
        x = self.normalize(depth_map, w)
        outs = [x] # [batch_size, 1, h, w]
        for layer in self.layers:
            x = layer(x) # [batch_size, hid_dim, h, w]
            outs.append(self.head(x))

        outs = torch.stack(outs).transpose(0, 1) # [batch_size, num_outs, 1, h, w]
        batch_size, num_outs = outs.shape[:2]
        if self.cfg.out_strategy == 'last':
            return outs[:, -1] + 0.0 * outs.max()
        elif self.cfg.out_strategy == 'mean':
            return outs.mean(dim=1)
        elif self.cfg.out_strategy == 'random':
            if self.training:
                # Generating the probabilities to choose each map (we try to reduce the pressure on the first one)
                out_idx = np.arange(num_outs) # [num_outs]
                sampling_slope = (1 - num_outs * self.start_p) * 2 / (num_outs * (num_outs - 1)) # [1]
                sampling_probs = out_idx * sampling_slope + self.start_p # [num_outs]
                random_idx = np.random.choice(out_idx, size=(batch_size,), p=sampling_probs) # [batch_size]
                random_idx = torch.from_numpy(random_idx)
            else:
                # Using the last depth map
                random_idx = (num_outs - 1) * torch.ones(batch_size, device=depth_map.device, dtype=torch.int64) # [batch_size]
            random_outs = outs[torch.arange(batch_size), random_idx] # [batch_size, 1, h, w]
            return random_outs + 0.0 * outs.max()
        else:
            raise NotImplementedError(f'Unknown out strategy: {self.cfg.out_strategy}')

#----------------------------------------------------------------------------
