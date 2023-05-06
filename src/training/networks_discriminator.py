from typing import Dict, List
import numpy as np
import torch
from omegaconf import DictConfig
from src.torch_utils import misc
from src.torch_utils import persistence
from src.torch_utils.ops import upfirdn2d

from src.training.layers import (
    FullyConnectedLayer,
    MappingNetwork,
    Conv2dLayer,
    ScalarEncoder1d,
)

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorBlock(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,                    # Main config.
        in_channels,                        # Number of input channels, 0 = first block.
        tmp_channels,                       # Number of intermediate channels.
        out_channels,                       # Number of output channels.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of input color channels.
        first_layer_idx,                    # Index of the first layer.
        activation          = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        freeze_layers       = 0,            # Freeze-D: Number of layers to freeze.
        down                = 2,            # Downsampling factor
        c_dim               = 0,            # Hyper-conditioning dimension
        hyper_mod              = False,        # Should we use hyper-cond in Conv2dLayer?
    ):
        assert in_channels in [0, tmp_channels]
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.first_layer_idx = first_layer_idx
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.num_layers = 0
        def trainable_gen():
            while True:
                layer_idx = self.first_layer_idx + self.num_layers
                trainable = (layer_idx >= freeze_layers)
                self.num_layers += 1
                yield trainable
        trainable_iter = trainable_gen()

        self.fromrgb = Conv2dLayer(img_channels, tmp_channels, kernel_size=1, activation=activation, c_dim=c_dim, hyper_mod=False,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)
        self.conv0 = Conv2dLayer(tmp_channels, tmp_channels, kernel_size=3, activation=activation, c_dim=c_dim, hyper_mod=False,
            trainable=next(trainable_iter), conv_clamp=conv_clamp, channels_last=self.channels_last)
        self.conv1 = Conv2dLayer(tmp_channels, out_channels, kernel_size=3, activation=activation, down=down, c_dim=c_dim, hyper_mod=hyper_mod,
            trainable=next(trainable_iter), resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last)
        self.skip = Conv2dLayer(tmp_channels, out_channels, kernel_size=1, bias=False, down=down, c_dim=c_dim, hyper_mod=False,
            trainable=next(trainable_iter), resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, c: torch.Tensor=None, force_fp32=False):
        if (x if x is not None else img).device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format

        # Input.
        if x is not None:
            x = x.to(dtype=dtype, memory_format=memory_format)

        # FromRGB.
        if self.in_channels == 0:
            img = img.to(dtype=dtype, memory_format=memory_format)
            y = self.fromrgb(img, c=c)
            x = x + y if x is not None else y

        # Main layers.
        y = self.skip(x, c=c, gain=np.sqrt(0.5))
        x = self.conv0(x, c=c)
        x = self.conv1(x, c=c, gain=np.sqrt(0.5))
        x = y.add_(x)

        assert x.dtype == dtype
        return x

    def extra_repr(self):
        return f'resolution={self.resolution:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        batch_size, C, H, W = x.shape
        with misc.suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(batch_size)).item() if self.group_size is not None else batch_size
        F = self.num_channels
        c = C // F
        num_groups = batch_size // G

        y = x.reshape(G, num_groups, F, c, H, W)    # [group_size, num_groups, F, c, H, W]  Split minibatch batch_size into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)                       # [group_size, num_groups, F, c, H, W]  Subtract mean over group.
        y = y.square().mean(dim=0)                  # [num_groups, F, c, H, W]              Calc variance over group.
        y = (y + 1e-8).sqrt()                       # [num_groups, F, c, H, W]              Calc stddev over group.
        y = y.mean(dim=[2,3,4])                     # [num_groups, F]                       Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)                  # [num_groups, F, 1, 1]                 Add missing dimensions.
        y = y.repeat(G, 1, H, W)                    # [batch_size, F, H, W]                 Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)                # [N, C + 1, H, W]                      Append to input as new channels.
        return x

    def extra_repr(self):
        return f'group_size={self.group_size}, num_channels={self.num_channels:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer, 0 = disable.
        activation          = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        feat_predict_dim    = 0,        # Dimensionality of the features D should predict.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), out_features=in_channels, activation=activation)
        self.out = FullyConnectedLayer(in_channels, out_features=(1 if cmap_dim == 0 else cmap_dim))
        if feat_predict_dim > 0:
            self.feat_out = torch.nn.Sequential(
                FullyConnectedLayer(in_channels * (resolution ** 2), out_features=in_channels, activation=activation),
                FullyConnectedLayer(in_channels, feat_predict_dim)
            )
        else:
            self.feat_out = None

    def forward(self, x, cmap, force_fp32=False, predict_feat: bool=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format

        # FromRGB.
        x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = x.flatten(1) # [batch_size, in_channels * (resolution ** 2)]
        f = self.feat_out(x) if predict_feat else None # [batch_size, feat_dim]
        x = self.fc(x) # [batch_size, in_channels]
        x = self.out(x) # [batch_size, 1 or cmap_dim]

        # Conditioning.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        assert x.dtype == dtype
        return x, f

    def extra_repr(self):
        return f'resolution={self.resolution:d}'

#----------------------------------------------------------------------------

class Discriminator(torch.nn.Module):
    def __init__(self,
        cfg: DictConfig,                # Main config.
        input_resolution,               # Input resolution.
        img_channels,                   # Number of input color channels.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()

        self.cfg = cfg
        assert self.cfg.num_additional_start_blocks >= 0, f'Cannot have negative amount of additional blocks: {self.cfg.num_additional_start_blocks}'
        self.img_resolution = input_resolution * (2 ** self.cfg.num_additional_start_blocks)
        self.img_resolution_log2 = int(np.log2(self.img_resolution))
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        self.img_channels = img_channels
        channels_dict = {res: min(int(self.cfg.cbase * self.cfg.fmaps) // res, self.cfg.cmax) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        if cmap_dim is None:
            cmap_dim = channels_dict[4]

        if self.cfg.patch.patch_params_cond > 0:
            self.scalar_enc = ScalarEncoder1d(coord_dim=3, x_multiplier=1000.0, const_emb_dim=256)
            assert self.scalar_enc.get_dim() > 0
        else:
            self.scalar_enc = None

        if (self.cfg.c_dim == 0) and (self.scalar_enc is None) and (not self.cfg.camera_cond):
            cmap_dim = 0

        if self.cfg.hyper_mod:
            hyper_mod_dim = 512
            self.hyper_mod_mapping = MappingNetwork(
                z_dim=0, c_dim=self.scalar_enc.get_dim(), camera_cond=False, camera_cond_drop_p=0.0,
                w_dim=hyper_mod_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        else:
            self.hyper_mod_mapping = None
            hyper_mod_dim = 0

        common_kwargs = dict(img_channels=self.img_channels, conv_clamp=conv_clamp)
        total_conditioning_dim = self.cfg.c_dim + (0 if self.scalar_enc is None else self.scalar_enc.get_dim())
        cur_layer_idx = 0

        for i, res in enumerate(self.block_resolutions):
            in_channels = channels_dict[res] if res < self.img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            down = 1 if i < self.cfg.num_additional_start_blocks else 2
            block = DiscriminatorBlock(
                cfg, in_channels, tmp_channels, out_channels, resolution=res, first_layer_idx=cur_layer_idx, use_fp16=use_fp16,
                down=down, c_dim=hyper_mod_dim, hyper_mod=self.cfg.hyper_mod, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if self.cfg.c_dim > 0 or not self.scalar_enc is None:
            self.head_mapping = MappingNetwork(
                z_dim=0, c_dim=total_conditioning_dim, camera_cond=self.cfg.camera_cond, camera_cond_drop_p=self.cfg.camera_cond_drop_p,
                w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        else:
            self.head_mapping = None
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, patch_params: torch.Tensor=None, camera_angles: torch.Tensor=None, update_emas=False, predict_feat: bool=False, **block_kwargs):
        _ = update_emas # unused
        batch_size, _, h, w = img.shape

        if not self.scalar_enc is None:
            patch_scales = patch_params['scales'] # [batch_size, 2]
            patch_offsets = patch_params['offsets'] # [batch_size, 2]
            patch_params_cond = torch.cat([patch_scales[:, [0]], patch_offsets], dim=1) # [batch_size, 3]
            misc.assert_shape(patch_params_cond, [batch_size, 3])
            patch_scale_embs = self.scalar_enc(patch_params_cond) # [batch_size, fourier_dim]
            c = torch.cat([c, patch_scale_embs], dim=1) # [batch_size, c_dim + fourier_dim]

        if not self.hyper_mod_mapping is None:
            hyper_mod_c = self.hyper_mod_mapping(z=None, c=patch_scale_embs) # [batch_size, 512]
        else:
            hyper_mod_c = None

        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x = block(x, img, c=hyper_mod_c, **block_kwargs)

        if self.cfg.c_dim > 0 or not self.scalar_enc is None:
            assert c.shape[1] > 0
        if not self.head_mapping is None:
            cmap = self.head_mapping(z=None, c=c, camera_angles=camera_angles) # [TODO]
        else:
            cmap = None

        x, f = self.b4(x, cmap, predict_feat=predict_feat)
        x = x.squeeze(1) # [batch_size]
        misc.assert_shape(x, [batch_size])

        return x, f

    def extra_repr(self):
        return f'c_dim={self.self.cfg.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}'

#----------------------------------------------------------------------------
