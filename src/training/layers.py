from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.torch_utils import misc
from src.torch_utils import persistence
from src.torch_utils.ops import conv2d_resample
from src.torch_utils.ops import upfirdn2d
from src.torch_utils.ops import bias_act

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x: torch.Tensor, dim=1, eps=1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        activation         = 'linear', # Activation function: 'relu', 'lrelu', etc.
        bias               = True,     # Apply additive bias before the activation function?
        lr_multiplier      = 1,        # Learning rate multiplier.
        weight_init        = 1,        # Initial standard deviation of the weight tensor.
        bias_init          = 0,        # Initial value of the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) * (weight_init / lr_multiplier))
        bias_init = np.broadcast_to(np.asarray(bias_init, dtype=np.float32), [out_features])
        self.bias = torch.nn.Parameter(torch.from_numpy(bias_init / lr_multiplier)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        # Selecting the weights
        w = self.weight # [c_out, c_in]
        b = self.bias if not self.bias is None else None # [c_out]
        # Adjusting the scales
        w = w.to(x.dtype) * self.weight_gain # [c_out, c_in] or [batch_size, c_out, c_in]
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain
        # Applying the weights
        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                         # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                         # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                         # Intermediate latent (W) dimensionality.
        num_ws,                        # Number of intermediate latents to output, None = do not broadcast.
        num_layers         = 2,        # Number of mapping layers.
        embed_features     = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features     = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation         = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier      = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta         = 0.998,    # Decay for tracking the moving average of W during training, None = do not track.
        camera_cond        = False,    # Camera conditioning
        camera_cond_drop_p = 0.0,      # Camera conditioning dropout
        camera_raw_scalars = False,    # Should we use raw camera angles as input or preprocess them with Fourier Features?
        mean_camera_params = None,     # Average camera pose for use at test time.
    ):
        super().__init__()
        if camera_cond:
            if camera_raw_scalars:
                self.camera_scalar_enc = ScalarEncoder1d(coord_dim=2, x_multiplier=0.0, const_emb_dim=0, use_raw=True)
            else:
                self.camera_scalar_enc = ScalarEncoder1d(coord_dim=2, x_multiplier=64.0, const_emb_dim=0)
            c_dim = c_dim + self.camera_scalar_enc.get_dim()
            assert self.camera_scalar_enc.get_dim() > 0
        else:
            self.camera_scalar_enc = None

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta
        self.camera_cond_drop_p = camera_cond_drop_p

        if self.c_dim > 0:
            embed_features = w_dim if embed_features is None else embed_features
            self.embed = FullyConnectedLayer(self.c_dim, embed_features)
        else:
            assert embed_features is None or embed_features == 0, f"Cannot use embed_features={embed_features}, when c_dim={c_dim}"
            embed_features = 0

        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

        if not mean_camera_params is None:
            self.register_buffer('mean_camera_params', mean_camera_params)
        else:
            self.mean_camera_params = None

    def forward(self, z, c, camera_angles: torch.Tensor=None, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if (not self.camera_scalar_enc is None) and (not self.training) and (camera_angles is None):
            camera_angles = self.mean_camera_params[:3].unsqueeze(0).repeat(len(z), 1) # [batch_size, 3]

        if not self.camera_scalar_enc is None:
            # Using only yaw and pitch for conditioning (roll is always zero)
            camera_angles = camera_angles[:, [0, 1]] # [batch_size, 2]
            camera_angles = camera_angles.sign() * ((camera_angles.abs() % (2.0 * np.pi)) / (2.0 * np.pi)) # [batch_size, 2]
            camera_angles_embs = self.camera_scalar_enc(camera_angles) # [batch_size, fourier_dim]
            camera_angles_embs = F.dropout(camera_angles_embs, p=self.camera_cond_drop_p, training=self.training) # [batch_size, fourier_dim]
            c = torch.zeros(len(camera_angles_embs), 0, device=camera_angles_embs.device) if c is None else c # [batch_size, c_dim]
            c = torch.cat([c, camera_angles_embs], dim=1) # [batch_size, c_dim + angle_emb_dim]

        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if update_emas and self.w_avg_beta is not None:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

    def extra_repr(self):
        return f'z_dim={self.z_dim:d}, c_dim={self.c_dim:d}, w_dim={self.w_dim:d}, num_ws={self.num_ws:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                      # Number of input channels.
        out_channels,                     # Number of output channels.
        kernel_size,                      # Width and height of the convolution kernel.
        bias              = True,         # Apply additive bias before the activation function?
        activation        = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up                = 1,            # Integer upsampling factor.
        down              = 1,            # Integer downsampling factor.
        resample_filter   = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp        = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last     = False,        # Expect the input to have memory_format=channels_last?
        trainable         = True,         # Update the weights of this layer during training?
        c_dim             = 0,            # Passing c via re-normalization?
        hyper_mod         = False,        # Should we use hypernet-based modulation?
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None
        if hyper_mod:
            assert c_dim > 0
            self.affine = FullyConnectedLayer(c_dim, in_channels, bias_init=0)
        else:
            self.affine = None

    def forward(self, x, c: torch.Tensor=None, gain=1):
        w = self.weight * self.weight_gain # [c_out, c_in, k, k]
        flip_weight = (self.up == 1) # slightly faster
        if not self.affine is None:
            weights = 1.0 + self.affine(c).tanh().unsqueeze(2).unsqueeze(3) # [batch_size, c_in, 1, 1]
            x = (x * weights).to(x.dtype) # [batch_size, c_out, h, w]
        x = conv2d_resample.conv2d_resample(
            x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up,
            down=self.down, padding=self.padding, flip_weight=flip_weight)
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, activation={self.activation:s},',
            f'up={self.up}, down={self.down}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class ScalarEncoder1d(nn.Module):
    """
    1-dimensional Fourier Features encoder (i.e. encodes raw scalars)
    Assumes that scalars are in [0, 1]
    """
    def __init__(self, coord_dim: int, x_multiplier: float, const_emb_dim: int, use_raw: bool=False, **fourier_enc_kwargs):
        super().__init__()
        self.coord_dim = coord_dim
        self.const_emb_dim = const_emb_dim
        self.x_multiplier = x_multiplier
        self.use_raw = use_raw

        if self.const_emb_dim > 0 and self.x_multiplier > 0:
            self.const_embed = nn.Embedding(int(np.ceil(x_multiplier)) + 1, self.const_emb_dim)
        else:
            self.const_embed = None

        if self.x_multiplier > 0:
            self.fourier_encoder = FourierEncoder1d(coord_dim, max_x_value=x_multiplier, **fourier_enc_kwargs)
            self.fourier_dim = self.fourier_encoder.get_dim()
        else:
            self.fourier_encoder = None
            self.fourier_dim = 0

        self.raw_dim = 1 if self.use_raw else 0

    def get_dim(self) -> int:
        return self.coord_dim * (self.const_emb_dim + self.fourier_dim + self.raw_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assumes that x is in [0, 1] range
        """
        misc.assert_shape(x, [None, self.coord_dim])
        batch_size, coord_dim = x.shape
        out = torch.empty(batch_size, self.coord_dim, 0, device=x.device, dtype=x.dtype) # [batch_size, coord_dim, 0]
        if self.use_raw:
            out = torch.cat([out, x.unsqueeze(2)], dim=2) # [batch_size, coord_dim, 1]
        if not self.fourier_encoder is None or not self.const_embed is None:
            # Convert from [0, 1] to the [0, `x_multiplier`] range
            x = x.float() * self.x_multiplier # [batch_size, coord_dim]
        if not self.fourier_encoder is None:
            fourier_embs = self.fourier_encoder(x) # [batch_size, coord_dim, fourier_dim]
            out = torch.cat([out, fourier_embs], dim=2) # [batch_size, coord_dim, raw_dim + fourier_dim]
        if not self.const_embed is None:
            const_embs = self.const_embed(x.round().long()) # [batch_size, coord_dim, const_emb_dim]
            out = torch.cat([out, const_embs], dim=2) # [batch_size, coord_dim, raw_dim + fourier_dim + const_emb_dim]
        out = out.view(batch_size, coord_dim * (self.raw_dim + self.const_emb_dim + self.fourier_dim)) # [batch_size, coord_dim * (raw_dim + const_emb_dim + fourier_dim)]
        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class FourierEncoder1d(nn.Module):
    def __init__(self,
            coord_dim: int,                 # Number of scalars to encode for each sample
            max_x_value: float=100.0,       # Maximum scalar value (influences the amount of fourier features we use). Can be also seen as a grid resolution
            transformer_pe: bool=False,     # Whether we should use positional embeddings from Transformer
            use_cos: bool=True,
            **construct_freqs_kwargs,
        ):
        super().__init__()
        assert coord_dim >= 1, f"Wrong coord_dim: {coord_dim}"
        self.coord_dim = coord_dim
        self.use_cos = use_cos
        if transformer_pe:
            d_model = 512
            fourier_coefs = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)) # [d_model]
        else:
            fourier_coefs = construct_log_spaced_freqs(max_x_value, **construct_freqs_kwargs)
        self.register_buffer('fourier_coefs', fourier_coefs) # [num_fourier_feats]
        self.fourier_dim = self.fourier_coefs.shape[0]

    def get_dim(self) -> int:
        return self.fourier_dim * (2 if self.use_cos else 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, f"Wrong shape: {x.shape}"
        assert x.shape[1] == self.coord_dim
        fourier_raw_embs = self.fourier_coefs.view(1, 1, self.fourier_dim) * x.float().unsqueeze(2) # [batch_size, coord_dim, fourier_dim]
        if self.use_cos:
            fourier_embs = torch.cat([fourier_raw_embs.sin(), fourier_raw_embs.cos()], dim=2) # [batch_size, coord_dim, 2 * fourier_dim]
        else:
            fourier_embs = fourier_raw_embs.sin() # [batch_size, coord_dim, fourier_dim]
        return fourier_embs

#----------------------------------------------------------------------------

def construct_log_spaced_freqs(grid_res: int, skip_n_high_freqs: int=0, skip_n_low_freqs: int=0) -> Tuple[int, torch.Tensor]:
    """
    We construct the frequency coefficients in such a way,
    that the lowest frequency has the period of the provided grid resolution
    """
    num_freqs = np.ceil(np.log2(grid_res)).astype(int)
    grid_res = 2 ** num_freqs
    fourier_coefs = torch.tensor([2.0]).repeat(num_freqs) ** torch.arange(num_freqs) / grid_res # [num_fourier_feats]
    fourier_coefs = fourier_coefs.float() * np.pi # [num_fourier_feats]
    fourier_coefs = fourier_coefs[skip_n_low_freqs:len(fourier_coefs) - skip_n_high_freqs] # [num_fourier_feats]

    return fourier_coefs

#----------------------------------------------------------------------------
