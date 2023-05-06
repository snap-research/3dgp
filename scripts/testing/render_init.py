"""
This script computes imgs/sec for a generator in the eval mode
for different batch sizes
"""
import os
import sys; sys.path.extend(['..', '.', 'src'])

import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from torchvision import utils
import torchvision.transforms.functional as TVF

from src import dnnlib
from src.infra.utils import recursive_instantiate
from src.training.rendering_utils import sample_camera_params
from src.torch_utils import misc

#----------------------------------------------------------------------------

def instantiate_G(cfg: DictConfig, use_grad: bool=False) -> nn.Module:
    G_kwargs = dnnlib.EasyDict(class_name=None, cfg=cfg.model.generator, mapping_kwargs=dnnlib.EasyDict())
    G_kwargs.mapping_kwargs.camera_cond = cfg.training.get('camera_cond', False)
    G_kwargs.mapping_kwargs.camera_cond_drop_p = cfg.training.get('camera_cond_drop_p', 0.0)

    if cfg.model.name == 'stylegan2':
        G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    elif cfg.model.name in ('epigraf', '3dgp'):
        G_kwargs.class_name = 'training.networks_epigraf.Generator'
        G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    else:
        raise NotImplementedError(f'Unknown generator: {cfg.model.name}')
    G_kwargs.img_resolution = cfg.get('resolution', 256)
    G_kwargs.img_channels = 3

    if cfg.model.generator.fp32_only:
        G_kwargs.num_fp16_res = 0
        G_kwargs.conv_clamp = None

    G = dnnlib.util.construct_class_by_name(**G_kwargs).eval().requires_grad_(use_grad)

    return G

#----------------------------------------------------------------------------

@hydra.main(config_path="../../configs", config_name="config.yaml")
def render_init(cfg: DictConfig):
    recursive_instantiate(cfg)
    device = 'cuda'
    batch_size = cfg.get('render_batch_size', 16)
    all_imgs = []
    save_path = os.path.join(cfg.env.project_path, 'fakes_init.png')

    for i in range(4):
        G = instantiate_G(cfg, use_grad=False).to(device)

        with torch.set_grad_enabled(mode=False):
            z = torch.randn(batch_size, G.z_dim, device=device)
            c = torch.zeros(batch_size, G.c_dim, device=device)
            camera_params = sample_camera_params(G.cfg.camera, batch_size, device)
            gen_out = G(z, c, camera_params=camera_params, render_opts=dict(ignore_bg=cfg.get('ignore_bg', False), bg_only=cfg.get('bg_only', False), concat_depth=cfg.get('concat_depth', False), return_depth=True))
            if cfg.get('print_depth_stats', False):
                misc.print_stats('depth', gen_out.depth)
            gen_out.img = gen_out.img[:, :3].clamp(-1, 1).cpu() * 0.5 + 0.5 # [b, c, h, w]
        all_imgs.append(utils.make_grid(gen_out.img, nrow=int(np.sqrt(len(gen_out.img))), padding=2))

    img = utils.make_grid(torch.stack(all_imgs), nrow=int(np.sqrt(len(all_imgs))), padding=5, pad_value=1.0)
    TVF.to_pil_image(img).save(save_path, q=95)

    print(f'Saved into {save_path}')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    render_init()

#----------------------------------------------------------------------------
