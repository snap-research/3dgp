import sys; sys.path.extend(['.'])
from omegaconf import DictConfig
import torch
import numpy as np
import os
import hydra
from tqdm import tqdm

from scripts.utils import load_generator, set_seed
from scripts.inference import sample_ws_from_seeds, parse_range

#----------------------------------------------------------------------------

@hydra.main(config_path="../configs/scripts", config_name="extract_geometry.yaml")
def extract_geometry(cfg: DictConfig):
    device = torch.device('cuda')
    set_seed(42) # To fix non-z randomization
    assert not (cfg.seeds is None and cfg.num_seeds is None), "You must specify either `num_seeds` or `seeds`"
    assert cfg.seeds is None or cfg.num_seeds is None, "You cannot specify both `num_seeds` and `seeds`"
    seeds = cfg.seeds if cfg.num_seeds is None else np.arange(cfg.num_seeds)
    classes = None if cfg.classes is None else parse_range(cfg.classes)
    G = load_generator(cfg.ckpt, verbose=cfg.verbose)[0].to(device).eval()
    ws, _z, _c = sample_ws_from_seeds(G, seeds, cfg, device, classes=classes) # [num_grids, num_ws, w_dim]
    sample_names = [f'{s:04d}' for s in seeds] if classes is None else [f'c{c:04d}-s{s:04d}' for c in classes for s in seeds]

    for name, ws in tqdm(zip(sample_names, ws.split(1, dim=0)), desc='Extracting geometry...', total=len(ws)):
        batch_size = 1
        coords = create_voxel_coords(cfg.volume_res, cfg.voxel_origin, cfg.cube_size, batch_size) # [batch_size, volume_res ** 3, 3]
        coords = coords.to(ws.device) # [batch_size, volume_res ** 3, 3]
        sigma = G.synthesis.compute_densities(ws, coords, noise_mode='const') # [batch_size, volume_res ** 3, 1]
        assert batch_size == 1
        sigma = sigma.reshape(cfg.volume_res, cfg.volume_res, cfg.volume_res).cpu().numpy() # [volume_res ** 3]
        sigma = sigma[cfg.volume_res//8:-cfg.volume_res//8:, cfg.volume_res//2:, :-cfg.volume_res//3]

        os.makedirs(cfg.output_dir, exist_ok=True)

        if cfg.save_obj or cfg.save_ply:
            import mcubes
            import trimesh
            vertices, triangles = mcubes.marching_cubes(sigma, cfg.thresh_value)
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.apply_scale(1.0 / mesh.scale)
            if cfg.save_obj:
                mesh.export(os.path.join(cfg.output_dir, f'{name}.obj'))
            if cfg.save_ply:
                mesh.export(os.path.join(cfg.output_dir, f'{name}.ply'))

        if cfg.save_mrc:
            import mrcfile
            with mrcfile.new_mmap(os.path.join(cfg.output_dir, f'{name}.mrc'), overwrite=True, shape=sigma.shape, mrc_mode=2) as mrc:
                mrc.data[:] = sigma

#----------------------------------------------------------------------------

def create_voxel_coords(resolution=256, voxel_origin=[0.0, 0.0, 0.0], cube_size=2.0, batch_size: int=1):
    """Adapted from pi-GAN: https://github.com/marcoamonteiro/pi-GAN"""
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_size / 2.0
    voxel_size = cube_size / (resolution - 1)

    overall_index = torch.arange(0, resolution ** 3, 1, out=torch.LongTensor())
    coords = torch.zeros(resolution ** 3, 3) # [h * w * d, 3]

    # transform first 3 columns
    # to be the x, y, z index
    coords[:, 2] = overall_index % resolution
    coords[:, 1] = (overall_index.float() / resolution) % resolution
    coords[:, 0] = ((overall_index.float() / resolution) / resolution) % resolution

    # transform first 3 columns
    # to be the x, y, z coordinate
    coords[:, 0] = (coords[:, 0] * voxel_size) + voxel_origin[2] # [voxel_res ** 3]
    coords[:, 1] = (coords[:, 1] * voxel_size) + voxel_origin[1] # [voxel_res ** 3]
    coords[:, 2] = (coords[:, 2] * voxel_size) + voxel_origin[0] # [voxel_res ** 3]

    return coords.repeat(batch_size, 1, 1) # [batch_size, voxel_res ** 3, 3]

#----------------------------------------------------------------------------

if __name__ == "__main__":
    extract_geometry() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------