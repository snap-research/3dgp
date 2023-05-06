# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks"."""

import os
import re
import shutil
import tempfile
import warnings

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from src import dnnlib
from src.training import training_loop
from src.metrics import metric_main
from src.torch_utils import training_stats
from src.torch_utils import custom_ops
from src.training.rendering_utils import get_max_sampling_value, validate_frustum, get_mean_sampling_value, compute_viewing_frustum_sizes
from src.training.tri_plane_renderer import validate_image_plane

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)

    c.run_dir = os.path.join(outdir, 'output')

    # Print options.
    print()
    if c.cfg.env.get('symlink_output', None):
        print(f'Output directory:    {c.run_dir} (symlinked to {c.cfg.env.symlink_output})')
    else:
        print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:          {c.num_gpus}')
    print(f'Batch size:              {c.batch_size} images')
    print(f'Training duration:       {c.total_kimg} kimg')
    print(f'Dataset path:            {c.training_set_kwargs.path}')
    print(f'Dataset size:            {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:      {c.training_set_kwargs.resolution}')
    print(f'Dataset cfg:             {c.training_set_kwargs.cfg}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    if c.cfg.env.get('symlink_output', None):
        if os.path.exists(c.cfg.env.symlink_output) and not c.resume_whole_state:
            print(f'Deleting old output dir: {c.cfg.env.symlink_output} ...')
            shutil.rmtree(c.cfg.env.symlink_output)

        if not c.resume_whole_state:
            os.makedirs(c.cfg.env.symlink_output, exist_ok=False)
            os.symlink(c.cfg.env.symlink_output, c.run_dir)
            print(f'Symlinked `output` into `{c.cfg.env.symlink_output}`')
        else:
            print(f'Did not symlink `{c.cfg.env.symlink_output}` since resuming training.')
    else:
        os.makedirs(c.run_dir, exist_ok=True)

    # Launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset(cfg: DictConfig):
    try:
        dataset_kwargs = dnnlib.EasyDict.init_recursively(dict(
            class_name='src.training.dataset.ImageFolderDataset', path=cfg.dataset.path,
            max_size=None, use_depth=cfg.training.use_depth, cfg=cfg.dataset))
        dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset.resolution # Be explicit about resolution.
        dataset_kwargs.max_size = len(dataset) # Be explicit about dataset size.
        mean_camera_params = torch.from_numpy(dataset.mean_camera_params) # [5]

        if cfg.dataset.c_dim > 0:
            assert dataset.has_labels
        if cfg.training.use_depth:
            assert dataset.has_depth
        if cfg.camera.origin.angles.dist == 'custom':
            print('Validating camera params in the dataset...', end='')
            camera_angles = torch.from_numpy(np.array([dataset.get_camera_angles(i) for i in range(len(dataset))]))
            assert camera_angles[:, [0]].pow(2).sum().sqrt() > 0.1, "Broken yaw angles (all zeros)."
            assert camera_angles[:, [1]].pow(2).sum().sqrt() > 0.1, "Broken pitch angles (all zeros)."
            assert camera_angles[:, [0]].min() >= -np.pi, f"Broken yaw angles (too small): {camera_angles[:, [0]].min()}"
            assert not torch.any(camera_angles[:, [0]] > np.pi), f"Number of broken yaw angles (too large): {torch.sum(camera_angles[:, [0]] > np.pi)}"
            assert camera_angles[:, [1]].min() >= 0.0, f"Broken pitch angles (too small): {camera_angles[:, [1]].min()}"
            assert not torch.any(camera_angles[:, [1]] > np.pi), f"Number of broken pitch angles (too large): {torch.sum(camera_angles[:, [1]] > np.pi)}"
            print('done!')

        return dataset_kwargs, mean_camera_params
    except IOError as err:
        raise ValueError(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@hydra.main(config_path="..", config_name="experiment_config.yaml")
def main(cfg: DictConfig):
    # Initialize config.
    OmegaConf.set_struct(cfg, True)

    opts = cfg.training # Training arguments.
    c = dnnlib.EasyDict(cfg=cfg) # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, cfg=cfg.model.generator, mapping_kwargs=dnnlib.EasyDict())
    c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_discriminator.Discriminator', cfg=cfg.model.discriminator, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', **cfg.model.generator.optim.kwargs)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', **cfg.model.discriminator.optim.kwargs)
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss', cfg=cfg)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, mean_camera_params = init_dataset(cfg)

    # Hyperparameters & settings.
    c.num_gpus = cfg.num_gpus
    c.batch_size = opts.batch_size
    c.batch_gpu = opts.batch_gpu or opts.batch_size // cfg.num_gpus
    c.G_kwargs.mapping_kwargs.camera_cond = cfg.model.generator.camera_cond
    c.G_kwargs.mapping_kwargs.camera_cond_drop_p = cfg.model.generator.camera_cond_drop_p
    c.G_kwargs.mapping_kwargs.mean_camera_params = mean_camera_params
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = cfg.model.discriminator.mbstd_group_size
    c.D_kwargs.epilogue_kwargs.feat_predict_dim = cfg.dataset.embedding_dim if cfg.model.loss_kwargs.kd.discr.weight > 0 else 0
    c.loss_kwargs.r1_gamma = (0.0002 * (cfg.dataset.resolution ** 2) / opts.batch_size) if cfg.model.loss_kwargs.gamma == 'auto' else cfg.model.loss_kwargs.gamma
    c.metrics = [] if opts.metrics is None else opts.metrics.split(',')
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers
    c.G_reg_interval = 4 if cfg.model.loss_kwargs.pl_weight > 0 else 0 # Enable lazy regularization for G.
    c.D_reg_interval = 16 if c.loss_kwargs.r1_gamma > 0 else None

    assert cfg.model.loss_kwargs.blur_fade_kimg <= 1000, f"`blur_fade_kimg` is too large: {cfg.model.loss_kwargs.blur_fade_kimg}"

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise ValueError('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise ValueError('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < cfg.model.discriminator.mbstd_group_size:
        raise ValueError('--batch-gpu cannot be smaller than --mbstd')
    if any(not metric_main.is_valid_metric(metric) for metric in c.metrics):
        raise ValueError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))

    # Base configuration.
    if cfg.model.name == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9 # Enable style mixing regularization.
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
    elif cfg.model.name in ('epigraf', '3dgp'):
        c.G_kwargs.class_name = 'training.networks_epigraf.Generator'
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.

        if cfg.camera.validate_viewing_frustum:
            print('Validating that the viewing frustum is inside the cube...', end='')
            assert cfg.camera.fov.dist in ('uniform', 'truncnorm') or cfg.camera.fov.std == 0
            assert cfg.camera.origin.radius.dist in ('uniform', 'truncnorm') or cfg.camera.origin.radius.std == 0
            if cfg.model.generator.use_full_box:
                assert validate_image_plane(
                    fov=get_max_sampling_value(cfg.camera.fov),
                    radius=get_max_sampling_value(cfg.camera.origin.radius),
                    scale=cfg.camera.cube_scale,
                ), f"Please, increase the scale: {cfg.camera.cube_scale}"
            else:
                assert validate_frustum(
                    fov=get_max_sampling_value(cfg.camera.fov),
                    radius=get_max_sampling_value(cfg.camera.origin.radius),
                    scale=cfg.camera.cube_scale,
                    near=cfg.camera.ray.start,
                    far=cfg.camera.ray.end,
                ), f"Please, increase the scale: {cfg.camera.cube_scale}"
            print('Done!')
        if not cfg.model.generator.use_full_box:
            vf_sizes = compute_viewing_frustum_sizes(cfg.camera.ray.start, cfg.camera.ray.end, get_mean_sampling_value(cfg.camera.fov))
            print(f'Your viewing frustum has the (average when FoV is stochastic) sizes: ' \
                  f'[altitute: {vf_sizes.altitute}, bottom base: {vf_sizes.bottom_base}, top base: {vf_sizes.top_base}')
    else:
        raise NotImplementedError(f'Unknown model: {cfg.model.name}')

    # Augmentation.
    assert opts.augment.mode in ['noaug', 'ada', 'fixed'], f"Uknown augmentation mode: {opts.augment.mode}"
    if opts.augment.mode != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **opts.augment.probs)
        if opts.augment.mode == 'ada':
            c.ada_target = opts.augment.target
        if opts.augment.mode == 'fixed':
            c.augment_p = opts.augment.p

    if cfg.training.patch.enabled:
        if cfg.training.patch.distribution in ('uniform', 'discrete_uniform', 'beta'):
            assert cfg.training.patch.min_scale_trg * cfg.dataset.resolution >= cfg.training.patch.resolution, \
                f"It does not make any sense to have so small patch size of {cfg.training.patch.min_scale_trg} " \
                f"at resolution {cfg.training.patch.resolution} when the dataset resolution is just {cfg.dataset.resolution}"

    # Resume.
    c.resume_whole_state = False
    if opts.resume == 'latest':
        ckpt_regex = re.compile("^network-snapshot-\d{6}.pkl$")
        run_dir = os.path.join(cfg.experiment_dir, 'output')
        ckpts = sorted([f for f in os.listdir(run_dir) if ckpt_regex.match(f)]) if os.path.isdir(run_dir) else []

        if len(ckpts) > 0:
            c.resume_pkl = os.path.join(run_dir, ckpts[-1])
            c.resume_whole_state = True
            print(f'Will resume training from {ckpts[-1]}')
        else:
            warnings.warn("Was requested to continue training, but couldn't find any checkpoints. Please remove `training.resume=latest` argument.")
    elif opts.resume is not None:
        c.resume_pkl = opts.resume
        if opts.resume_only_G:
            c.ada_kimg = 100 # Make ADA react faster at the beginning.
            c.ema_rampup = None # Disable EMA rampup.
            c.cfg.model.loss_kwargs.blur_init_sigma = 0 # Disable blur rampup.
        else:
            print('Will load whole state from {c.resume_pkl}')
            c.resume_whole_state = True

    # Performance-related toggles.
    if cfg.model.generator.fp32_only:
        c.G_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = None
    if cfg.model.discriminator.fp32_only:
        c.D_kwargs.num_fp16_res = 0
        c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    # Launch.
    launch_training(c=c, outdir=cfg.experiment_dir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
