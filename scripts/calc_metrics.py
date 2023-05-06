# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Calculate quality metrics for previous training run or pretrained network pickle."""

# Add `src` to sys.path. Otherwise, we get ModuleNotFound for torch_utils :(
# (when loading the inception model). TODO: wtf?
import sys; sys.path.extend(['.', 'src'])

import os
import copy
import tempfile

import torch
import hydra
from omegaconf import DictConfig

from src import dnnlib
from src.metrics import metric_main
from src.metrics import metric_utils
from src.torch_utils import training_stats
from src.torch_utils import custom_ops
from src.torch_utils import misc
from src.torch_utils.ops import conv2d_gradfix
from scripts.utils import load_generator
from src.training.rendering_utils import sample_camera_params

#----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = 'none'

    # Configure torch
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    # Print network summary.
    G = copy.deepcopy(args.G).eval().requires_grad_(False).to(device)
    if rank == 0 and args.verbose:
        z = torch.empty([2, G.z_dim], device=device)
        c = torch.empty([2, G.c_dim], device=device)
        camera_params = sample_camera_params(G.cfg.camera, 1, device)
        misc.print_module_summary(G, [z[[0]], c[[0]]], module_kwargs={'camera_params': camera_params[[0]]}) # [1, c, h, w]

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f'Calculating {metric}...')
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(
            metric=metric,
            G=G,
            dataset_kwargs=args.dataset_kwargs,
            num_gpus=args.num_gpus,
            rank=rank,
            device=device,
            progress=progress,
        )
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir, snapshot_pkl=args.network_pkl)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print('Exiting...')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

@hydra.main(config_path="../configs/scripts", config_name="calc_metrics.yaml")
def calc_metrics(cfg: DictConfig):
    dnnlib.util.Logger(should_flush=True)

    device = torch.device('cuda')
    G, snapshot, network_pkl = load_generator(cfg.ckpt, verbose=cfg.verbose)
    G = G.to(device).eval()
    G.cfg.camera = G.cfg.dataset.camera

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=cfg.metrics.split(','), num_gpus=cfg.gpus, network_pkl=network_pkl, verbose=cfg.verbose, G=G)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        raise ValueError('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    if not args.num_gpus >= 1:
        raise ValueError('--gpus must be at least 1')

    # Initialize dataset options.
    if cfg.data is not None:
        args.dataset_kwargs = dnnlib.EasyDict(class_name='src.training.dataset.ImageFolderDataset', path=cfg.data)
    elif snapshot['training_set_kwargs'] is not None:
        args.dataset_kwargs = dnnlib.EasyDict(snapshot['training_set_kwargs'])
    else:
        print('Not using the dataset. Assume that the metric is computed without it.')
        args.dataset_kwargs = dnnlib.EasyDict()

    # Finalize dataset options.
    args.G.synthesis.img_resolution = args.G.synthesis.test_resolution = args.G.img_resolution = (args.G.img_resolution if cfg.img_resolution is None else cfg.img_resolution)
    args.dataset_kwargs.resolution = args.G.img_resolution
    args.dataset_kwargs.use_depth = False
    args.dataset_kwargs.cfg = dnnlib.EasyDict.init_recursively(args.G.cfg.dataset)

    # Locate run dir.
    args.run_dir = None
    if os.path.isfile(network_pkl):
        pkl_dir = os.path.dirname(network_pkl)
        if os.path.isfile(os.path.join(pkl_dir, 'training_options.json')):
            args.run_dir = pkl_dir

    # Launch processes.
    if args.verbose:
        print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    calc_metrics() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
