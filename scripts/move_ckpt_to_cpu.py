import sys; sys.path.extend(['.', 'src'])
import os
import torch
import pickle
import dill
import argparse

from src import dnnlib

#----------------------------------------------------------------------------

def optimizer_to(optim, device):
    """
    Copy-pasted from https://github.com/pytorch/pytorch/issues/8741#issuecomment-402129385
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return optim

#----------------------------------------------------------------------------

def move_ckpt_to_cpu(ckpt_path: os.PathLike, save_path: str=None):
    if save_path is None:
        save_path = ckpt_path.replace('.pkl', '-cpu.pkl')
    with dnnlib.util.open_url(ckpt_path) as f:
        snapshot_data = pickle.Unpickler(f).load()
    snapshot_data = {k: (v.cpu() if hasattr(v, 'cpu') else v) for k, v in snapshot_data.items()}
    snapshot_data = {k: (optimizer_to(v, 'cpu') if isinstance(v, torch.optim.Optimizer) else v) for k, v in snapshot_data.items()}
    with open(save_path, 'wb') as f:
        dill.dump(snapshot_data, f)
    print(f'Saved into {save_path}')

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_path', type=str, help='Path to a checkpoint')
    parser.add_argument('-s', '--save_path', type=str, help='Where to save the images?')
    args = parser.parse_args()
    move_ckpt_to_cpu(args.ckpt_path, args.save_path)

#----------------------------------------------------------------------------