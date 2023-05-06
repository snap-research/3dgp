"""
Sometimes we launch "test" experiments just for a couple of epochs.
They occupy a lot of space and pollute the tensorboard.
This script helps to clean them out.
"""

import os
import time
import shutil
import argparse
from typing import List, Union
from tqdm import tqdm

#----------------------------------------------------------------------------

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

#----------------------------------------------------------------------------

def clean_dummy_exps(directory: os.PathLike, more_than_n_days_old: float=1, verbose: bool=False, min_iter: int=None, follow_output_link: bool=False, print_only: bool=False, delete_dirs_without_output: bool=False, **kwargs):
    num_dirs_removed = 0
    num_permission_errors = 0
    amount_of_space_freed = 0
    exp_dir_paths = find_exp_directories(directory, verbose=verbose, **kwargs)
    print(f'Found {len(exp_dir_paths)} experiments in total')
    exp_dir_paths = [p for p in exp_dir_paths if should_delete_exp(p, min_iter, delete_dirs_without_output)]
    print(f'Going to delete {len(exp_dir_paths)} experiments.')
    dirs_to_delete = exp_dir_paths
    if follow_output_link:
        output_dirs = [os.path.join(d, 'output') for d in dirs_to_delete]
        dirs_to_delete.extend([os.readlink(d) for d in output_dirs if os.path.islink(d) and os.path.exists(os.readlink(d))])
    now = time.time()
    dirs_to_delete = [d for d in dirs_to_delete if (now - os.stat(d).st_mtime) > 60 * 60 * 24 * more_than_n_days_old]

    for dir_path in tqdm(dirs_to_delete):
        assert os.path.isdir(dir_path), f"Not a directory: {dir_path}"

        dir_size = get_dir_size(dir_path)

        try:
            if print_only:
                print(f'Pseudo-removing {dir_path}')
            else:
                shutil.rmtree(dir_path)
        except PermissionError:
            num_permission_errors += 1
            continue

        num_dirs_removed += 1
        amount_of_space_freed += dir_size

    print(f'Deleted {num_dirs_removed} experiments. Couldnt remove {num_permission_errors} experiments due to PermissionError.')
    print(f'Freed up {sizeof_fmt(amount_of_space_freed)} space.')

#----------------------------------------------------------------------------

def find_exp_directories(root_dir: os.PathLike, recursive: bool=False, verbose: bool=False, ignore_dir_names: List[str]=[]) -> List[os.PathLike]:
    assert os.path.isdir(root_dir), f"Not a directory: {root_dir}"
    dirs_list = []
    print(f'Collecting directories to remove inside {root_dir}...')

    for dir_name in os.listdir(root_dir):
        if dir_name in ignore_dir_names:
            continue

        dir_path = os.path.join(root_dir, dir_name)

        if is_experiment_dir(dir_path) or is_experiment_proxy_dir(dir_path):
            dirs_list.append(dir_path)
            continue

        # If not the experiment, then recursively check the contents
        if os.path.isdir(dir_path) and recursive:
            dirs_list.extend(find_exp_directories(dir_path, recursive, verbose, ignore_dir_names))
            continue

        if os.path.islink(dir_path) and recursive:
            real_dir_path = os.readlink(dir_path)
            dirs_list.extend(find_exp_directories(real_dir_path, recursive, verbose, ignore_dir_names))
            continue

    return dirs_list

#----------------------------------------------------------------------------

def is_experiment_dir(dir_path: os.PathLike) -> bool:
    REQUIRED_CONTENTS = {
        'src': 'dir',
        'experiment_config.yaml': 'file',
        # 'output': ['link', 'dir'],
        'configs': 'dir',
        'training_cmd.sh': 'file',
        'data': 'link',
    }
    return all([has_object(dir_path, o, t) for o, t in REQUIRED_CONTENTS.items()])

#----------------------------------------------------------------------------

def is_experiment_proxy_dir(dir_path: os.PathLike) -> bool:
    return os.listdir(dir_path) == ['output']

#----------------------------------------------------------------------------

def has_object(dir_path: os.PathLike, object_name: str, object_type: Union[str, List[str]]) -> bool:
    """
    Checks whether the directory contains the object with name `object_name` and type `object_type`
    """
    object_path = os.path.join(dir_path, object_name)
    if os.path.islink(object_path):
        # print('is link?', object_path, object_type, 'link' == object_type or 'link' in object_type)
        return 'link' == object_type or 'link' in object_type
    if os.path.isdir(object_path):
        # print('is dir?', object_path, object_type, 'dir' == object_type or 'dir' in object_type)
        return 'dir' == object_type or 'dir' in object_type
    if os.path.isfile(object_path):
        # print('is file?', object_path, object_type, 'file' == object_type or 'file' in object_type)
        return 'file' == object_type or 'file' in object_type
    return False

#----------------------------------------------------------------------------

def should_delete_exp(exp_dir_path: os.PathLike, min_iter: int, delete_dirs_without_output: bool=False) -> bool:
    assert is_experiment_dir(exp_dir_path) or is_experiment_proxy_dir(exp_dir_path)
    output_path = os.path.join(exp_dir_path, 'output')
    if (not has_object(exp_dir_path, 'output', ['link', 'dir'])) or (os.path.islink(output_path) and not os.path.exists(output_path)):
        return delete_dirs_without_output
    checkpoints_dir = os.path.join(exp_dir_path, 'output')
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith('network-snapshot') and f.endswith('.pkl')]

    if len(checkpoints) == 0:
        last_iter = -1
    else:
        last_ckpt = sorted(checkpoints)[-1]
        last_iter = int(last_ckpt[last_ckpt.rfind('-') + 1:-len('.pkl')])

    return last_iter < min_iter

#----------------------------------------------------------------------------

def get_dir_size(start_path: os.PathLike):
    """
    Copy-pasted from https://stackoverflow.com/a/1392549/2685677
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--min_iter', type=int, required=True, help='Do not delete experiments which progressed for more than `min_iter` iterations.')
    parser.add_argument('--more_than_n_days_old', type=float, default=1, help='Do not delete experiments which are newer than `more_than_n_days_old`.')
    parser.add_argument('--verbose', action='store_true', help='Should we print intermediate progress information?')
    parser.add_argument('--recursive', action='store_true', help='Should we walk over the directories recursively?')
    parser.add_argument('--ignore_dir_names', type=str, help='Ignore directories with one of these names. This speeds up the search.')
    parser.add_argument('--print_only', action='store_true', help='Print only and exit or really remove?')
    parser.add_argument('--follow_output_link', action='store_true', help='Should we follow the `output` symlink?')
    parser.add_argument('--delete_dirs_without_output', action='store_true', help='Delete directories which do not have any output?')
    args = parser.parse_args()

    ignore_dir_names = [] if args.ignore_dir_names is None else args.ignore_dir_names.split(',')

    clean_dummy_exps(
        directory=args.directory,
        min_iter=args.min_iter,
        more_than_n_days_old=args.more_than_n_days_old,
        verbose=args.verbose,
        recursive=args.recursive,
        ignore_dir_names=ignore_dir_names,
        print_only=args.print_only,
        delete_dirs_without_output=args.delete_dirs_without_output,
    )

#----------------------------------------------------------------------------
