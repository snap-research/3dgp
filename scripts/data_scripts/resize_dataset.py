import sys; sys.path.extend(['.'])
import os
import re
import shutil
import argparse

from tqdm import tqdm
from PIL import Image
from joblib import Parallel, delayed

from scripts.utils import tqdm_joblib, resize_and_save_image, file_ext

#----------------------------------------------------------------------------

def resize_dataset(
    source_dir, target_dir: str=None, size: int=None, format: str=None, num_jobs: int=8, ignore_regex: str=None,
    ignore_ext: str=None, images_only: bool=False, fname_prefix: str='', rename_enum: bool=False, **resizing_kwargs):

    assert not size is None

    Image.init() # to load the extensions
    target_dir = f'{source_dir}_{size}' if target_dir is None else target_dir
    file_names = {os.path.relpath(os.path.join(root, fname), start=source_dir) for root, _dirs, files in os.walk(source_dir) for fname in files}

    if not ignore_ext is None:
        file_names = {f for f in file_names if not f.endswith(ignore_ext)}

    if not ignore_regex is None:
        file_names = {f for f in file_names if not re.fullmatch(ignore_regex, f)}

    jobs = []
    dirs_to_create = set()

    for i, file_name in tqdm(enumerate(file_names), desc=f'Collecting jobs'):
        src_path = os.path.join(source_dir, file_name)
        src_ext = file_ext(src_path)

        if src_ext in Image.EXTENSION:
            trg_file_basename = f'{i:08d}' if rename_enum else (fname_prefix + file_name[:file_name.rfind('.')])
            trg_path = os.path.join(target_dir, trg_file_basename + (src_ext if format is None else format))
            jobs.append(delayed(resize_and_save_image)(
                src_path=src_path,
                trg_path=trg_path,
                size=size,
                **resizing_kwargs,
            ))
        elif not images_only:
            assert not os.path.islink(src_path)
            trg_path = os.path.join(target_dir, file_name)
            print(f'Copying {src_path} => {trg_path} since it is not an image')
            jobs.append(delayed(shutil.copyfile)(src=src_path, dst=trg_path))
        else:
            trg_path = None

        if not trg_path is None:
            dirs_to_create.add(os.path.dirname(trg_path))

    for d in tqdm(dirs_to_create, desc='Creating necessary directories'):
        if d != '':
            os.makedirs(d, exist_ok=True)

    with tqdm_joblib(tqdm(desc="Executing jobs", total=len(jobs))) as progress_bar:
        Parallel(n_jobs=num_jobs)(jobs)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--source_dir', required=True, type=str, help='Source directory')
    parser.add_argument('-t', '--target_dir', required=False, type=str, default=None, help='Target directory (default: `{source_dir}_{size}`)')
    parser.add_argument('-s', '--size', required=True, type=int, help='Target size.')
    parser.add_argument('-f', '--format', type=str, default=None, help='In which format should we save? If none, then use the source file format.')
    parser.add_argument('-j', '--num_jobs', type=int, default=8, help='Number of jobs for parallel execution')
    parser.add_argument('--ignore_ext', type=str, default='.DS_Store', help='File extension to ignore.')
    parser.add_argument('--fname_prefix', type=str, default='', help='Add this prefix to each file name.')
    parser.add_argument('--images_only', action='store_true', help='Process images only?')
    parser.add_argument('--rename_enum', action='store_true', help='Should we rename each file name with a numeric id?')
    parser.add_argument('--ignore_grayscale', action='store_true', help='Should we ignore grayscale images?')
    parser.add_argument('--ignore_broken', action='store_true', help='Should we ignore images which we failed to process?')
    parser.add_argument('--ignore_existing', action='store_true', help='Should we ignore images which have been already saved?')
    args = parser.parse_args()

    resize_dataset(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        size=args.size,
        format=args.format,
        num_jobs=args.num_jobs,
        ignore_ext=args.ignore_ext,
        fname_prefix=args.fname_prefix,
        images_only=args.images_only,
        rename_enum=args.rename_enum,
        ignore_grayscale=args.ignore_grayscale,
        ignore_broken=args.ignore_broken,
        ignore_existing=args.ignore_existing,
    )

#----------------------------------------------------------------------------
