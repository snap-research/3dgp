# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import json
import zipfile
from typing import Tuple, Optional

import PIL.Image
import torch
import numpy as np
from src import dnnlib
from src.training.rendering_utils import get_mean_sampling_value, get_mean_angles_values

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                      # Name of the dataset.
        raw_shape,                 # Shape of the raw image data (NCHW).
        max_size       = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_depth      = False,    # Enable training with depths? False = dummy depths are returned
        random_seed    = 0,        # Random seed to use when applying max_size.
        cfg            = {},       # Main dataset config.
    ):
        self.cfg = cfg
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = self.cfg.c_dim > 0
        self._use_embeddings = cfg.use_embeddings
        self._use_depth = use_depth
        self._raw_labels = None
        self._raw_embeddings = None
        self._idx2embidx = None # In case we'll read embeddings from h5
        self._raw_camera_angles = None
        self._mean_camera_params = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if self.cfg.mirror:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                assert not self._use_labels, f"We planned to use labels, but couldn't load them from dataset.json"
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)

            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]

            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)

        return self._raw_labels

    def _get_raw_embeddings(self):
        if self._raw_embeddings is None:
            if self._use_embeddings:
                self._idx2embidx, self._raw_embeddings = self._load_embeddings_memmap()
            else:
                self._idx2embidx = np.arange(self._raw_shape[0])
                self._raw_embeddings = np.zeros([self._raw_shape[0], 0], dtype=np.float32)

        return self._raw_embeddings

    def _get_raw_camera_angles(self):
        if self._raw_camera_angles is None:
            self._raw_camera_angles = self._load_raw_camera_angles()
            if self._raw_camera_angles is None:
                self._raw_camera_angles = np.zeros([self._raw_shape[0], 3], dtype=np.float32)
            else:
                self._raw_camera_angles = self._raw_camera_angles.astype(np.float32)
            assert isinstance(self._raw_camera_angles, np.ndarray)
            assert self._raw_camera_angles.shape[0] == self._raw_shape[0]
        return self._raw_camera_angles

    def compute_num_classes(self) -> int:
        return len(np.unique(self._get_raw_labels()))

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
            # TODO: are we sure that we've closed the h5py.File?
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape, f"Wrong shape: {image.shape} vs {self.image_shape}"
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        return {
            'image': image.copy(),
            'label': self.get_label(idx),
            'camera_angles': self.get_camera_angles(idx),
            'depth': self.get_depth(idx).copy() if self._use_depth else np.array([[0]], dtype=np.int32),
            'embedding': self.get_embedding(idx),
        }

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_embedding(self, idx):
        raw_embeddings = self._get_raw_embeddings() # get or init the embeddings
        emb_idx = self._idx2embidx[self._raw_idx[idx]]
        emb = np.array(raw_embeddings[emb_idx]) # Convert memmap to ndarray
        return emb.copy()

    def get_camera_angles(self, idx) -> Tuple[float]:
        camera_angles = self._get_raw_camera_angles()[self._raw_idx[idx]].copy() # [3]
        if self._xflip[idx]:
            assert len(camera_angles) == 3, f"Wrong shape: {camera_angles.shape}"
            camera_angles[0] = -(camera_angles[0] - self.mean_camera_params[0]) + self.mean_camera_params[0] # Flipping the rotation angle
        return camera_angles

    def get_depth(self, idx: int) -> np.ndarray:
        assert self._use_depth
        depth = self._load_raw_depth(self._raw_idx[idx])
        assert isinstance(depth, np.ndarray)
        assert list(depth.shape) == [1, *self.image_shape[1:]], f"Wrong depth shape: {depth.shape}"
        assert depth.dtype == np.int32
        if self._xflip[idx]:
            assert depth.ndim == 3 # CHW
            depth = depth[:, :, ::-1] # [1, h, w]
        return depth

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self.get_label(idx)
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_depth(self) -> bool:
        # TODO: in fact, we should check that every image has its depth map, but that would be too slow...
        return self.get_depth(0).size > 1

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

    @property
    def mean_camera_params(self) -> np.ndarray:
        if self._mean_camera_params is None:
            if self.cfg.camera.origin.angles.dist == 'custom':
                camera_angles = np.array([self.get_camera_angles(i) for i in range(len(self._get_raw_camera_angles()))]) # [dataset_size, 3]
                mean_camera_angles = camera_angles.mean(axis=0) # [3]
            else:
                mean_camera_angles = get_mean_angles_values(self.cfg.camera.origin.angles) # [3]
            self._mean_camera_params = np.concatenate([mean_camera_angles, np.array([get_mean_sampling_value(self.cfg.camera.fov), get_mean_sampling_value(self.cfg.camera.origin.radius)])]) # [5]
        return self._mean_camera_params

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError(f'Path must point to a directory or zip, but got {self._path}.')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION and not fname.endswith('_depth.png'))
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_depth(self, raw_idx) -> np.ndarray:
        image_base_name = self._image_fnames[raw_idx][:-len(self._file_ext(self._image_fnames[raw_idx]))]
        depth_fname = f'{image_base_name}_depth.png'
        with self._open_file(depth_fname) as f:
            depth = pyspng.load(f.read()) # [h, w, num_depth_channels]
            # LeReS produces [h, w, 2], while ZoeDepth produces [h, w]
            assert depth.ndim in (2, 3), f'Unsupported depth ndim {depth.ndim}'
            assert depth.dtype in (np.uint8, np.uint16), f'Unsupported depth dtype {depth.dtype}'
            depth = depth[:, :, [0]] if depth.ndim > 2 else depth[:, :, np.newaxis]  # [h, w, 1]
            # For LeReS, we have 16 bits per color sample. For ZoeDepth, it is 8.
            depth = depth.astype(np.uint16) * 256 if depth.dtype == np.uint8 else depth # [h, w, 1]
            depth = depth.astype(np.int32) # [h, w, 1]
            depth = depth.transpose(2, 0, 1) # [1, h, w]
        return depth

    def _load_field(self, field_name: str):
        dataset_file = self._get_file_by_suffix('dataset.json')
        if dataset_file is None:
            return None
        with self._open_file(dataset_file) as f:
            values = json.load(f).get(field_name)
        if values is None:
            return None
        values = dict(values)
        values = [values[remove_root(fname, self._name).replace('\\', '/')] for fname in self._image_fnames]
        values = np.array(values)
        return values

    def _load_raw_labels(self):
        labels = self._load_field('labels')
        if labels is None:
            return None
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _load_raw_camera_angles(self):
        return self._load_field('camera_angles')

    def _get_file_by_suffix(self, suffix: str) -> Optional[os.PathLike]:
        files = [f for f in self._all_fnames if f.endswith(suffix)]
        if len(files) == 0:
            return None
        assert len(files) == 1, f"There can be only a single {suffix} file"
        return files[0]

    def _load_embeddings_memmap(self) -> Tuple:
        with open(self.cfg.embeddings_desc_path) as f:
            desc = json.load(f)
        embeddings = np.memmap(self.cfg.embeddings_path, dtype='float32', mode='r', shape=tuple(desc['shape']))
        idx2embidx = [desc['filepath_to_idx'][remove_root(fname, self._name).replace('\\', '/')] for fname in self._image_fnames]
        idx2embidx = np.array(idx2embidx).astype(np.int32)
        return idx2embidx, embeddings

#----------------------------------------------------------------------------

def remove_root(fname: os.PathLike, root_name: os.PathLike):
    """`root_name` should NOT start with '/'"""
    if fname == root_name or fname == ('/' + root_name):
        return ''
    elif fname.startswith(root_name + '/'):
        return fname[len(root_name) + 1:]
    elif fname.startswith('/' + root_name + '/'):
        return fname[len(root_name) + 2:]
    else:
        return fname

#----------------------------------------------------------------------------
