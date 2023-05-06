import sys; sys.path.extend(['.', 'src'])
import os
import json
import shutil
import pickle
from typing import Callable, List, Dict

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from src.infra.utils import recursive_instantiate
from scripts.utils import find_images_in_dir

#----------------------------------------------------------------------------

class ImagePathsDataset(VisionDataset):
    def __init__(self, img_paths: List[os.PathLike], transform: Callable):
        self.transform = transform
        self.imgs_paths = img_paths

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx: int):
        img = pil_loader(self.imgs_paths[idx]).convert('RGB')
        img = self.transform(img)

        return img

#----------------------------------------------------------------------------

class TimmModel(torch.nn.Module):
    def __init__(self, model_name: str, return_logits: bool=False):
        super().__init__()

        num_classes_kwargs = dict() if return_logits else dict(num_classes=0)
        self.model = timm.create_model(model_name, pretrained=True, **num_classes_kwargs)
        self.model.eval()
        self.transform = create_transform(**resolve_data_config({}, model=self.model))

    def get_transform(self) -> Callable:
        return transforms.Compose([lambda x: x.convert('RGB'), self.transform])

    def forward(self, x):
        return self.model(x)

#----------------------------------------------------------------------------

def extract_features(extractor: Callable, dataloader: DataLoader, device: str) -> List[np.ndarray]:
    all_feats = []

    for imgs in tqdm(dataloader, desc='Extracting features'):
        imgs = torch.stack(imgs).to(device)
        feats = extractor(imgs).cpu().numpy()
        all_feats.extend(feats)

    return np.stack(all_feats)

#----------------------------------------------------------------------------

def is_image_file(p: os.PathLike) -> bool:
    return os.path.splitext(p)[1].lower() in Image.EXTENSION

#----------------------------------------------------------------------------

def compute_covariance(feats: torch.Tensor, skip_size_check=False) -> torch.Tensor:
    """
    Computes empirical covariance matrix for a batch of feature vectors
    feats: [n, d]
    """
    assert feats.ndim == 2
    if not skip_size_check:
        assert feats.shape[1] < 10000, "The feature dimension is too big. Likely a mistake."

    feats = feats - feats.mean(dim=0, keepdim=True) # [n, d]
    cov_unscaled = feats.t() @ feats # [d, d]
    cov = cov_unscaled / (feats.shape[0] - 1) # [d, d]

    return cov

#----------------------------------------------------------------------------

def save_feats_memmap(feats: Dict, output_path: os.PathLike):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_items = list(feats.items())
    filepath_to_idx = {k: i for i, (k, _) in enumerate(data_items)}
    values = np.array([v for _, v in data_items])

    with open(output_path.replace('.memmap', '_desc.json'), 'w') as f:
        json.dump({'shape': values.shape, 'filepath_to_idx': filepath_to_idx}, f)

    f = np.memmap(output_path, dtype='float32', mode='w+', shape=values.shape)
    f[:] = values[:]
    f.flush()

#----------------------------------------------------------------------------

@hydra.main(config_name="../../configs/scripts/extract_features.yaml")
def launch_feat_extraction(cfg: DictConfig):
    torch.set_grad_enabled(False)

    Image.init()
    recursive_instantiate(cfg)
    assert cfg.output_path.endswith('.memmap'), f"Invalid target path: {cfg.output_path}"

    print('Loading a feature extractor...')
    embedder = TimmModel(cfg.embedder)
    embedder.to(cfg.device).eval().requires_grad_(False)

    print('Initializing the dataloader')
    img_paths = find_images_in_dir(cfg.dataset_path, ignore_regex='.*_depth.png')
    assert len(img_paths) > 0, f"Empty dataset: {cfg.dataset_path}"
    dataset = ImagePathsDataset(img_paths, embedder.get_transform())
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=5, collate_fn=lambda x:x)

    print('Computing features...')
    feats = extract_features(embedder, dataloader, cfg.device)

    assert len(feats) == len(img_paths), f"Number of features ({len(feats)}) does not match the number of images ({len(img_paths)})."

    print('Saving features...')
    img_names = [os.path.relpath(p, start=cfg.dataset_path) for p in img_paths]
    feats_data = {n: f for n, f in zip(img_names, feats)}
    save_feats_memmap(feats_data, cfg.output_path)

#----------------------------------------------------------------------------

if __name__ == '__main__':
    launch_feat_extraction()

#----------------------------------------------------------------------------
