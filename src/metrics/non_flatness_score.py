"""Computes Non-Flatness Score"""

import numpy as np
import torch
from src.metrics import metric_utils

#----------------------------------------------------------------------------

def compute_flatness_score(opts, num_gen: int, min_depth: float, max_depth: float, num_bins: int=64, cut_quantile: float=0.5):
    # Step 1: extract depth maps
    depth_gen = metric_utils.compute_flattened_depth_maps(opts=opts, rel_lo=0, rel_hi=1, max_items=num_gen, capture_all=True, cut_quantile=cut_quantile) # [num_gen, h * w]
    depth_gen = depth_gen.clamp(min_depth, max_depth) # [num_gen, h * w]

    # Step 2: convert depth maps into histograms
    depth_histograms = convert_depth_maps_to_histograms(depth_gen, bins=num_bins, min=min_depth, max=max_depth) # [num_gen, num_bins]

    # Step 3: compute entropy for the histogram
    entropy = compute_histogram_entropy(depth_histograms) # [num_gen]
    flatness_score = entropy.exp().mean().item() # [1]

    return float(flatness_score)

#----------------------------------------------------------------------------

@torch.no_grad()
def convert_depth_maps_to_histograms(depth_maps: torch.Tensor, *args, **kwargs):
    """
    Unfortunately, torch cannot compute histograms batch-wise...
    """
    histograms = torch.stack([torch.histc(d, *args, **kwargs) for d in depth_maps], dim=0) # [num_depth_maps, num_bins]
    # Validating the histograms
    counts = histograms.sum(dim=1) # [num_depth_maps]
    assert counts.min() == counts.max() == depth_maps[0].numel(), f"Histograms countain OOB values: {counts.min(), counts.max(), depth_maps.shape}"

    return histograms

#----------------------------------------------------------------------------

def compute_histogram_entropy(histograms: torch.Tensor) -> torch.Tensor:
    assert histograms.ndim == 2, f"Wrong shape: {histograms.shape}"
    probs = histograms / histograms.sum(dim=1, keepdim=True) # [batch_size, num_bins]
    return -1.0 * (torch.log(probs + 1e-12) * probs).sum(dim=1)

#----------------------------------------------------------------------------
