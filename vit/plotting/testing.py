from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def coordinate_text(crop_bbox: tuple[int, int, int, int], slice_index: int) -> str:
    y0, y1, x0, x1 = crop_bbox
    return f'slice={slice_index} | y={y0}:{y1} | x={x0}:{x1} | h={y1-y0} | w={x1-x0}'


def build_kmeans_figure(
    image_crop: np.ndarray,
    prediction_map: np.ndarray,
    crop_gt: np.ndarray,
    key: str,
    slice_index: int,
    crop_bbox: tuple[int, int, int, int],
) -> plt.Figure:
    seg_cmap = ListedColormap(['#111827', '#ef4444', '#10b981', '#3b82f6'])
    valid_mask = prediction_map >= 0
    display_pred = np.ma.masked_where(~valid_mask, prediction_map)
    display_gt = np.ma.masked_where(crop_gt < 0, crop_gt)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    caption = coordinate_text(crop_bbox, slice_index)

    axes[0].imshow(image_crop, cmap='gray')
    axes[0].set_title(f'{key} slice {slice_index} image crop\n{caption}')
    axes[0].axis('off')

    axes[1].imshow(image_crop, cmap='gray', alpha=0.35)
    axes[1].imshow(display_pred, cmap=seg_cmap, vmin=0, vmax=3, alpha=0.85)
    axes[1].set_title(f'KMeans prediction (k=4)\n{caption}')
    axes[1].axis('off')

    axes[2].imshow(image_crop, cmap='gray', alpha=0.35)
    axes[2].imshow(display_gt, cmap=seg_cmap, vmin=0, vmax=3, alpha=0.85)
    axes[2].set_title(f'Ground truth\n{caption}')
    axes[2].axis('off')

    return fig
