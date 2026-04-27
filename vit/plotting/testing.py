from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def coordinate_text(crop_bbox: tuple[int, int, int, int] | None, slice_index: int) -> str:
    if crop_bbox is None:
        return f'slice={slice_index} | full slice'
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


def build_direct_segmentation_figure(
    image_crop: np.ndarray,
    reconstruction_crop: np.ndarray,
    reconstruction_coverage_mask: np.ndarray,
    prediction_map: np.ndarray,
    crop_gt: np.ndarray,
    key: str,
    slice_index: int,
    crop_bbox: tuple[int, int, int, int] | None,
) -> plt.Figure:
    seg_cmap = ListedColormap(['#111827', '#ef4444', '#10b981', '#3b82f6'])
    valid_mask = prediction_map >= 0
    display_pred = np.ma.masked_where(~valid_mask, prediction_map)
    display_gt = np.ma.masked_where(crop_gt < 0, crop_gt)
    display_reconstruction = np.ma.masked_where(~reconstruction_coverage_mask, reconstruction_crop)

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), constrained_layout=True)
    caption = coordinate_text(crop_bbox, slice_index)

    axes[0].imshow(image_crop, cmap='gray')
    axes[0].set_title(f'{key} slice {slice_index} image crop\n{caption}')
    axes[0].axis('off')

    axes[1].imshow(image_crop, cmap='gray', alpha=0.20)
    axes[1].imshow(display_reconstruction, cmap='gray')
    axes[1].set_title(f'Reconstruction\n{caption}')
    axes[1].axis('off')

    axes[2].imshow(image_crop, cmap='gray', alpha=0.35)
    axes[2].imshow(display_pred, cmap=seg_cmap, vmin=0, vmax=3, alpha=0.85)
    axes[2].set_title(f'Direct classifier segmentation\n{caption}')
    axes[2].axis('off')

    axes[3].imshow(image_crop, cmap='gray', alpha=0.35)
    axes[3].imshow(display_gt, cmap=seg_cmap, vmin=0, vmax=3, alpha=0.85)
    axes[3].set_title(f'Ground truth\n{caption}')
    axes[3].axis('off')

    return fig


def build_reconstruction_figure(
    image_crop: np.ndarray,
    reconstruction_crop: np.ndarray,
    coverage_mask: np.ndarray,
    key: str,
    slice_index: int,
    crop_bbox: tuple[int, int, int, int] | None,
    effective_local_bbox: tuple[int, int, int, int],
) -> plt.Figure:
    display_reconstruction = np.ma.masked_where(~coverage_mask, reconstruction_crop)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    caption = coordinate_text(crop_bbox, slice_index)
    covered_fraction = float(coverage_mask.mean()) if coverage_mask.size > 0 else 0.0

    axes[0].imshow(image_crop, cmap='gray')
    axes[0].set_title(f'{key} slice {slice_index} image crop\n{caption}')
    axes[0].axis('off')

    axes[1].imshow(image_crop, cmap='gray', alpha=0.20)
    axes[1].imshow(display_reconstruction, cmap='gray')
    axes[1].set_title(f'Reconstruction\ncovered={covered_fraction:.1%} | local_bbox={effective_local_bbox}')
    axes[1].axis('off')

    axes[2].imshow(coverage_mask.astype(np.float32), cmap='gray', vmin=0.0, vmax=1.0)
    axes[2].set_title(f'Reconstruction coverage\n{caption}')
    axes[2].axis('off')

    return fig


def build_confusion_matrix_figure(
    confusion_matrix: np.ndarray,
    key: str,
    slice_index: int,
    crop_bbox: tuple[int, int, int, int] | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(confusion_matrix, cmap='Blues')
    caption = coordinate_text(crop_bbox, slice_index)
    ax.set_title(f'Confusion matrix counts\n{key} slice {slice_index}\n{caption}')
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(range(confusion_matrix.shape[1]))
    ax.set_yticks(range(confusion_matrix.shape[0]))
    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            ax.text(col, row, str(int(confusion_matrix[row, col])), ha='center', va='center', color='black')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig


def build_confidence_heatmap_figure(
    confidence_matrix: np.ndarray,
    title: str,
    key: str,
    slice_index: int,
    crop_bbox: tuple[int, int, int, int] | None,
) -> plt.Figure:
    display = np.ma.masked_invalid(confidence_matrix)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(display, cmap='viridis', vmin=0.0, vmax=1.0)
    caption = coordinate_text(crop_bbox, slice_index)
    ax.set_title(f'{title}\n{key} slice {slice_index}\n{caption}')
    ax.set_xlabel('Probability assigned to class')
    ax.set_ylabel('True class')
    ax.set_xticks(range(confidence_matrix.shape[1]))
    ax.set_yticks(range(confidence_matrix.shape[0]))
    for row in range(confidence_matrix.shape[0]):
        for col in range(confidence_matrix.shape[1]):
            value = confidence_matrix[row, col]
            label = 'nan' if np.isnan(value) else f'{value:.3f}'
            ax.text(col, row, label, ha='center', va='center', color='white' if not np.isnan(value) and value > 0.5 else 'black')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig
