from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.app_config import (
    TEST_IMAGE_EXTRACTOR_DEFAULT_DATA_DIR,
    TEST_IMAGE_EXTRACTOR_DEFAULT_EXPORT_DIR,
    test_image_extractor_default_output,
)
from label_utils import remap_label_array
from plotting.common import append_timestamp

DEFAULT_KEY = 'high_c4'
DEFAULT_SLICE_INDEX = 626
DEFAULT_CROP_HEIGHT = 128
DEFAULT_CROP_WIDTH = 128
DEFAULT_SEARCH_STRIDE = 8
DEFAULT_PATCH_SIZE = 32
DEFAULT_CLASSES = (0, 1, 2, 3)
DEFAULT_EXPORT_DIR = TEST_IMAGE_EXTRACTOR_DEFAULT_EXPORT_DIR
DEFAULT_OUTPUT = test_image_extractor_default_output(DEFAULT_KEY, DEFAULT_SLICE_INDEX)


def default_crop_png_path(key: str = DEFAULT_KEY, slice_index: int = DEFAULT_SLICE_INDEX, export_dir: Path = DEFAULT_EXPORT_DIR) -> Path:
    return export_dir / f'{key}_slice{slice_index}_crop.png'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Find a fixed crop on high_c4 slice 626 that contains all four classes.'
    )
    parser.add_argument('--data-dir', type=Path, default=TEST_IMAGE_EXTRACTOR_DEFAULT_DATA_DIR)
    parser.add_argument('--key', default=DEFAULT_KEY)
    parser.add_argument('--slice-index', type=int, default=DEFAULT_SLICE_INDEX)
    parser.add_argument('--crop-height', type=int, default=DEFAULT_CROP_HEIGHT)
    parser.add_argument('--crop-width', type=int, default=DEFAULT_CROP_WIDTH)
    parser.add_argument('--search-stride', type=int, default=DEFAULT_SEARCH_STRIDE)
    parser.add_argument('--patch-size', type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Path to save the extracted crop preview PNG. Defaults inside the exports folder.')
    return parser.parse_args()


def load_image_and_label_slice(data_dir: Path, key: str, slice_index: int) -> tuple[np.ndarray, np.ndarray]:
    image_path = data_dir / key / f'{key}_source.tif'
    label_path = data_dir / key / f'{key}_gt.tif'
    image_slice = tiff.imread(image_path)[slice_index].astype(np.float32)
    label_slice = remap_label_array(tiff.imread(label_path)[slice_index].astype(np.int64))
    return image_slice, label_slice


def find_best_crop(
    label_slice: np.ndarray,
    crop_height: int,
    crop_width: int,
    search_stride: int,
    patch_size: int,
    classes: tuple[int, ...] = DEFAULT_CLASSES,
) -> tuple[tuple[int, int, int, int], np.ndarray, tuple[int, int, int]]:
    height, width = label_slice.shape
    valid_margin = patch_size // 2
    best_score = None
    best_bbox = None
    best_counts = None

    for y0 in range(valid_margin, height - crop_height - valid_margin + 1, search_stride):
        for x0 in range(valid_margin, width - crop_width - valid_margin + 1, search_stride):
            y1, x1 = y0 + crop_height, x0 + crop_width
            crop = label_slice[y0:y1, x0:x1]
            present = set(np.unique(crop[crop >= 0]).tolist())
            if not set(classes).issubset(present):
                continue

            counts = np.bincount(crop[crop >= 0].ravel(), minlength=max(classes) + 1)
            score = ((counts > 0).sum(), int(counts.min()), int(counts.sum()))
            if best_score is None or score > best_score:
                best_score = score
                best_bbox = (y0, y1, x0, x1)
                best_counts = counts.copy()

    if best_bbox is None or best_counts is None or best_score is None:
        raise RuntimeError('Could not find a crop containing all requested classes.')

    return best_bbox, best_counts, best_score


def describe_crop(bbox: tuple[int, int, int, int], slice_shape: tuple[int, int]) -> str:
    y0, y1, x0, x1 = bbox
    crop_height = y1 - y0
    crop_width = x1 - x0
    lines = [
        f'slice shape       : {slice_shape}',
        f'python bbox       : {(y0, y1, x0, x1)}',
        f'top-left          : (y={y0}, x={x0})',
        f'bottom-right      : (y={y1 - 1}, x={x1 - 1})',
        f'height x width    : {crop_height} x {crop_width}',
    ]
    return "\n".join(lines)


def save_crop_png(image_slice: np.ndarray, bbox: tuple[int, int, int, int], output: Path, key: str, slice_index: int) -> Path:
    y0, y1, x0, x1 = bbox
    crop = image_slice[y0:y1, x0:x1]
    output = append_timestamp(default_crop_png_path(key=key, slice_index=slice_index, export_dir=output) if output.suffix == '' else output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.imshow(crop, cmap='gray')
    ax.set_title(f'{key} slice {slice_index} crop\ny={y0}:{y1}, x={x0}:{x1}')
    ax.axis('off')
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output


def main() -> None:
    args = parse_args()
    image_slice, label_slice = load_image_and_label_slice(args.data_dir, args.key, args.slice_index)
    bbox, counts, score = find_best_crop(
        label_slice=label_slice,
        crop_height=args.crop_height,
        crop_width=args.crop_width,
        search_stride=args.search_stride,
        patch_size=args.patch_size,
    )
    saved_output = save_crop_png(image_slice, bbox, args.output, args.key, args.slice_index)

    print(f'key               : {args.key}')
    print(f'slice index       : {args.slice_index}')
    print(describe_crop(bbox, label_slice.shape))
    print(f'class counts      : {counts.tolist()}')
    print(f'search score      : {score}')
    print(f'saved crop png    : {saved_output}')


if __name__ == '__main__':
    main()
