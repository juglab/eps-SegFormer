from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'vit') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'vit'))

from config.app_config import (
    MODEL_CEL_DEFAULT_CACHE_ROOT,
    MODEL_CEL_DEFAULT_DATA_DIR,
    MODEL_CEL_DEFAULT_MODEL_PATH,
)
from label_utils import remap_label_array
from model_kmeans_test import (
    build_datamodule,
    load_checkpoint_bundle,
    load_model,
    load_slice_arrays,
    resolve_checkpoint_path,
)
from plotting.common import append_timestamp
from plotting.testing import build_reconstruction_figure
from test_image_extractor import (
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_KEY,
    DEFAULT_PATCH_SIZE,
    DEFAULT_SEARCH_STRIDE,
    DEFAULT_SLICE_INDEX,
    find_best_crop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Reconstruct a crop or full slice with non-overlapping ViT autoencoder tiles.'
    )
    parser.add_argument('--data-dir', type=Path, default=MODEL_CEL_DEFAULT_DATA_DIR)
    parser.add_argument('--cache-root', type=Path, default=MODEL_CEL_DEFAULT_CACHE_ROOT)
    parser.add_argument(
        '--model-path',
        '--checkpoint',
        dest='model_path',
        type=Path,
        default=MODEL_CEL_DEFAULT_MODEL_PATH,
        help='Path to a training run directory or a checkpoint file such as best.pt/last.pt.',
    )
    parser.add_argument('--key', default=DEFAULT_KEY)
    parser.add_argument('--slice-index', type=int, default=DEFAULT_SLICE_INDEX)
    parser.add_argument('--crop-y0', type=int, default=None)
    parser.add_argument('--crop-x0', type=int, default=None)
    parser.add_argument('--crop-height', type=int, default=DEFAULT_CROP_HEIGHT)
    parser.add_argument('--crop-width', type=int, default=DEFAULT_CROP_WIDTH)
    parser.add_argument('--full-slice', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    return parser.parse_args()


def build_eval_bbox(
    args: argparse.Namespace,
    label_slice: np.ndarray,
    use_full_slice: bool,
) -> tuple[int, int, int, int]:
    height, width = label_slice.shape
    if use_full_slice:
        return 0, height, 0, width

    if args.crop_y0 is None or args.crop_x0 is None:
        crop_bbox, _, _ = find_best_crop(
            label_slice=remap_label_array(label_slice),
            crop_height=args.crop_height,
            crop_width=args.crop_width,
            search_stride=DEFAULT_SEARCH_STRIDE,
            patch_size=DEFAULT_PATCH_SIZE,
        )
        return crop_bbox

    y0 = int(args.crop_y0)
    x0 = int(args.crop_x0)
    y1 = y0 + int(args.crop_height)
    x1 = x0 + int(args.crop_width)
    if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
        raise ValueError(f'Crop bbox {(y0, y1, x0, x1)} is outside slice shape {(height, width)}.')
    return y0, y1, x0, x1


def reconstruct_region(
    image_slice_norm: np.ndarray,
    image_slice: np.ndarray,
    eval_bbox: tuple[int, int, int, int],
    model,
    device: torch.device,
    batch_size: int,
    data_mean: float,
    data_std: float,
) -> dict[str, object]:
    y0, y1, x0, x1 = eval_bbox
    tile_size = int(model.image_size)
    region_height = y1 - y0
    region_width = x1 - x0
    if region_height < tile_size or region_width < tile_size:
        raise ValueError(
            f'Evaluation region {(region_height, region_width)} is smaller than one reconstruction tile '
            f'of size {tile_size}x{tile_size}.'
        )

    tiles_y = region_height // tile_size
    tiles_x = region_width // tile_size
    covered_height = tiles_y * tile_size
    covered_width = tiles_x * tile_size
    reconstruction = np.full((region_height, region_width), np.nan, dtype=np.float32)
    coverage_mask = np.zeros((region_height, region_width), dtype=bool)
    image_region = image_slice[y0:y1, x0:x1]

    tile_coords = [
        (tile_y, tile_x, y0 + tile_y * tile_size, x0 + tile_x * tile_size)
        for tile_y in range(tiles_y)
        for tile_x in range(tiles_x)
    ]
    for start in tqdm(range(0, len(tile_coords), batch_size), desc='Reconstructing tiles'):
        batch_coords = tile_coords[start:start + batch_size]
        patches = [
            image_slice_norm[
                global_y:global_y + tile_size,
                global_x:global_x + tile_size,
            ]
            for _, _, global_y, global_x in batch_coords
        ]
        batch_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            aux = model.forward_with_aux(batch_tensor, mask_ratio=0.0)
            reconstructed_batch = aux.reconstruction[:, 0].cpu().numpy().astype(np.float32)
        reconstructed_batch = reconstructed_batch * float(data_std) + float(data_mean)

        for (tile_y, tile_x, _, _), reconstructed_patch in zip(batch_coords, reconstructed_batch):
            local_y0 = tile_y * tile_size
            local_x0 = tile_x * tile_size
            reconstruction[
                local_y0:local_y0 + tile_size,
                local_x0:local_x0 + tile_size,
            ] = reconstructed_patch
            coverage_mask[
                local_y0:local_y0 + tile_size,
                local_x0:local_x0 + tile_size,
            ] = True

    return {
        'image_region': image_region,
        'reconstruction': reconstruction,
        'coverage_mask': coverage_mask,
        'effective_local_bbox': (0, covered_height, 0, covered_width),
        'effective_global_bbox': (y0, y0 + covered_height, x0, x0 + covered_width),
        'tile_size': tile_size,
        'tile_grid_shape': (tiles_y, tiles_x),
    }


def build_output_path(output: Path | None, checkpoint_path: Path, key: str, slice_index: int, full_slice: bool) -> Path:
    if output is not None:
        return output
    suffix = 'fullslice' if full_slice else 'crop'
    return append_timestamp(
        checkpoint_path.parent / 'model_reconstruction_test_images' / f'{key}_slice{slice_index}_reconstruction_{suffix}.png'
    )


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    datamodule = build_datamodule(args)
    data_mean, data_std = datamodule.get_data_statistics()
    checkpoint_path = resolve_checkpoint_path(args.model_path)
    checkpoint = load_checkpoint_bundle(checkpoint_path)
    model = load_model(checkpoint, device=device)
    image_slice, label_slice = load_slice_arrays(args.data_dir, args.key, args.slice_index)
    image_slice_norm = (image_slice - float(data_mean)) / float(data_std)
    eval_bbox = build_eval_bbox(args=args, label_slice=label_slice, use_full_slice=args.full_slice)
    reconstruction_result = reconstruct_region(
        image_slice_norm=image_slice_norm,
        image_slice=image_slice,
        eval_bbox=eval_bbox,
        model=model,
        device=device,
        batch_size=args.batch_size,
        data_mean=float(data_mean),
        data_std=float(data_std),
    )
    output_path = build_output_path(args.output, checkpoint_path, args.key, args.slice_index, args.full_slice)
    figure = build_reconstruction_figure(
        image_crop=reconstruction_result['image_region'],
        reconstruction_crop=reconstruction_result['reconstruction'],
        coverage_mask=reconstruction_result['coverage_mask'],
        key=args.key,
        slice_index=args.slice_index,
        crop_bbox=None if args.full_slice else eval_bbox,
        effective_local_bbox=reconstruction_result['effective_local_bbox'],
    )
    return {
        'checkpoint_path': checkpoint_path,
        'crop_bbox': eval_bbox,
        'figure': figure,
        'output_path': output_path,
        **reconstruction_result,
    }


def main() -> None:
    args = parse_args()
    result = run_experiment(args)
    output_path = result['output_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result['figure'].savefig(output_path, dpi=200, bbox_inches='tight')
    print('checkpoint path         :', result['checkpoint_path'])
    print('evaluation bbox         :', result['crop_bbox'])
    print('effective global bbox   :', result['effective_global_bbox'])
    print('tile grid shape         :', result['tile_grid_shape'])
    print('saved reconstruction    :', output_path)

    if not args.no_show:
        result['figure'].show()
        plt.show()
    else:
        plt.close(result['figure'])


if __name__ == '__main__':
    main()
