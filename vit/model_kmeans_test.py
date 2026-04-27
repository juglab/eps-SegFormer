from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm.auto import tqdm

from eps_seg.config.datasets import BetaSegDatasetConfig
from eps_seg.config.train import TrainConfig
from eps_seg.dataloaders.datamodules import EPSSegDataModule

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / 'vit') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'vit'))

from config.app_config import (
    MODEL_KMEANS_DEFAULT_CACHE_ROOT,
    MODEL_KMEANS_DEFAULT_DATA_DIR,
    MODEL_KMEANS_DEFAULT_MODEL_PATH,
    model_kmeans_output_dir,
)
from label_utils import remap_label_array
from models_vit import ViTAutoencoder
from plotting.common import append_timestamp
from plotting.testing import build_kmeans_figure
from test_image_extractor import (
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_KEY,
    DEFAULT_SLICE_INDEX,
)

DEFAULT_CROP_Y0 = 168
DEFAULT_CROP_X0 = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run KMeans on per-pixel ViT embeddings for the fixed test crop.'
    )
    parser.add_argument('--data-dir', type=Path, default=MODEL_KMEANS_DEFAULT_DATA_DIR)
    parser.add_argument('--cache-root', type=Path, default=MODEL_KMEANS_DEFAULT_CACHE_ROOT)
    parser.add_argument(
        '--model-path',
        '--checkpoint',
        dest='model_path',
        type=Path,
        default=MODEL_KMEANS_DEFAULT_MODEL_PATH,
        help='Path to a training run directory or a checkpoint file such as best.pt/last.pt.',
    )
    parser.add_argument('--key', default=DEFAULT_KEY)
    parser.add_argument('--slice-index', type=int, default=DEFAULT_SLICE_INDEX)
    parser.add_argument('--crop-y0', type=int, default=DEFAULT_CROP_Y0)
    parser.add_argument('--crop-x0', type=int, default=DEFAULT_CROP_X0)
    parser.add_argument('--crop-height', type=int, default=DEFAULT_CROP_HEIGHT)
    parser.add_argument('--crop-width', type=int, default=DEFAULT_CROP_WIDTH)
    parser.add_argument('--pixel-batch-size', type=int, default=512)
    parser.add_argument('--pixel-stride', type=int, default=1)
    parser.add_argument('--max-pixels', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--export-dir', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    return parser.parse_args()


def build_datamodule(args: argparse.Namespace) -> EPSSegDataModule:
    dataset_config = BetaSegDatasetConfig(
        dim=2,
        name='betaseg_2d',
        fold=0,
        max_folds=5,
        data_dir=str(args.data_dir),
        cache_dir=str(args.cache_root),
        enable_cache=True,
        train_keys=['high_c1', 'high_c2', 'high_c3'],
        test_keys=[args.key],
        test_center_slices=[args.slice_index],
        test_steppings=[1],
        test_half_depths=[0],
        predict_center_slices=[args.slice_index],
        predict_half_depths=[0],
        seed=args.seed,
        patch_size=32,
        samples_per_class_training={0: 1, 1: 2, 2: 1, 3: 1},
        samples_per_class_validation={0: 2, 1: 4, 2: 2, 3: 2},
    )
    train_config = TrainConfig(
        model_name='vit_autoencoder',
        batch_size=64,
        batches_per_pseudoepoch=100,
        test_batch_size=512,
    )
    datamodule = EPSSegDataModule(dataset_config, train_cfg=train_config)
    datamodule.prepare_data()
    datamodule.setup('fit')
    return datamodule


def load_checkpoint_bundle(checkpoint_path: Path) -> dict[str, object]:
    return torch.load(checkpoint_path, map_location='cpu')


def resolve_checkpoint_path(model_path: Path) -> Path:
    if model_path.is_file():
        return model_path

    if model_path.is_dir():
        for checkpoint_name in ('best.pt', 'last.pt'):
            checkpoint_path = model_path / checkpoint_name
            if checkpoint_path.exists():
                return checkpoint_path
        raise FileNotFoundError(
            f'No checkpoint file found in model directory: {model_path}. '
            'Expected best.pt or last.pt.'
        )

    if model_path.suffix == '.pt':
        raise FileNotFoundError(f'Missing checkpoint file: {model_path}')

    if model_path.exists():
        raise FileNotFoundError(f'Unsupported model path: {model_path}')

    raise FileNotFoundError(f'Missing model path: {model_path}')


def infer_depth(state_dict: dict[str, torch.Tensor]) -> int:
    return len({int(key.split('.')[2]) for key in state_dict if key.startswith('encoder.layers.')})


def load_model(checkpoint: dict[str, object], device: torch.device) -> ViTAutoencoder:
    model_config = checkpoint['model_config']
    state_dict = checkpoint['model_state']
    depth = infer_depth(state_dict)
    training_config = checkpoint.get('training_config', {})
    mlp_ratio = training_config.get('mlp_ratio')
    if mlp_ratio is None:
        mlp_ratio = state_dict['encoder.layers.0.linear1.weight'].shape[0] / model_config['embed_dim']
    num_heads = int(training_config.get('num_heads', 8))
    dropout = float(training_config.get('dropout', 0.0))

    model = ViTAutoencoder(
        image_size=model_config['image_size'],
        patch_size=model_config['patch_size'],
        in_channels=model_config['in_channels'],
        embed_dim=model_config['embed_dim'],
        token_embed_dim=model_config.get('token_embed_dim', model_config['embed_dim']),
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        num_classes=int(model_config.get('num_classes', 4)),
        segmentation_head=str(model_config.get('segmentation_head', 'linear')),
        classifier_context_kernel_size=int(model_config.get('classifier_context_kernel_size', 1)),
        classifier_hidden_dim=model_config.get('classifier_hidden_dim'),
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    return model


def load_slice_arrays(data_dir: Path, key: str, z_index: int) -> tuple[np.ndarray, np.ndarray]:
    image_path = data_dir / key / f'{key}_source.tif'
    label_path = data_dir / key / f'{key}_gt.tif'
    image_volume = tiff.imread(image_path).astype(np.float32)
    label_volume = remap_label_array(tiff.imread(label_path).astype(np.int64))
    return image_volume[z_index], label_volume[z_index]


def get_crop_bbox(args: argparse.Namespace) -> tuple[int, int, int, int]:
    return args.crop_y0, args.crop_y0 + args.crop_height, args.crop_x0, args.crop_x0 + args.crop_width


def validate_crop_bbox(crop_bbox: tuple[int, int, int, int], image_shape: tuple[int, int], model: ViTAutoencoder) -> None:
    y0, y1, x0, x1 = crop_bbox
    height, width = image_shape
    half = model.image_size // 2
    if y0 < 0 or x0 < 0 or y1 > height or x1 > width:
        raise ValueError(f'Crop bbox {crop_bbox} is outside slice shape {(height, width)}.')
    if y0 < half or x0 < half or y1 > height - half or x1 > width - half:
        raise ValueError(f'Crop bbox {crop_bbox} is too close to the border for {model.image_size}x{model.image_size} patches.')


def collect_valid_coords(
    image_slice_norm: np.ndarray,
    crop_labels: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    patch_image_size: int,
    pixel_stride: int,
) -> list[tuple[int, int, int, int]]:
    y0, _, x0, _ = crop_bbox
    half = patch_image_size // 2
    valid_coords = []
    for local_y in range(0, crop_labels.shape[0], pixel_stride):
        for local_x in range(0, crop_labels.shape[1], pixel_stride):
            global_y = y0 + local_y
            global_x = x0 + local_x
            if crop_labels[local_y, local_x] < 0:
                continue
            if global_y < half or global_y >= image_slice_norm.shape[0] - half:
                continue
            if global_x < half or global_x >= image_slice_norm.shape[1] - half:
                continue
            valid_coords.append((local_y, local_x, global_y, global_x))
    return valid_coords


def downsample_valid_coords(valid_coords: list[tuple[int, int, int, int]], max_pixels: int | None, seed: int) -> list[tuple[int, int, int, int]]:
    if max_pixels is None or len(valid_coords) <= max_pixels:
        return valid_coords
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(valid_coords), size=max_pixels, replace=False)
    selected.sort()
    return [valid_coords[index] for index in selected]


def extract_center_embeddings(
    image_slice_norm: np.ndarray,
    label_slice: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    model: ViTAutoencoder,
    device: torch.device,
    batch_size: int,
    pixel_stride: int,
    max_pixels: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y0, y1, x0, x1 = crop_bbox
    crop_labels = label_slice[y0:y1, x0:x1]
    valid_coords = collect_valid_coords(image_slice_norm, crop_labels, crop_bbox, model.image_size, pixel_stride)
    valid_coords = downsample_valid_coords(valid_coords, max_pixels=max_pixels, seed=seed)
    if not valid_coords:
        raise RuntimeError('No valid labeled pixel centers were found inside the chosen crop.')

    half = model.image_size // 2
    feature_vectors = []
    class_targets = []
    local_positions = []

    for start in tqdm(range(0, len(valid_coords), batch_size), desc='Embedding sampled crop pixels'):
        batch_coords = valid_coords[start:start + batch_size]
        patches = []
        for local_y, local_x, global_y, global_x in batch_coords:
            patch = image_slice_norm[global_y - half:global_y + half, global_x - half:global_x + half]
            patches.append(patch)
            class_targets.append(int(label_slice[global_y, global_x]))
            local_positions.append((local_y, local_x))

        batch_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            feature_map = model.extract_embeddings(batch_tensor, pooling='feature_map')
            feature_map = F.interpolate(feature_map, size=(model.image_size, model.image_size), mode='bilinear', align_corners=False)
            center = model.image_size // 2
            center_features = feature_map[:, :, center - 1:center + 1, center - 1:center + 1].mean(dim=(-1, -2))
        feature_vectors.append(center_features.cpu().numpy())

    return (
        np.concatenate(feature_vectors, axis=0),
        np.asarray(class_targets, dtype=np.int64),
        np.asarray(local_positions, dtype=np.int64),
        crop_labels,
    )


def best_cluster_to_class_mapping(cluster_ids: np.ndarray, class_ids: np.ndarray, n_classes: int = 4) -> dict[int, int]:
    best_perm = None
    best_correct = -1
    for perm in itertools.permutations(range(n_classes)):
        mapped = np.array([perm[c] for c in cluster_ids], dtype=np.int64)
        correct = int((mapped == class_ids).sum())
        if correct > best_correct:
            best_correct = correct
            best_perm = perm
    return {cluster: klass for cluster, klass in enumerate(best_perm)}


def render_prediction_map(crop_labels: np.ndarray, local_positions: np.ndarray, class_predictions: np.ndarray) -> np.ndarray:
    prediction_map = np.full_like(crop_labels, fill_value=-1)
    for (local_y, local_x), pred in zip(local_positions, class_predictions):
        prediction_map[local_y, local_x] = int(pred)
    return prediction_map


def export_arrays(
    export_dir: Path,
    args: argparse.Namespace,
    crop_bbox: tuple[int, int, int, int],
    image_crop: np.ndarray,
    crop_gt: np.ndarray,
    prediction_map: np.ndarray,
    features: np.ndarray,
    gt_classes: np.ndarray,
    cluster_ids: np.ndarray,
    class_predictions: np.ndarray,
    local_positions: np.ndarray,
    cluster_to_class: dict[int, int],
    pixel_accuracy: float,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    np.save(export_dir / 'image_crop.npy', image_crop)
    np.save(export_dir / 'ground_truth_crop.npy', crop_gt)
    np.save(export_dir / 'prediction_map.npy', prediction_map)
    np.save(export_dir / 'features.npy', features)
    np.save(export_dir / 'ground_truth_classes.npy', gt_classes)
    np.save(export_dir / 'cluster_ids.npy', cluster_ids)
    np.save(export_dir / 'class_predictions.npy', class_predictions)
    np.save(export_dir / 'local_positions.npy', local_positions)
    np.save(export_dir / 'crop_bbox.npy', np.asarray(crop_bbox, dtype=np.int64))

    metadata = {
        'key': args.key,
        'slice_index': args.slice_index,
        'crop_bbox_python': list(map(int, crop_bbox)),
        'crop_y0': int(args.crop_y0),
        'crop_x0': int(args.crop_x0),
        'crop_height': int(args.crop_height),
        'crop_width': int(args.crop_width),
        'pixel_stride': args.pixel_stride,
        'max_pixels': args.max_pixels,
        'pixel_batch_size': args.pixel_batch_size,
        'feature_shape': list(features.shape),
        'pixel_accuracy': float(pixel_accuracy),
        'cluster_to_class': {str(k): int(v) for k, v in cluster_to_class.items()},
    }
    (export_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')



def format_token(value: object) -> str:
    if isinstance(value, float):
        text = format(value, '.3g')
    else:
        text = str(value)
    return text.replace(' ', '').replace('/', '-').replace('.', 'p').replace('-', 'm')


def build_image_filename(args: argparse.Namespace, checkpoint: dict[str, object], crop_bbox: tuple[int, int, int, int]) -> str:
    model_config = checkpoint.get('model_config', {})
    training_config = checkpoint.get('training_config', {})
    state_dict = checkpoint.get('model_state', {})
    depth = training_config.get('depth', infer_depth(state_dict) if state_dict else 'na')
    num_heads = training_config.get('num_heads', 8)
    epoch = checkpoint.get('epoch', 'na')
    val_loss = checkpoint.get('val_loss', 'na')

    parts = [
        args.key,
        f'slice{args.slice_index}',
        f'y{args.crop_y0}',
        f'x{args.crop_x0}',
        f'h{args.crop_height}',
        f'w{args.crop_width}',
        f'img{model_config.get("image_size", training_config.get("patch_size", "na"))}',
        f'vitp{model_config.get("patch_size", training_config.get("vit_patch_size", "na"))}',
        f'emb{model_config.get("embed_dim", training_config.get("embed_dim", "na"))}',
        f'd{depth}',
        f'hd{num_heads}',
        f'bs{training_config.get("batch_size", "na")}',
        f'lr{format_token(training_config.get("lr", "na"))}',
        f'wd{format_token(training_config.get("weight_decay", "na"))}',
        f'seed{training_config.get("seed", args.seed)}',
        f'ep{epoch}',
        f'val{format_token(val_loss)}',
    ]
    return '_'.join(str(part) for part in parts) + '.png'


def resolve_output_path(args: argparse.Namespace, checkpoint_path: Path, checkpoint: dict[str, object], crop_bbox: tuple[int, int, int, int]) -> Path:
    if args.output is not None:
        return append_timestamp(args.output)
    output_dir = model_kmeans_output_dir(checkpoint_path.parent)
    filename = build_image_filename(args, checkpoint, crop_bbox)
    return append_timestamp(output_dir / filename)


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.pixel_stride < 1:
        raise ValueError('--pixel-stride must be >= 1')
    if args.max_pixels is not None and args.max_pixels < 1:
        raise ValueError('--max-pixels must be >= 1 when provided')

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

    crop_bbox = get_crop_bbox(args)
    validate_crop_bbox(crop_bbox, image_slice.shape, model)

    features, gt_classes, local_positions, crop_gt = extract_center_embeddings(
        image_slice_norm=image_slice_norm,
        label_slice=label_slice,
        crop_bbox=crop_bbox,
        model=model,
        device=device,
        batch_size=args.pixel_batch_size,
        pixel_stride=args.pixel_stride,
        max_pixels=args.max_pixels,
        seed=args.seed,
    )

    kmeans = KMeans(n_clusters=4, n_init='auto', random_state=args.seed)
    cluster_ids = kmeans.fit_predict(features)
    cluster_to_class = best_cluster_to_class_mapping(cluster_ids, gt_classes, n_classes=4)
    class_predictions = np.array([cluster_to_class[c] for c in cluster_ids], dtype=np.int64)
    prediction_map = render_prediction_map(crop_gt, local_positions, class_predictions)

    pixel_accuracy = float((class_predictions == gt_classes).mean())
    y0, y1, x0, x1 = crop_bbox
    image_crop = image_slice[y0:y1, x0:x1]
    fig = build_kmeans_figure(image_crop, prediction_map, crop_gt, args.key, args.slice_index, crop_bbox)
    output_path = resolve_output_path(args, checkpoint_path, checkpoint, crop_bbox)

    return {
        'checkpoint_path': checkpoint_path,
        'checkpoint': checkpoint,
        'crop_bbox': crop_bbox,
        'image_crop': image_crop,
        'crop_gt': crop_gt,
        'prediction_map': prediction_map,
        'features': features,
        'gt_classes': gt_classes,
        'cluster_ids': cluster_ids,
        'class_predictions': class_predictions,
        'local_positions': local_positions,
        'cluster_to_class': cluster_to_class,
        'pixel_accuracy': pixel_accuracy,
        'figure': fig,
        'slice_shape': image_slice.shape,
        'output_path': output_path,
    }


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f'Missing data directory: {args.data_dir}')

    result = run_experiment(args)
    crop_bbox = result['crop_bbox']

    print('checkpoint path :', result['checkpoint_path'])
    print('slice shape     :', result['slice_shape'])
    print('crop bbox       :', crop_bbox)
    print('crop y-range    :', (crop_bbox[0], crop_bbox[1] - 1))
    print('crop x-range    :', (crop_bbox[2], crop_bbox[3] - 1))
    print('crop size       :', (args.crop_height, args.crop_width))
    print('sampled pixels  :', len(result['gt_classes']))
    print('feature shape   :', result['features'].shape)
    print('cluster mapping :', result['cluster_to_class'])
    print('pixel accuracy  :', f"{result['pixel_accuracy']:.4f}")
    print('gt class counts :', np.bincount(result['gt_classes'], minlength=4).tolist())
    print('pixel stride    :', args.pixel_stride)
    print('max pixels      :', args.max_pixels)

    fig = result['figure']
    output_path = result['output_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print('saved figure to :', output_path)

    if args.export_dir is not None:
        export_arrays(
            export_dir=args.export_dir,
            args=args,
            crop_bbox=result['crop_bbox'],
            image_crop=result['image_crop'],
            crop_gt=result['crop_gt'],
            prediction_map=result['prediction_map'],
            features=result['features'],
            gt_classes=result['gt_classes'],
            cluster_ids=result['cluster_ids'],
            class_predictions=result['class_predictions'],
            local_positions=result['local_positions'],
            cluster_to_class=result['cluster_to_class'],
            pixel_accuracy=result['pixel_accuracy'],
        )
        print('saved exports to:', args.export_dir)

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
