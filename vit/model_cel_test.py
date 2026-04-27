from __future__ import annotations

import argparse
import json
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
    MODEL_CEL_DEFAULT_WANDB_CONFIG_PATH,
    TRAIN_DEFAULT_DATASET_SIZE,
    model_cel_output_dir,
)
from dataloader import _csv_path_for_split, compute_source_stats, load_coordinate_rows
from label_utils import NUM_CLASSES, remap_label_array
from model_kmeans_test import (
    build_datamodule,
    downsample_valid_coords,
    load_checkpoint_bundle,
    load_model,
    load_slice_arrays,
    resolve_checkpoint_path,
)
from model_reconstruction_test import reconstruct_region
from plotting.common import append_timestamp
from plotting.testing import (
    build_confidence_heatmap_figure,
    build_confusion_matrix_figure,
    build_direct_segmentation_figure,
    build_reconstruction_figure,
)
from test_image_extractor import (
    DEFAULT_CROP_HEIGHT,
    DEFAULT_CROP_WIDTH,
    DEFAULT_KEY,
    DEFAULT_PATCH_SIZE,
    DEFAULT_SEARCH_STRIDE,
    DEFAULT_SLICE_INDEX,
    find_best_crop,
)
from train import load_wandb_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run direct classifier-head segmentation on a crop or an entire slice.'
    )
    parser.add_argument('--data-dir', type=Path, default=MODEL_CEL_DEFAULT_DATA_DIR)
    parser.add_argument('--cache-root', type=Path, default=MODEL_CEL_DEFAULT_CACHE_ROOT)
    parser.add_argument('--dataset-root', type=Path, default=None, help='Optional dataset root containing baseline_coords/ and datasets/betaseg/.')
    parser.add_argument('--dataset-size', type=str, default=None, help='Dataset size token used for coordinate CSV lookup.')
    parser.add_argument('--train-coords-csv', type=Path, default=None, help='Optional CSV used to compute normalization statistics.')
    parser.add_argument('--eval-coords-csv', type=Path, default=None, help='Optional CSV describing the eval split metadata.')
    parser.add_argument('--eval-split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--norm-mean', type=float, default=None, help='Explicit normalization mean.')
    parser.add_argument('--norm-std', type=float, default=None, help='Explicit normalization std.')
    parser.add_argument('--output-subdir-name', type=str, default='model_cel_test_images')
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
    parser.add_argument('--crop-y0', type=int, default=None, help='Crop top-left y. If omitted, use the crop identified by test_image_extractor.')
    parser.add_argument('--crop-x0', type=int, default=None, help='Crop top-left x. If omitted, use the crop identified by test_image_extractor.')
    parser.add_argument('--crop-height', type=int, default=DEFAULT_CROP_HEIGHT)
    parser.add_argument('--crop-width', type=int, default=DEFAULT_CROP_WIDTH)
    parser.add_argument(
        '--full-slice',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Evaluate all valid labeled pixels in the full slice instead of only a crop.',
    )
    parser.add_argument('--pixel-batch-size', type=int, default=64)
    parser.add_argument('--pixel-stride', type=int, default=1)
    parser.add_argument('--max-pixels', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--self-test', action='store_true', help='Run lightweight center-token self-tests and exit.')
    parser.add_argument('--output', type=Path, default=None, help='Optional output path for the main segmentation figure.')
    parser.add_argument('--export-dir', type=Path, default=None)
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument(
        '--wandb-config',
        type=Path,
        default=MODEL_CEL_DEFAULT_WANDB_CONFIG_PATH,
        help='Optional JSON file with default Weights & Biases settings.',
    )
    parser.add_argument('--wandb-project', type=str, default=None, help='Enable Weights & Biases logging for this project.')
    parser.add_argument('--wandb-entity', type=str, default=None, help='Weights & Biases entity/team.')
    parser.add_argument('--wandb-group', type=str, default=None, help='Optional Weights & Biases run group.')
    parser.add_argument(
        '--wandb-mode',
        choices=('online', 'offline', 'disabled'),
        default=None,
        help='Weights & Biases mode. Use offline on clusters without outbound internet.',
    )
    parser.add_argument('--wandb-tags', nargs='*', default=None, help='Optional Weights & Biases tags.')
    args = parser.parse_args()
    if (args.norm_mean is None) != (args.norm_std is None):
        raise SystemExit('--norm-mean and --norm-std must be provided together.')
    return args


def _optional_path(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == 'none':
        return None
    return Path(text)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _checkpoint_training_config(checkpoint: dict[str, object]) -> dict[str, object]:
    training_config = checkpoint.get('training_config', {})
    return training_config if isinstance(training_config, dict) else {}


def get_center_token_position(grid_size: int) -> tuple[int, int]:
    if grid_size <= 0:
        raise ValueError(f'grid_size must be positive, got {grid_size}.')
    if grid_size % 2 == 1:
        center = grid_size // 2
        return center, center
    center = grid_size // 2
    return center - 1, center - 1


def extract_center_token_logits(token_logits: torch.Tensor, grid_size: int) -> torch.Tensor:
    if token_logits.ndim != 3:
        raise ValueError(
            f'Expected token_logits with shape [B, num_tokens, num_classes], got {tuple(token_logits.shape)}.'
        )

    row, col = get_center_token_position(grid_size)
    center_token_index = row * grid_size + col
    if token_logits.shape[1] != grid_size * grid_size:
        raise ValueError(
            f'Expected {grid_size * grid_size} tokens for grid_size={grid_size}, got {token_logits.shape[1]}.'
        )
    return token_logits[:, center_token_index, :]


def classify_center_token_logits(token_logits: torch.Tensor, grid_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    center_logits = extract_center_token_logits(token_logits, grid_size=grid_size)
    probabilities = torch.softmax(center_logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    return predictions, probabilities


def run_center_token_self_tests() -> None:
    assert get_center_token_position(16) == (7, 7)

    grid_size = 16
    num_classes = 4
    token_logits = torch.zeros((1, grid_size * grid_size, num_classes), dtype=torch.float32)
    for token_index in range(grid_size * grid_size):
        token_logits[0, token_index] = torch.tensor(
            [token_index, token_index + 1000, token_index + 2000, token_index + 3000],
            dtype=torch.float32,
        )

    center_logits = extract_center_token_logits(token_logits, grid_size=grid_size)
    expected_index = 7 * grid_size + 7
    expected = torch.tensor(
        [[expected_index, expected_index + 1000, expected_index + 2000, expected_index + 3000]],
        dtype=torch.float32,
    )
    assert torch.equal(center_logits, expected)

    token_logits = torch.zeros((3, grid_size * grid_size, num_classes), dtype=torch.float32)
    center_index = 7 * grid_size + 7
    token_logits[:, center_index, :] = torch.tensor(
        [
            [4.0, 1.0, 0.0, -1.0],
            [0.0, 5.0, 1.0, -2.0],
            [-3.0, -1.0, 2.0, 6.0],
        ],
        dtype=torch.float32,
    )

    predictions, probabilities = classify_center_token_logits(token_logits, grid_size=grid_size)
    assert predictions.shape == (3,)
    assert probabilities.shape == (3, num_classes)
    assert torch.equal(predictions, torch.tensor([0, 1, 3]))


def classify_patch_batch(model, batch_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    token_logits, _ = model.predict_token_logits(batch_tensor, mask_ratio=0.0)
    predictions, probabilities = classify_center_token_logits(token_logits, grid_size=model.grid_size)
    return (
        predictions.cpu().numpy().astype(np.int64),
        probabilities.cpu().numpy().astype(np.float32),
    )


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


def collect_region_coords(
    image_slice_norm: np.ndarray,
    label_region: np.ndarray,
    eval_bbox: tuple[int, int, int, int],
    patch_image_size: int,
    pixel_stride: int,
) -> list[tuple[int, int, int, int]]:
    y0, _, x0, _ = eval_bbox
    half = patch_image_size // 2
    valid_coords: list[tuple[int, int, int, int]] = []
    for local_y in range(0, label_region.shape[0], pixel_stride):
        for local_x in range(0, label_region.shape[1], pixel_stride):
            if label_region[local_y, local_x] < 0:
                continue
            global_y = y0 + local_y
            global_x = x0 + local_x
            if global_y < half or global_y >= image_slice_norm.shape[0] - half:
                continue
            if global_x < half or global_x >= image_slice_norm.shape[1] - half:
                continue
            valid_coords.append((local_y, local_x, global_y, global_x))
    return valid_coords


def predict_crop_pixels(
    image_slice_norm: np.ndarray,
    label_slice: np.ndarray,
    crop_bbox: tuple[int, int, int, int],
    model,
    device: torch.device,
    batch_size: int,
    pixel_stride: int,
    max_pixels: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    label_slice = remap_label_array(label_slice)
    y0, y1, x0, x1 = crop_bbox
    crop_labels = label_slice[y0:y1, x0:x1]
    valid_coords = collect_region_coords(
        image_slice_norm=image_slice_norm,
        label_region=crop_labels,
        eval_bbox=crop_bbox,
        patch_image_size=model.image_size,
        pixel_stride=pixel_stride,
    )
    valid_coords = downsample_valid_coords(valid_coords, max_pixels=max_pixels, seed=seed)
    if not valid_coords:
        raise RuntimeError('No valid labeled pixel centers were found inside the chosen crop.')

    half = model.image_size // 2
    class_predictions: list[np.ndarray] = []
    class_probabilities: list[np.ndarray] = []
    class_targets: list[int] = []
    local_positions: list[tuple[int, int]] = []

    for start in tqdm(range(0, len(valid_coords), batch_size), desc='Classifying pixels'):
        batch_coords = valid_coords[start:start + batch_size]
        patches = []
        for local_y, local_x, global_y, global_x in batch_coords:
            patch = image_slice_norm[
                global_y - half:global_y - half + model.image_size,
                global_x - half:global_x - half + model.image_size,
            ]
            patches.append(patch)
            class_targets.append(int(label_slice[global_y, global_x]))
            local_positions.append((local_y, local_x))

        batch_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device=device, dtype=torch.float32)
        predictions, probabilities = classify_patch_batch(model, batch_tensor)
        class_probabilities.append(probabilities)
        class_predictions.append(predictions)

    return (
        np.concatenate(class_predictions, axis=0).astype(np.int64),
        np.concatenate(class_probabilities, axis=0).astype(np.float32),
        np.asarray(class_targets, dtype=np.int64),
        np.asarray(local_positions, dtype=np.int64),
        crop_labels,
    )


def render_prediction_map(target_labels: np.ndarray, local_positions: np.ndarray, class_predictions: np.ndarray) -> np.ndarray:
    prediction_map = np.full_like(target_labels, fill_value=-1)
    for (local_y, local_x), pred in zip(local_positions, class_predictions):
        prediction_map[local_y, local_x] = int(pred)
    return prediction_map


def build_confusion_matrix(class_targets: np.ndarray, class_predictions: np.ndarray, n_classes: int = NUM_CLASSES) -> np.ndarray:
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_class, pred_class in zip(class_targets, class_predictions):
        confusion[int(true_class), int(pred_class)] += 1
    return confusion


def compute_average_confidence_by_true_class(
    class_probabilities: np.ndarray,
    class_targets: np.ndarray,
    class_predictions: np.ndarray,
    correct_only: bool,
    n_classes: int = NUM_CLASSES,
) -> np.ndarray:
    averages = np.full((n_classes, n_classes), np.nan, dtype=np.float32)
    matches = class_predictions == class_targets
    selection_mask = matches if correct_only else ~matches
    for true_class in range(n_classes):
        class_mask = (class_targets == true_class) & selection_mask
        if np.any(class_mask):
            averages[true_class] = class_probabilities[class_mask].mean(axis=0, dtype=np.float64)
    return averages


def compute_per_class_accuracy(confusion_matrix: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for class_id in range(confusion_matrix.shape[0]):
        support = int(confusion_matrix[class_id].sum())
        metrics[f'class_{class_id}_support'] = float(support)
        metrics[f'class_{class_id}_accuracy'] = float(confusion_matrix[class_id, class_id] / support) if support > 0 else float('nan')
    return metrics


def compute_per_class_f1(confusion_matrix: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    f1_values: list[float] = []
    for class_id in range(confusion_matrix.shape[0]):
        true_positive = float(confusion_matrix[class_id, class_id])
        false_positive = float(confusion_matrix[:, class_id].sum() - confusion_matrix[class_id, class_id])
        false_negative = float(confusion_matrix[class_id, :].sum() - confusion_matrix[class_id, class_id])
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1 = (2.0 * true_positive / denominator) if denominator > 0 else float('nan')
        metrics[f'class_{class_id}_f1'] = f1
        if not np.isnan(f1):
            f1_values.append(f1)
    metrics['mean_f1'] = float(np.mean(f1_values)) if f1_values else float('nan')
    return metrics


def build_output_base(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.output is not None:
        return args.output.with_suffix('') if args.output.suffix else args.output
    suffix = 'fullslice' if args.full_slice else 'crop'
    return checkpoint_path.parent / args.output_subdir_name / f'{args.key}_slice{args.slice_index}_model_cel_{suffix}'


def resolve_output_paths(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Path]:
    base = build_output_base(args, checkpoint_path)
    parent = base.parent
    stem = base.name
    return {
        'segmentation': append_timestamp(parent / f'{stem}_segmentation.png'),
        'reconstruction': append_timestamp(parent / f'{stem}_reconstruction.png'),
        'confusion': append_timestamp(parent / f'{stem}_confusion.png'),
        'confidence_correct': append_timestamp(parent / f'{stem}_confidence_correct.png'),
        'confidence_incorrect': append_timestamp(parent / f'{stem}_confidence_incorrect.png'),
    }


def save_metrics_summary(
    result: dict[str, object],
    checkpoint_path: Path,
) -> dict[str, Path]:
    eval_args = result['eval_args']
    base = build_output_base(eval_args, checkpoint_path)
    base.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'checkpoint_path': str(result['checkpoint_path']),
        'key': result['context']['key'],
        'slice_index': int(result['context']['slice_index']),
        'full_slice': bool(eval_args.full_slice),
        'crop_bbox': [int(v) for v in result['crop_bbox']],
        'pixel_batch_size': int(eval_args.pixel_batch_size),
        'pixel_stride': int(eval_args.pixel_stride),
        'max_pixels': None if eval_args.max_pixels is None else int(eval_args.max_pixels),
        'pixel_accuracy': float(result['pixel_accuracy']),
        'mean_f1': float(result['f1_metrics']['mean_f1']),
        'mean_predicted_confidence': float(result['mean_predicted_confidence']),
        'per_class_accuracy': {
            key: float(value) for key, value in result['per_class_metrics'].items()
        },
        'per_class_f1': {
            key: float(value) for key, value in result['f1_metrics'].items()
        },
        'confusion_matrix': result['confusion_matrix'].astype(int).tolist(),
    }

    latest_path = base.parent / f'{base.name}_metrics.json'
    timestamped_path = append_timestamp(base.parent / f'{base.name}_metrics.json')
    json_text = json.dumps(payload, indent=2) + '\n'
    latest_path.write_text(json_text, encoding='utf-8')
    timestamped_path.write_text(json_text, encoding='utf-8')
    return {
        'latest': latest_path,
        'timestamped': timestamped_path,
    }


def export_arrays(
    export_dir: Path,
    args: argparse.Namespace,
    crop_bbox: tuple[int, int, int, int],
    image_region: np.ndarray,
    ground_truth_region: np.ndarray,
    prediction_map: np.ndarray,
    class_targets: np.ndarray,
    class_predictions: np.ndarray,
    class_probabilities: np.ndarray,
    local_positions: np.ndarray,
    confusion_matrix: np.ndarray,
    avg_correct_confidence: np.ndarray,
    avg_incorrect_confidence: np.ndarray,
    reconstruction_region: np.ndarray,
    reconstruction_coverage_mask: np.ndarray,
    pixel_accuracy: float,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    np.save(export_dir / 'image_region.npy', image_region)
    np.save(export_dir / 'ground_truth_region.npy', ground_truth_region)
    np.save(export_dir / 'prediction_map.npy', prediction_map)
    np.save(export_dir / 'ground_truth_classes.npy', class_targets)
    np.save(export_dir / 'class_predictions.npy', class_predictions)
    np.save(export_dir / 'class_probabilities.npy', class_probabilities)
    np.save(export_dir / 'local_positions.npy', local_positions)
    np.save(export_dir / 'confusion_matrix.npy', confusion_matrix)
    np.save(export_dir / 'avg_correct_confidence.npy', avg_correct_confidence)
    np.save(export_dir / 'avg_incorrect_confidence.npy', avg_incorrect_confidence)
    np.save(export_dir / 'reconstruction_region.npy', reconstruction_region)
    np.save(export_dir / 'reconstruction_coverage_mask.npy', reconstruction_coverage_mask)
    np.save(export_dir / 'crop_bbox.npy', np.asarray(crop_bbox, dtype=np.int64))

    metadata = {
        'key': args.key,
        'slice_index': args.slice_index,
        'full_slice': bool(args.full_slice),
        'crop_bbox_python': list(map(int, crop_bbox)),
        'crop_y0': None if args.crop_y0 is None else int(args.crop_y0),
        'crop_x0': None if args.crop_x0 is None else int(args.crop_x0),
        'crop_height': int(args.crop_height),
        'crop_width': int(args.crop_width),
        'pixel_stride': args.pixel_stride,
        'max_pixels': args.max_pixels,
        'pixel_batch_size': args.pixel_batch_size,
        'pixel_accuracy': float(pixel_accuracy),
        'mean_predicted_confidence': float(class_probabilities.max(axis=1).mean()),
    }
    (export_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def load_saved_wandb_metadata(checkpoint: dict[str, object], checkpoint_path: Path) -> dict[str, str] | None:
    saved = checkpoint.get('wandb')
    if isinstance(saved, dict) and saved.get('run_id'):
        return {str(key): str(value) for key, value in saved.items() if value is not None}

    metadata_path = checkpoint_path.parent / 'wandb_run.json'
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding='utf-8'))
        if isinstance(data, dict) and data.get('run_id'):
            return {str(key): str(value) for key, value in data.items() if value is not None}
    return None


def init_wandb_for_eval(
    args: argparse.Namespace,
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    output_dir: Path,
):
    wandb_settings = load_wandb_settings(args.wandb_config)
    run_name = checkpoint_path.parent.name
    mode = args.wandb_mode or wandb_settings.get('mode', 'online')
    if mode == 'disabled':
        return None

    saved_metadata = load_saved_wandb_metadata(checkpoint, checkpoint_path)
    project = args.wandb_project or (saved_metadata or {}).get('project') or wandb_settings.get('project')
    entity = args.wandb_entity or (saved_metadata or {}).get('entity') or wandb_settings.get('entity')
    group = args.wandb_group or wandb_settings.get('group')
    tags = args.wandb_tags if args.wandb_tags is not None else wandb_settings.get('tags')

    if project is None:
        return None

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb logging was requested, but the 'wandb' package is not installed. "
            "Run 'uv sync' or omit --wandb-project."
        ) from exc

    config = {
        'data_dir': str(args.data_dir),
        'dataset_root': None if args.dataset_root is None else str(args.dataset_root),
        'train_coords_csv': None if args.train_coords_csv is None else str(args.train_coords_csv),
        'eval_coords_csv': None if args.eval_coords_csv is None else str(args.eval_coords_csv),
        'cache_root': str(args.cache_root),
        'model_path': str(args.model_path),
        'key': args.key,
        'slice_index': args.slice_index,
        'crop_y0': args.crop_y0,
        'crop_x0': args.crop_x0,
        'crop_height': args.crop_height,
        'crop_width': args.crop_width,
        'pixel_batch_size': args.pixel_batch_size,
        'pixel_stride': args.pixel_stride,
        'max_pixels': args.max_pixels,
        'seed': args.seed,
        'device': args.device,
        'norm_mean': args.norm_mean,
        'norm_std': args.norm_std,
    }
    run_id = None if saved_metadata is None else saved_metadata.get('run_id')
    if run_id:
        return wandb.init(
            entity=entity,
            project=project,
            id=run_id,
            resume='must',
            mode=mode,
            dir=str(output_dir),
            config=config,
        )

    return wandb.init(
        entity=entity,
        project=project,
        group=group,
        tags=tags,
        mode=mode,
        name=run_name,
        dir=str(output_dir),
        config=config,
    )


def _resolve_dataset_root(args: argparse.Namespace, checkpoint: dict[str, object]) -> Path | None:
    if args.dataset_root is not None:
        return args.dataset_root
    training_config = _checkpoint_training_config(checkpoint)
    return _optional_path(training_config.get('dataset_root'))


def _resolve_dataset_size(args: argparse.Namespace, checkpoint: dict[str, object]) -> str:
    if args.dataset_size is not None:
        return args.dataset_size
    training_config = _checkpoint_training_config(checkpoint)
    return _optional_str(training_config.get('dataset_size')) or TRAIN_DEFAULT_DATASET_SIZE


def _resolve_train_coords_csv(
    args: argparse.Namespace,
    checkpoint: dict[str, object],
    dataset_root: Path | None,
    dataset_size: str,
) -> Path | None:
    if args.train_coords_csv is not None:
        return args.train_coords_csv
    training_config = _checkpoint_training_config(checkpoint)
    saved_path = _optional_path(training_config.get('train_coords_csv'))
    if saved_path is not None:
        return saved_path
    if dataset_root is not None:
        return _csv_path_for_split(dataset_root, size=dataset_size, split='train')
    return None


def _resolve_eval_coords_csv(
    args: argparse.Namespace,
    checkpoint: dict[str, object],
    dataset_root: Path | None,
    dataset_size: str,
) -> Path | None:
    if args.eval_coords_csv is not None:
        return args.eval_coords_csv
    training_config = _checkpoint_training_config(checkpoint)
    split_key = f'{args.eval_split}_coords_csv'
    saved_path = _optional_path(training_config.get(split_key))
    if saved_path is not None:
        return saved_path
    if dataset_root is not None:
        return _csv_path_for_split(dataset_root, size=dataset_size, split=args.eval_split)
    return None


def _resolve_data_dir(args: argparse.Namespace, dataset_root: Path | None) -> Path:
    if dataset_root is not None:
        return dataset_root / 'datasets' / 'betaseg'
    return args.data_dir


def _resolve_eval_target(args: argparse.Namespace, eval_coords_csv: Path | None) -> tuple[str, int]:
    if args.key is not None and args.slice_index is not None:
        return str(args.key), int(args.slice_index)
    if eval_coords_csv is not None and eval_coords_csv.exists():
        eval_rows = load_coordinate_rows(eval_coords_csv)
        if not eval_rows:
            raise ValueError(f'Eval CSV contains no rows: {eval_coords_csv}')
        anchor_row = eval_rows[0]
        key = str(args.key) if args.key is not None else anchor_row.name
        slice_index = int(args.slice_index) if args.slice_index is not None else int(anchor_row.z)
        return key, slice_index
    key = str(args.key) if args.key is not None else DEFAULT_KEY
    slice_index = int(args.slice_index) if args.slice_index is not None else DEFAULT_SLICE_INDEX
    return key, slice_index


def resolve_normalization_stats(args: argparse.Namespace, checkpoint: dict[str, object]) -> tuple[float, float, dict[str, object]]:
    if args.norm_mean is not None and args.norm_std is not None:
        return float(args.norm_mean), float(args.norm_std), {'source': 'explicit'}

    dataset_root = _resolve_dataset_root(args, checkpoint)
    dataset_size = _resolve_dataset_size(args, checkpoint)
    train_coords_csv = _resolve_train_coords_csv(args, checkpoint, dataset_root, dataset_size)
    if dataset_root is not None and train_coords_csv is not None:
        train_rows = load_coordinate_rows(train_coords_csv)
        mean, std = compute_source_stats(dataset_root, names=(row.name for row in train_rows))
        return mean, std, {
            'source': 'coordinate_csv',
            'dataset_root': dataset_root,
            'dataset_size': dataset_size,
            'train_coords_csv': train_coords_csv,
        }

    datamodule = build_datamodule(args)
    mean, std = datamodule.get_data_statistics()
    return float(mean), float(std), {
        'source': 'datamodule',
        'data_dir': args.data_dir,
        'cache_root': args.cache_root,
    }


def build_context(args: argparse.Namespace, checkpoint: dict[str, object]) -> dict[str, object]:
    dataset_root = _resolve_dataset_root(args, checkpoint)
    dataset_size = _resolve_dataset_size(args, checkpoint)
    train_coords_csv = _resolve_train_coords_csv(args, checkpoint, dataset_root, dataset_size)
    eval_coords_csv = _resolve_eval_coords_csv(args, checkpoint, dataset_root, dataset_size)
    key, slice_index = _resolve_eval_target(args, eval_coords_csv)
    data_dir = _resolve_data_dir(args, dataset_root)
    data_mean, data_std, normalization_context = resolve_normalization_stats(args, checkpoint)
    return {
        'dataset_root': dataset_root,
        'dataset_size': dataset_size,
        'train_coords_csv': train_coords_csv,
        'eval_coords_csv': eval_coords_csv,
        'data_dir': data_dir,
        'key': key,
        'slice_index': slice_index,
        'data_mean': float(data_mean),
        'data_std': float(data_std),
        'normalization_context': normalization_context,
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    if args.pixel_stride < 1:
        raise ValueError('--pixel-stride must be >= 1')
    if args.max_pixels is not None and args.max_pixels < 1:
        raise ValueError('--max-pixels must be >= 1 when provided')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint_path(args.model_path)
    checkpoint = load_checkpoint_bundle(checkpoint_path)
    context = build_context(args, checkpoint)
    eval_args = argparse.Namespace(**vars(args))
    eval_args.data_dir = context['data_dir']
    eval_args.key = context['key']
    eval_args.slice_index = context['slice_index']

    model = load_model(checkpoint, device=device)
    image_slice, label_slice = load_slice_arrays(context['data_dir'], context['key'], context['slice_index'])
    image_slice_norm = (image_slice - context['data_mean']) / context['data_std']

    eval_bbox = build_eval_bbox(
        args=eval_args,
        label_slice=label_slice,
        use_full_slice=eval_args.full_slice,
    )

    class_predictions, class_probabilities, class_targets, local_positions, gt_region = predict_crop_pixels(
        image_slice_norm=image_slice_norm,
        label_slice=label_slice,
        crop_bbox=eval_bbox,
        model=model,
        device=device,
        batch_size=eval_args.pixel_batch_size,
        pixel_stride=eval_args.pixel_stride,
        max_pixels=eval_args.max_pixels,
        seed=eval_args.seed,
    )

    prediction_map = render_prediction_map(gt_region, local_positions, class_predictions)
    confusion_matrix = build_confusion_matrix(class_targets, class_predictions, n_classes=model.num_classes)
    avg_correct_confidence = compute_average_confidence_by_true_class(
        class_probabilities,
        class_targets,
        class_predictions,
        correct_only=True,
        n_classes=model.num_classes,
    )
    avg_incorrect_confidence = compute_average_confidence_by_true_class(
        class_probabilities,
        class_targets,
        class_predictions,
        correct_only=False,
        n_classes=model.num_classes,
    )
    pixel_accuracy = float((class_predictions == class_targets).mean())
    mean_predicted_confidence = float(class_probabilities.max(axis=1).mean())

    y0, y1, x0, x1 = eval_bbox
    image_region = image_slice[y0:y1, x0:x1]
    reconstruction_result = reconstruct_region(
        image_slice_norm=image_slice_norm,
        image_slice=image_slice,
        eval_bbox=eval_bbox,
        model=model,
        device=device,
        batch_size=eval_args.pixel_batch_size,
        data_mean=context['data_mean'],
        data_std=context['data_std'],
    )
    figures = {
        'segmentation': build_direct_segmentation_figure(
            image_region,
            reconstruction_result['reconstruction'],
            reconstruction_result['coverage_mask'],
            prediction_map,
            gt_region,
            context['key'],
            context['slice_index'],
            None if eval_args.full_slice else eval_bbox,
        ),
        'reconstruction': build_reconstruction_figure(
            image_crop=image_region,
            reconstruction_crop=reconstruction_result['reconstruction'],
            coverage_mask=reconstruction_result['coverage_mask'],
            key=context['key'],
            slice_index=context['slice_index'],
            crop_bbox=None if eval_args.full_slice else eval_bbox,
            effective_local_bbox=reconstruction_result['effective_local_bbox'],
        ),
        'confusion': build_confusion_matrix_figure(confusion_matrix, context['key'], context['slice_index'], None if eval_args.full_slice else eval_bbox),
        'confidence_correct': build_confidence_heatmap_figure(
            avg_correct_confidence,
            title='Average softmax on correct predictions',
            key=context['key'],
            slice_index=context['slice_index'],
            crop_bbox=None if eval_args.full_slice else eval_bbox,
        ),
        'confidence_incorrect': build_confidence_heatmap_figure(
            avg_incorrect_confidence,
            title='Average softmax on incorrect predictions',
            key=context['key'],
            slice_index=context['slice_index'],
            crop_bbox=None if eval_args.full_slice else eval_bbox,
        ),
    }
    output_paths = resolve_output_paths(eval_args, checkpoint_path)

    return {
        'checkpoint_path': checkpoint_path,
        'checkpoint': checkpoint,
        'context': context,
        'eval_args': eval_args,
        'crop_bbox': eval_bbox,
        'image_region': image_region,
        'ground_truth_region': gt_region,
        'prediction_map': prediction_map,
        'reconstruction_region': reconstruction_result['reconstruction'],
        'reconstruction_coverage_mask': reconstruction_result['coverage_mask'],
        'reconstruction_effective_local_bbox': reconstruction_result['effective_local_bbox'],
        'reconstruction_effective_global_bbox': reconstruction_result['effective_global_bbox'],
        'class_targets': class_targets,
        'class_predictions': class_predictions,
        'class_probabilities': class_probabilities,
        'local_positions': local_positions,
        'confusion_matrix': confusion_matrix,
        'avg_correct_confidence': avg_correct_confidence,
        'avg_incorrect_confidence': avg_incorrect_confidence,
        'pixel_accuracy': pixel_accuracy,
        'mean_predicted_confidence': mean_predicted_confidence,
        'per_class_metrics': compute_per_class_accuracy(confusion_matrix),
        'f1_metrics': compute_per_class_f1(confusion_matrix),
        'figures': figures,
        'slice_shape': image_slice.shape,
        'output_paths': output_paths,
    }


def save_figures(figures: dict[str, plt.Figure], output_paths: dict[str, Path]) -> None:
    for name, fig in figures.items():
        output_path = output_paths[name]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches='tight')


def log_wandb(
    wandb_run,
    result: dict[str, object],
    output_paths: dict[str, Path],
) -> None:
    if wandb_run is None:
        return

    import wandb

    confusion_matrix = result['confusion_matrix']
    log_payload: dict[str, object] = {
        'eval/pixel_accuracy': result['pixel_accuracy'],
        'eval/mean_f1': result['f1_metrics']['mean_f1'],
        'eval/mean_predicted_confidence': result['mean_predicted_confidence'],
        'eval/segmentation': wandb.Image(str(output_paths['segmentation'])),
        'eval/reconstruction_plot': wandb.Image(str(output_paths['reconstruction'])),
        'eval/confusion_matrix_plot': wandb.Image(str(output_paths['confusion'])),
        'eval/confidence_correct_plot': wandb.Image(str(output_paths['confidence_correct'])),
        'eval/confidence_incorrect_plot': wandb.Image(str(output_paths['confidence_incorrect'])),
    }
    for metric_name, metric_value in result['per_class_metrics'].items():
        log_payload[f'eval/{metric_name}'] = metric_value
    for metric_name, metric_value in result['f1_metrics'].items():
        log_payload[f'eval/{metric_name}'] = metric_value
    for true_class in range(confusion_matrix.shape[0]):
        for pred_class in range(confusion_matrix.shape[1]):
            log_payload[f'eval/confusion_true{true_class}_pred{pred_class}'] = int(confusion_matrix[true_class, pred_class])
    wandb_run.log(log_payload)
    wandb_run.summary['eval/pixel_accuracy'] = result['pixel_accuracy']
    wandb_run.summary['eval/mean_f1'] = result['f1_metrics']['mean_f1']
    wandb_run.summary['eval/mean_predicted_confidence'] = result['mean_predicted_confidence']


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_center_token_self_tests()
        print('center-token self-tests passed')
        return

    checkpoint_path = resolve_checkpoint_path(args.model_path)
    checkpoint = load_checkpoint_bundle(checkpoint_path)
    context = build_context(args, checkpoint)
    if not context['data_dir'].exists():
        raise FileNotFoundError(f"Missing data directory: {context['data_dir']}")

    result = run_experiment(args)

    wandb_args = argparse.Namespace(**vars(args))
    wandb_args.data_dir = result['context']['data_dir']
    wandb_args.dataset_root = result['context']['dataset_root']
    wandb_args.train_coords_csv = result['context']['train_coords_csv']
    wandb_args.eval_coords_csv = result['context']['eval_coords_csv']
    wandb_args.key = result['context']['key']
    wandb_args.slice_index = result['context']['slice_index']
    wandb_args.norm_mean = result['context']['data_mean']
    wandb_args.norm_std = result['context']['data_std']
    output_dir = args.export_dir if args.export_dir is not None else checkpoint_path.parent
    wandb_run = init_wandb_for_eval(
        wandb_args,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )
    output_paths = result['output_paths']
    save_figures(result['figures'], output_paths)
    metrics_paths = save_metrics_summary(result, checkpoint_path)

    crop_bbox = result['crop_bbox']
    print('checkpoint path            :', result['checkpoint_path'])
    print('data dir                   :', result['context']['data_dir'])
    print('dataset root               :', result['context']['dataset_root'])
    print('dataset size               :', result['context']['dataset_size'])
    print('train coords csv           :', result['context']['train_coords_csv'])
    print('eval coords csv            :', result['context']['eval_coords_csv'])
    print('normalization source       :', result['context']['normalization_context']['source'])
    print('key                        :', result['context']['key'])
    print('slice index                :', result['context']['slice_index'])
    print('slice shape                :', result['slice_shape'])
    print('evaluation mode            :', 'full slice' if args.full_slice else 'crop')
    print('evaluation bbox            :', crop_bbox)
    print('evaluation y-range         :', (crop_bbox[0], crop_bbox[1] - 1))
    print('evaluation x-range         :', (crop_bbox[2], crop_bbox[3] - 1))
    print('evaluation size            :', (crop_bbox[1] - crop_bbox[0], crop_bbox[3] - crop_bbox[2]))
    print('evaluated pixels           :', len(result['class_targets']))
    print('reconstruction bbox        :', result['reconstruction_effective_global_bbox'])
    print('pixel accuracy             :', f"{result['pixel_accuracy']:.4f}")
    print('mean f1                    :', f"{result['f1_metrics']['mean_f1']:.4f}" if not np.isnan(result['f1_metrics']['mean_f1']) else 'nan')
    print('mean predicted confidence  :', f"{result['mean_predicted_confidence']:.4f}")
    print('gt class counts            :', np.bincount(result['class_targets'], minlength=NUM_CLASSES).tolist())
    print('pred class counts          :', np.bincount(result['class_predictions'], minlength=NUM_CLASSES).tolist())
    print('per-class accuracy         :', {k: round(v, 4) if not np.isnan(v) else 'nan' for k, v in result['per_class_metrics'].items() if k.endswith('_accuracy')})
    print('per-class f1               :', {k: round(v, 4) if not np.isnan(v) else 'nan' for k, v in result['f1_metrics'].items() if k.endswith('_f1')})
    print('saved segmentation figure  :', output_paths['segmentation'])
    print('saved reconstruction figure:', output_paths['reconstruction'])
    print('saved confusion figure     :', output_paths['confusion'])
    print('saved correct confidence   :', output_paths['confidence_correct'])
    print('saved incorrect confidence :', output_paths['confidence_incorrect'])
    print('saved metrics summary      :', metrics_paths['latest'])

    if args.export_dir is not None:
        export_args = result['eval_args']
        export_arrays(
            export_dir=args.export_dir,
            args=export_args,
            crop_bbox=result['crop_bbox'],
            image_region=result['image_region'],
            ground_truth_region=result['ground_truth_region'],
            prediction_map=result['prediction_map'],
            class_targets=result['class_targets'],
            class_predictions=result['class_predictions'],
            class_probabilities=result['class_probabilities'],
            local_positions=result['local_positions'],
            confusion_matrix=result['confusion_matrix'],
            avg_correct_confidence=result['avg_correct_confidence'],
            avg_incorrect_confidence=result['avg_incorrect_confidence'],
            reconstruction_region=result['reconstruction_region'],
            reconstruction_coverage_mask=result['reconstruction_coverage_mask'],
            pixel_accuracy=result['pixel_accuracy'],
        )
        export_metadata = {
            'data_dir': str(result['context']['data_dir']),
            'dataset_root': None if result['context']['dataset_root'] is None else str(result['context']['dataset_root']),
            'dataset_size': result['context']['dataset_size'],
            'train_coords_csv': None if result['context']['train_coords_csv'] is None else str(result['context']['train_coords_csv']),
            'eval_coords_csv': None if result['context']['eval_coords_csv'] is None else str(result['context']['eval_coords_csv']),
            'normalization_context': {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in result['context']['normalization_context'].items()
            },
        }
        (args.export_dir / 'evaluation_context.json').write_text(json.dumps(export_metadata, indent=2) + '\n', encoding='utf-8')
        print('saved exports to           :', args.export_dir)

    log_wandb(wandb_run, result, output_paths)
    if wandb_run is not None:
        wandb_run.finish()

    if not args.no_show:
        for figure in result['figures'].values():
            figure.show()
        plt.show()
    else:
        for figure in result['figures'].values():
            plt.close(figure)


if __name__ == '__main__':
    main()
