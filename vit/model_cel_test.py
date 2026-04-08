from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
    model_cel_output_dir,
)
from model_kmeans_test import (
    build_datamodule,
    downsample_valid_coords,
    load_checkpoint_bundle,
    load_model,
    load_slice_arrays,
    resolve_checkpoint_path,
)
from label_utils import NUM_CLASSES, remap_label_array
from plotting.common import append_timestamp
from plotting.testing import (
    build_confidence_heatmap_figure,
    build_confusion_matrix_figure,
    build_direct_segmentation_figure,
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
    parser.add_argument('--full-slice', action='store_true', help='Evaluate all valid labeled pixels in the full slice instead of only a crop.')
    parser.add_argument('--pixel-batch-size', type=int, default=64)
    parser.add_argument('--pixel-stride', type=int, default=1)
    parser.add_argument('--max-pixels', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
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
    return parser.parse_args()


def extract_center_logits(logit_maps: torch.Tensor, image_size: int) -> torch.Tensor:
    center = image_size // 2
    if image_size % 2 == 0:
        return logit_maps[:, :, center - 1:center + 1, center - 1:center + 1].mean(dim=(-1, -2))
    return logit_maps[:, :, center:center + 1, center:center + 1].squeeze(-1).squeeze(-1)


def classify_patch_batch(model, batch_tensor: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    token_logits, _ = model.predict_token_logits(batch_tensor, mask_ratio=0.0)
    grid_size = model.grid_size
    logits_map = token_logits.transpose(1, 2).reshape(batch_tensor.shape[0], model.num_classes, grid_size, grid_size)
    logits_map = F.interpolate(
        logits_map,
        size=(model.image_size, model.image_size),
        mode='bilinear',
        align_corners=False,
    )
    center_logits = extract_center_logits(logits_map, image_size=model.image_size)
    probabilities = F.softmax(center_logits, dim=1)
    predictions = probabilities.argmax(dim=1)
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
    label_region: np.ndarray,
    eval_bbox: tuple[int, int, int, int],
    pixel_stride: int,
) -> list[tuple[int, int, int, int]]:
    y0, _, x0, _ = eval_bbox
    valid_coords: list[tuple[int, int, int, int]] = []
    for local_y in range(0, label_region.shape[0], pixel_stride):
        for local_x in range(0, label_region.shape[1], pixel_stride):
            if label_region[local_y, local_x] < 0:
                continue
            valid_coords.append((local_y, local_x, y0 + local_y, x0 + local_x))
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
    valid_coords = collect_region_coords(crop_labels, crop_bbox, pixel_stride)
    valid_coords = downsample_valid_coords(valid_coords, max_pixels=max_pixels, seed=seed)
    if not valid_coords:
        raise RuntimeError('No valid labeled pixel centers were found inside the chosen crop.')

    half = model.image_size // 2
    padded_image = np.pad(image_slice_norm, pad_width=((half, half), (half, half)), mode='reflect')
    class_predictions: list[np.ndarray] = []
    class_probabilities: list[np.ndarray] = []
    class_targets: list[int] = []
    local_positions: list[tuple[int, int]] = []

    for start in tqdm(range(0, len(valid_coords), batch_size), desc='Classifying pixels'):
        batch_coords = valid_coords[start:start + batch_size]
        patches = []
        for local_y, local_x, global_y, global_x in batch_coords:
            patch = padded_image[global_y:global_y + model.image_size, global_x:global_x + model.image_size]
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


def build_output_base(args: argparse.Namespace, checkpoint_path: Path) -> Path:
    if args.output is not None:
        return args.output.with_suffix('') if args.output.suffix else args.output
    suffix = 'fullslice' if args.full_slice else 'crop'
    return model_cel_output_dir(checkpoint_path.parent) / f'{args.key}_slice{args.slice_index}_model_cel_{suffix}'


def resolve_output_paths(args: argparse.Namespace, checkpoint_path: Path) -> dict[str, Path]:
    base = build_output_base(args, checkpoint_path)
    parent = base.parent
    stem = base.name
    return {
        'segmentation': append_timestamp(parent / f'{stem}_segmentation.png'),
        'confusion': append_timestamp(parent / f'{stem}_confusion.png'),
        'confidence_correct': append_timestamp(parent / f'{stem}_confidence_correct.png'),
        'confidence_incorrect': append_timestamp(parent / f'{stem}_confidence_incorrect.png'),
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

    eval_bbox = build_eval_bbox(
        args=args,
        label_slice=label_slice,
        use_full_slice=args.full_slice,
    )

    class_predictions, class_probabilities, class_targets, local_positions, gt_region = predict_crop_pixels(
        image_slice_norm=image_slice_norm,
        label_slice=label_slice,
        crop_bbox=eval_bbox,
        model=model,
        device=device,
        batch_size=args.pixel_batch_size,
        pixel_stride=args.pixel_stride,
        max_pixels=args.max_pixels,
        seed=args.seed,
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
    figures = {
        'segmentation': build_direct_segmentation_figure(
            image_region,
            prediction_map,
            gt_region,
            args.key,
            args.slice_index,
            None if args.full_slice else eval_bbox,
        ),
        'confusion': build_confusion_matrix_figure(confusion_matrix, args.key, args.slice_index, None if args.full_slice else eval_bbox),
        'confidence_correct': build_confidence_heatmap_figure(
            avg_correct_confidence,
            title='Average softmax on correct predictions',
            key=args.key,
            slice_index=args.slice_index,
            crop_bbox=None if args.full_slice else eval_bbox,
        ),
        'confidence_incorrect': build_confidence_heatmap_figure(
            avg_incorrect_confidence,
            title='Average softmax on incorrect predictions',
            key=args.key,
            slice_index=args.slice_index,
            crop_bbox=None if args.full_slice else eval_bbox,
        ),
    }
    output_paths = resolve_output_paths(args, checkpoint_path)

    return {
        'checkpoint_path': checkpoint_path,
        'checkpoint': checkpoint,
        'crop_bbox': eval_bbox,
        'image_region': image_region,
        'ground_truth_region': gt_region,
        'prediction_map': prediction_map,
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
        'eval/mean_predicted_confidence': result['mean_predicted_confidence'],
        'eval/segmentation': wandb.Image(str(output_paths['segmentation'])),
        'eval/confusion_matrix_plot': wandb.Image(str(output_paths['confusion'])),
        'eval/confidence_correct_plot': wandb.Image(str(output_paths['confidence_correct'])),
        'eval/confidence_incorrect_plot': wandb.Image(str(output_paths['confidence_incorrect'])),
    }
    for metric_name, metric_value in result['per_class_metrics'].items():
        log_payload[f'eval/{metric_name}'] = metric_value
    for true_class in range(confusion_matrix.shape[0]):
        for pred_class in range(confusion_matrix.shape[1]):
            log_payload[f'eval/confusion_true{true_class}_pred{pred_class}'] = int(confusion_matrix[true_class, pred_class])
    wandb_run.log(log_payload)
    wandb_run.summary['eval/pixel_accuracy'] = result['pixel_accuracy']
    wandb_run.summary['eval/mean_predicted_confidence'] = result['mean_predicted_confidence']


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f'Missing data directory: {args.data_dir}')

    checkpoint_path = resolve_checkpoint_path(args.model_path)
    checkpoint = load_checkpoint_bundle(checkpoint_path)
    output_dir = args.export_dir if args.export_dir is not None else checkpoint_path.parent
    wandb_run = init_wandb_for_eval(
        args,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )

    result = run_experiment(args)
    output_paths = result['output_paths']
    save_figures(result['figures'], output_paths)

    crop_bbox = result['crop_bbox']
    print('checkpoint path            :', result['checkpoint_path'])
    print('slice shape                :', result['slice_shape'])
    print('evaluation mode            :', 'full slice' if args.full_slice else 'crop')
    print('evaluation bbox            :', crop_bbox)
    print('evaluation y-range         :', (crop_bbox[0], crop_bbox[1] - 1))
    print('evaluation x-range         :', (crop_bbox[2], crop_bbox[3] - 1))
    print('evaluation size            :', (crop_bbox[1] - crop_bbox[0], crop_bbox[3] - crop_bbox[2]))
    print('evaluated pixels           :', len(result['class_targets']))
    print('pixel accuracy             :', f"{result['pixel_accuracy']:.4f}")
    print('mean predicted confidence  :', f"{result['mean_predicted_confidence']:.4f}")
    print('gt class counts            :', np.bincount(result['class_targets'], minlength=NUM_CLASSES).tolist())
    print('pred class counts          :', np.bincount(result['class_predictions'], minlength=NUM_CLASSES).tolist())
    print('per-class accuracy         :', {k: round(v, 4) if not np.isnan(v) else 'nan' for k, v in result['per_class_metrics'].items() if k.endswith('_accuracy')})
    print('saved segmentation figure  :', output_paths['segmentation'])
    print('saved confusion figure     :', output_paths['confusion'])
    print('saved correct confidence   :', output_paths['confidence_correct'])
    print('saved incorrect confidence :', output_paths['confidence_incorrect'])

    if args.export_dir is not None:
        export_arrays(
            export_dir=args.export_dir,
            args=args,
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
            pixel_accuracy=result['pixel_accuracy'],
        )
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
