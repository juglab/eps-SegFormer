import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.app_config import (
    TRAIN_DEFAULT_CACHE_ROOT,
    TRAIN_DEFAULT_DATA_DIR,
    TRAIN_DEFAULT_DATASET_ROOT,
    TRAIN_DEFAULT_DATASET_SIZE,
    TRAIN_DEFAULT_OUTPUT_DIR,
    TRAIN_DEFAULT_TRAIN_COORDS_CSV,
    TRAIN_DEFAULT_VAL_COORDS_CSV,
    TRAIN_DEFAULT_WANDB_CONFIG_PATH,
)
from dataloader import build_train_val_loaders
from label_utils import remap_label_tensor, valid_class_mask
from models_vit import ViTAutoencoder
from plotting.training import plot_training_history

DEFAULT_WANDB_SETTINGS: dict[str, object] = {
    "entity": "juglab",
    "project": "eps_segformer",
    "mode": "online",
    "tags": ["vit", "autoencoder", "configurable-head"],
}
LOSS_MODES = ("ce_inpaint", "ce_reconstruct_all", "ce_reconstruct_visible")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ViT autoencoder on BetaSeg2D patches.")
    parser.add_argument("--dataset-root", type=Path, default=TRAIN_DEFAULT_DATASET_ROOT, help="Root containing baseline_coords/ and datasets/betaseg/.")
    parser.add_argument("--dataset-size", type=str, default=TRAIN_DEFAULT_DATASET_SIZE, help="Dataset size token used to select 2D_<size>_<split>.csv.")
    parser.add_argument("--data-dir", type=Path, default=TRAIN_DEFAULT_DATA_DIR, help="Override path to the betaseg dataset root.")
    parser.add_argument("--cache-root", type=Path, default=TRAIN_DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=TRAIN_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-coords-csv", type=Path, default=TRAIN_DEFAULT_TRAIN_COORDS_CSV)
    parser.add_argument("--val-coords-csv", type=Path, default=TRAIN_DEFAULT_VAL_COORDS_CSV)
    parser.add_argument(
        "legacy_run_name",
        nargs="?",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--run-name", type=str, default=None, help="Optional run directory name.")
    parser.add_argument("--train-keys", nargs="+", default=["high_c1", "high_c2", "high_c3"])
    parser.add_argument("--test-keys", nargs="+", default=["high_c4"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=25, help="BetaSeg2D patch size returned by the dataloader.")
    parser.add_argument("--vit-patch-size", type=int, default=5, help="Patch size used inside the ViT encoder.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches-per-pseudoepoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=192)
    parser.add_argument("--token-embed-dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--mask-ratio", type=float, default=0.00)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--loss-mode",
        choices=LOSS_MODES,
        default="ce_reconstruct_all",
        help="Reconstruction term combined with visible-token cross-entropy.",
    )
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Segmentation head configuration:
    # - linear: original per-token classifier
    # - neighbor_concat: ordered local neighborhood embeddings concatenated together
    parser.add_argument(
        "--segmentation-head",
        choices=("linear", "neighbor_concat"),
        default="linear",
        help="Segmentation head applied to encoder tokens.",
    )
    parser.add_argument(
        "--classifier-context-kernel-size",
        type=int,
        default=3,
        help="Odd neighborhood size over token grid used by the context-aware segmentation heads.",
    )
    parser.add_argument(
        "--classifier-hidden-dim",
        type=int,
        default=None,
        help="Optional hidden size for the context-aware segmentation heads.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument(
        "--wandb-config",
        type=Path,
        default=TRAIN_DEFAULT_WANDB_CONFIG_PATH,
        help="Optional JSON file with default Weights & Biases settings.",
    )
    parser.add_argument("--wandb-project", type=str, default=None, help="Enable Weights & Biases logging for this project.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Weights & Biases entity/team.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional Weights & Biases run group.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Weights & Biases mode. Use offline on clusters without outbound internet.",
    )
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional Weights & Biases tags.")
    parser.add_argument("--self-test", action="store_true", help="Run lightweight loss/shape self-tests and exit.")
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = args.legacy_run_name
    return args


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, tuple[float, float]]:
    return build_train_val_loaders(
        dataset_root=args.dataset_root,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        train_coords_csv=args.train_coords_csv,
        val_coords_csv=args.val_coords_csv,
    )


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, object | None]:
    if isinstance(batch, (tuple, list)):
        patches = batch[0].float()
        labels = batch[1] if len(batch) > 1 else None
        segments = batch[2] if len(batch) > 2 else None
        coords = batch[3] if len(batch) > 3 else None
        return patches, labels, segments, coords
    return batch.float(), None, None, None


def infer_in_channels(loader: Iterable) -> int:
    sample_batch = next(iter(loader))
    patches, _, _, _ = unpack_batch(sample_batch)
    return int(patches.shape[1])


def require_segments(batch) -> tuple[torch.Tensor, torch.Tensor]:
    patches, _, segments, _ = unpack_batch(batch)
    if segments is None:
        raise ValueError(
            "Training batches must include ground-truth label patches as the third item "
            "so the center-pixel classification loss can be computed."
        )
    return patches.float(), segments


def extract_token_center_targets(label_patches: torch.Tensor, vit_patch_size: int) -> torch.Tensor:
    if label_patches.ndim != 4:
        raise ValueError(f"Expected label patches with shape [B, 1, H, W], got {tuple(label_patches.shape)}.")
    _, channels, height, width = label_patches.shape
    if channels != 1:
        raise ValueError(f"Expected single-channel label patches, got {channels} channels.")
    if height % vit_patch_size != 0 or width % vit_patch_size != 0:
        raise ValueError(
            f"Label patch size {(height, width)} must be divisible by vit_patch_size={vit_patch_size}."
        )

    center_index = vit_patch_size // 2
    grid_h = height // vit_patch_size
    grid_w = width // vit_patch_size
    remapped_labels = remap_label_tensor(label_patches)
    reshaped = remapped_labels.reshape(
        label_patches.shape[0],
        1,
        grid_h,
        vit_patch_size,
        grid_w,
        vit_patch_size,
    )
    reshaped = reshaped.permute(0, 2, 4, 1, 3, 5)
    token_centers = reshaped[..., 0, center_index, center_index]
    return token_centers.reshape(label_patches.shape[0], grid_h * grid_w)


def compute_center_classification_metrics(
    token_logits: torch.Tensor,
    token_targets: torch.Tensor,
    visible_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    token_targets = remap_label_tensor(token_targets)
    supervised = (visible_mask & valid_class_mask(token_targets)).reshape(-1)
    if supervised.sum().item() == 0:
        zero = token_logits.new_zeros(())
        return zero, zero, 0

    flat_logits = token_logits.reshape(-1, token_logits.shape[-1])[supervised]
    flat_targets = token_targets.reshape(-1)[supervised]
    ce_loss = F.cross_entropy(flat_logits, flat_targets)
    predictions = flat_logits.argmax(dim=-1)
    accuracy = (predictions == flat_targets).float().mean()
    return ce_loss, accuracy, int(flat_targets.numel())


def expand_visible_mask_to_pixels(
    visible_mask: torch.Tensor,
    vit_patch_size: int,
    image_shape: tuple[int, int],
) -> torch.Tensor:
    if visible_mask.ndim != 2:
        raise ValueError(f"Expected visible_mask with shape [B, N], got {tuple(visible_mask.shape)}.")

    height, width = image_shape
    if height % vit_patch_size != 0 or width % vit_patch_size != 0:
        raise ValueError(
            f"Image shape {(height, width)} must be divisible by vit_patch_size={vit_patch_size}."
        )

    grid_h = height // vit_patch_size
    grid_w = width // vit_patch_size
    if visible_mask.shape[1] != grid_h * grid_w:
        raise ValueError(
            f"Expected {grid_h * grid_w} tokens for image shape {(height, width)}, got {visible_mask.shape[1]}."
        )

    pixel_mask = visible_mask.view(visible_mask.shape[0], grid_h, grid_w)
    pixel_mask = pixel_mask.repeat_interleave(vit_patch_size, dim=1)
    pixel_mask = pixel_mask.repeat_interleave(vit_patch_size, dim=2)
    return pixel_mask.unsqueeze(1)


def compute_reconstruction_loss(
    reconstruction: torch.Tensor,
    patches: torch.Tensor,
    visible_mask: torch.Tensor,
    vit_patch_size: int,
    loss_mode: str,
) -> torch.Tensor:
    if loss_mode == "ce_reconstruct_all":
        return F.mse_loss(reconstruction, patches)

    pixel_visible_mask = expand_visible_mask_to_pixels(
        visible_mask,
        vit_patch_size=vit_patch_size,
        image_shape=(patches.shape[-2], patches.shape[-1]),
    )
    if loss_mode == "ce_inpaint":
        selected = ~pixel_visible_mask
    elif loss_mode == "ce_reconstruct_visible":
        selected = pixel_visible_mask
    else:
        raise ValueError(f"Unsupported loss_mode '{loss_mode}'. Use one of {LOSS_MODES}.")

    selected = selected.expand_as(reconstruction)
    if not selected.any():
        return reconstruction.new_zeros(())

    squared_error = (reconstruction - patches).pow(2)
    return squared_error.masked_select(selected).mean()


def run_loss_mode_self_tests() -> None:
    patches = torch.arange(25.0, dtype=torch.float32).view(1, 1, 5, 5)
    reconstruction = patches + 1.0
    visible_mask_all = torch.ones((1, 1), dtype=torch.bool)
    visible_mask_none = torch.zeros((1, 1), dtype=torch.bool)

    loss_all = compute_reconstruction_loss(
        reconstruction,
        patches,
        visible_mask=visible_mask_all,
        vit_patch_size=5,
        loss_mode="ce_reconstruct_all",
    )
    assert torch.isclose(loss_all, torch.tensor(1.0)), loss_all

    loss_visible_all = compute_reconstruction_loss(
        reconstruction,
        patches,
        visible_mask=visible_mask_all,
        vit_patch_size=5,
        loss_mode="ce_reconstruct_visible",
    )
    assert torch.isclose(loss_visible_all, torch.tensor(1.0)), loss_visible_all

    loss_inpaint_zero = compute_reconstruction_loss(
        reconstruction,
        patches,
        visible_mask=visible_mask_all,
        vit_patch_size=5,
        loss_mode="ce_inpaint",
    )
    assert torch.isclose(loss_inpaint_zero, torch.tensor(0.0)), loss_inpaint_zero

    loss_inpaint_all = compute_reconstruction_loss(
        reconstruction,
        patches,
        visible_mask=visible_mask_none,
        vit_patch_size=5,
        loss_mode="ce_inpaint",
    )
    assert torch.isclose(loss_inpaint_all, torch.tensor(1.0)), loss_inpaint_all

    mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)
    pixel_mask = expand_visible_mask_to_pixels(mask, vit_patch_size=5, image_shape=(10, 10))
    assert pixel_mask.shape == (1, 1, 10, 10)
    assert pixel_mask[:, :, :5, :5].all()
    assert not pixel_mask[:, :, :5, 5:].any()
    assert pixel_mask[:, :, 5:, :5].all()
    assert not pixel_mask[:, :, 5:, 5:].any()


def run_masking_self_tests() -> None:
    model = ViTAutoencoder(
        image_size=10,
        patch_size=5,
        in_channels=1,
        embed_dim=8,
        token_embed_dim=8,
        depth=1,
        num_heads=1,
        decoder_channels=(4,),
    )
    with torch.no_grad():
        model.mask_token.fill_(2.0)
        model.pos_embed.copy_(
            torch.arange(model.num_patches * model.embed_dim).view(
                1,
                model.num_patches,
                model.embed_dim,
            )
        )

    torch.manual_seed(0)
    tokens = torch.randn(1, model.num_patches, model.embed_dim)
    masked, visible_mask = model._apply_random_mask(tokens, mask_ratio=0.5)
    masked_positions = (~visible_mask[0]).nonzero(as_tuple=False).flatten()

    assert masked_positions.numel() == 2
    assert visible_mask.sum().item() == 2
    assert torch.allclose(masked[0, masked_positions], model.mask_token[0].expand(2, -1))

    token_inputs = masked + model.pos_embed_scale * model.pos_embed
    masked_token_inputs = token_inputs[0, masked_positions]
    assert not torch.allclose(masked_token_inputs, torch.zeros_like(masked_token_inputs))
    assert not torch.allclose(masked_token_inputs[0], masked_token_inputs[1])


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    vit_patch_size: int,
    mask_ratio: float = 0.0,
    cls_loss_weight: float = 1.0,
    loss_mode: str = "ce_reconstruct_all",
    optimizer: AdamW | None = None,
    stage_name: str = "eval",
    epoch: int | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    total_correct = 0.0
    total_supervised_tokens = 0
    total_items = 0

    grad_context = torch.enable_grad() if training else torch.inference_mode()
    with grad_context:
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            if batch_idx == 1 and epoch is not None:
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] epoch={epoch:03d} stage={stage_name} started",
                    flush=True,
                )
            patches, segments = require_segments(batch)
            patches = patches.to(device, non_blocking=True)
            segments = segments.to(device=device, non_blocking=True)
            aux = model.forward_with_aux(patches, mask_ratio=mask_ratio)
            mse_loss = compute_reconstruction_loss(
                aux.reconstruction,
                patches,
                visible_mask=aux.visible_mask,
                vit_patch_size=vit_patch_size,
                loss_mode=loss_mode,
            )
            token_targets = extract_token_center_targets(segments, vit_patch_size=vit_patch_size)
            ce_loss, cls_acc, supervised_tokens = compute_center_classification_metrics(
                aux.token_logits,
                token_targets,
                aux.visible_mask,
            )
            loss = mse_loss + cls_loss_weight * ce_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = patches.shape[0]
            total_loss += loss.item() * batch_size
            total_mse += mse_loss.item() * batch_size
            total_ce += ce_loss.item() * supervised_tokens
            total_items += batch_size
            total_supervised_tokens += supervised_tokens
            total_correct += cls_acc.item() * supervised_tokens

    mean_items = max(total_items, 1)
    cls_acc_value = total_correct / max(total_supervised_tokens, 1)
    return {
        "total_loss": total_loss / mean_items,
        "mse_loss": total_mse / mean_items,
        "ce_loss": total_ce / max(total_supervised_tokens, 1),
        "cls_acc": cls_acc_value,
        "supervised_tokens": float(total_supervised_tokens),
    }


def build_training_config(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "batches_per_pseudoepoch": args.batches_per_pseudoepoch,
        "epochs": args.epochs,
        "dataset_root": str(args.dataset_root),
        "dataset_size": args.dataset_size,
        "data_dir": str(args.data_dir),
        "train_coords_csv": str(args.train_coords_csv),
        "val_coords_csv": str(args.val_coords_csv),
        "patch_size": args.patch_size,
        "vit_patch_size": args.vit_patch_size,
        "embed_dim": args.embed_dim,
        "token_embed_dim": args.token_embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mask_ratio": args.mask_ratio,
        "cls_loss_weight": args.cls_loss_weight,
        "loss_mode": args.loss_mode,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        # Segmentation head configuration saved into checkpoints/history so runs are easy to compare.
        "segmentation_head": args.segmentation_head,
        "classifier_context_kernel_size": args.classifier_context_kernel_size,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "fold": args.fold,
        "max_folds": args.max_folds,
        "train_keys": list(args.train_keys),
        "test_keys": list(args.test_keys),
        "run_name": args.run_name,
    }


def resolve_run_output_dir(args: argparse.Namespace) -> tuple[str, Path]:
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return run_name, args.output_dir / run_name


def load_wandb_settings(config_path: Path) -> dict[str, object]:
    settings = dict(DEFAULT_WANDB_SETTINGS)
    if not config_path.exists():
        return settings

    with config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    if not isinstance(data, dict):
        raise SystemExit(f"W&B config at {config_path} must be a JSON object.")

    settings.update(data)
    return settings


def init_wandb(args: argparse.Namespace, run_name: str, run_output_dir: Path):
    wandb_settings = load_wandb_settings(args.wandb_config)
    project = args.wandb_project or wandb_settings.get("project")
    entity = args.wandb_entity or wandb_settings.get("entity")
    group = args.wandb_group or wandb_settings.get("group")
    mode = args.wandb_mode or wandb_settings.get("mode", "online")
    tags = args.wandb_tags if args.wandb_tags is not None else wandb_settings.get("tags")

    if project is None or mode == "disabled":
        return None

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb logging was requested, but the 'wandb' package is not installed. "
            "Run 'uv sync' or omit --wandb-project."
        ) from exc

    config = build_training_config(args)
    config.update(
        {
            "dataset_root": str(args.dataset_root),
            "dataset_size": args.dataset_size,
            "data_dir": str(args.data_dir),
            "train_coords_csv": str(args.train_coords_csv),
            "val_coords_csv": str(args.val_coords_csv),
            "cache_root": str(args.cache_root),
            "output_dir": str(args.output_dir),
            "device": args.device,
        }
    )
    return wandb.init(
        entity=entity,
        project=project,
        group=group,
        tags=tags,
        mode=mode,
        name=run_name,
        dir=str(run_output_dir),
        config=config,
    )


def wandb_run_metadata(wandb_run) -> dict[str, str] | None:
    if wandb_run is None:
        return None
    return {
        "entity": str(wandb_run.entity),
        "project": str(wandb_run.project),
        "run_id": str(wandb_run.id),
        "run_name": str(wandb_run.name),
        "run_path": str(wandb_run.path),
        "run_url": str(wandb_run.url),
    }


def save_checkpoint(
    output_dir: Path,
    model: ViTAutoencoder,
    optimizer: AdamW,
    args: argparse.Namespace,
    run_name: str,
    epoch: int,
    val_loss: float,
    is_best: bool,
    wandb_run=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": {
            "image_size": model.image_size,
            "patch_size": model.patch_size,
            "in_channels": model.in_channels,
            "embed_dim": model.embed_dim,
            "token_embed_dim": model.token_embed_dim,
            "num_classes": model.num_classes,
            "segmentation_head": model.segmentation_head,
            "classifier_context_kernel_size": model.classifier_context_kernel_size,
            "classifier_hidden_dim": model.classifier_hidden_dim,
        },
        "run_name": run_name,
        "run_output_dir": str(output_dir.resolve()),
        "training_config": build_training_config(args),
        "wandb": wandb_run_metadata(wandb_run),
    }
    torch.save(state, output_dir / "last.pt")
    if is_best:
        torch.save(state, output_dir / "best.pt")


def save_embeddings_preview(
    model: ViTAutoencoder,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    split_name: str,
) -> None:
    batch = next(iter(loader))
    patches, _, _, _ = unpack_batch(batch)
    patches = patches.to(device, non_blocking=True)
    embeddings = model.extract_embeddings(patches, pooling="mean").cpu()
    torch.save(embeddings, output_dir / f"{split_name}_embeddings_preview.pt")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_loss_mode_self_tests()
        run_masking_self_tests()
        print("train.py self-tests passed.", flush=True)
        return
    if args.data_dir == TRAIN_DEFAULT_DATA_DIR:
        args.data_dir = args.dataset_root / "datasets" / "betaseg"
    torch.manual_seed(args.seed)
    run_name, run_output_dir = resolve_run_output_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] run_name={run_name} output_dir={run_output_dir}",
        flush=True,
    )
    wandb_run = init_wandb(args, run_name, run_output_dir)
    wandb_metadata = wandb_run_metadata(wandb_run)
    if wandb_metadata is not None:
        with open(run_output_dir / "wandb_run.json", "w", encoding="utf-8") as fp:
            json.dump(wandb_metadata, fp, indent=2)

    train_loader, val_loader, data_stats = build_loaders(args)

    in_channels = infer_in_channels(train_loader)
    device = torch.device(args.device)

    # Segmentation head selection:
    # - linear -> original per-token classifier
    # - neighbor_concat -> ordered local neighborhood embeddings flattened together
    model = ViTAutoencoder(
        image_size=args.patch_size,
        patch_size=args.vit_patch_size,
        in_channels=in_channels,
        embed_dim=args.embed_dim,
        token_embed_dim=args.token_embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        segmentation_head=args.segmentation_head,
        classifier_context_kernel_size=args.classifier_context_kernel_size,
        classifier_hidden_dim=args.classifier_hidden_dim,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history: list[dict[str, float]] = []
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] train_loader_ready "
        f"train_mean={data_stats[0]:.6f} train_std={data_stats[1]:.6f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_timestamp = datetime.now(timezone.utc).isoformat()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            vit_patch_size=args.vit_patch_size,
            mask_ratio=args.mask_ratio,
            cls_loss_weight=args.cls_loss_weight,
            loss_mode=args.loss_mode,
            optimizer=optimizer,
            stage_name="train",
            epoch=epoch,
            max_batches=args.batches_per_pseudoepoch,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            vit_patch_size=args.vit_patch_size,
            mask_ratio=args.mask_ratio,
            cls_loss_weight=args.cls_loss_weight,
            loss_mode=args.loss_mode,
            stage_name="val",
            epoch=epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "timestamp_utc": epoch_timestamp,
                "train_loss": train_metrics["total_loss"],
                "val_loss": val_metrics["total_loss"],
                "train_mse": train_metrics["mse_loss"],
                "val_mse": val_metrics["mse_loss"],
                "train_ce": train_metrics["ce_loss"],
                "val_ce": val_metrics["ce_loss"],
                "train_cls_acc": train_metrics["cls_acc"],
                "val_cls_acc": val_metrics["cls_acc"],
            }
        )
        is_best = val_metrics["total_loss"] < best_val
        if is_best:
            best_val = val_metrics["total_loss"]
        if is_best or epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(
                run_output_dir,
                model,
                optimizer,
                args,
                run_name,
                epoch,
                val_metrics["total_loss"],
                is_best,
                wandb_run=wandb_run,
            )

        print(
            f"[{epoch_timestamp}] epoch={epoch:03d} "
            f"train_total={train_metrics['total_loss']:.6f} train_mse={train_metrics['mse_loss']:.6f} "
            f"train_ce={train_metrics['ce_loss']:.6f} train_acc={train_metrics['cls_acc']:.4f} "
            f"val_total={val_metrics['total_loss']:.6f} val_mse={val_metrics['mse_loss']:.6f} "
            f"val_ce={val_metrics['ce_loss']:.6f} val_acc={val_metrics['cls_acc']:.4f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/total": train_metrics["total_loss"],
                    "train/mse": train_metrics["mse_loss"],
                    "train/ce": train_metrics["ce_loss"],
                    "train/acc": train_metrics["cls_acc"],
                    "val/total": val_metrics["total_loss"],
                    "val/mse": val_metrics["mse_loss"],
                    "val/ce": val_metrics["ce_loss"],
                    "val/acc": val_metrics["cls_acc"],
                },
                step=epoch,
            )

    save_embeddings_preview(model, val_loader, device, run_output_dir, split_name="val")
    with open(run_output_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    plot_training_history(history, run_output_dir)
    if wandb_run is not None:
        wandb_run.summary["best_val_total"] = best_val
        wandb_run.finish()


if __name__ == "__main__":
    main()
