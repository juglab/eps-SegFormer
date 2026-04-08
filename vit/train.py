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
    TRAIN_DEFAULT_OUTPUT_DIR,
    TRAIN_DEFAULT_WANDB_CONFIG_PATH,
)
from label_utils import NUM_CLASSES, remap_label_tensor, valid_class_mask
from models_vit import ViTAutoencoder
from plotting.training import plot_training_history

DEFAULT_WANDB_SETTINGS: dict[str, object] = {
    "entity": "juglab",
    "project": "eps_segformer",
    "mode": "online",
    "tags": ["vit", "autoencoder"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ViT autoencoder on BetaSeg2D patches.")
    parser.add_argument("--data-dir", type=Path, default=TRAIN_DEFAULT_DATA_DIR, help="Path to the betaseg dataset root.")
    parser.add_argument("--cache-root", type=Path, default=TRAIN_DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=TRAIN_DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--patch-size", type=int, default=80, help="BetaSeg2D patch size returned by the dataloader.")
    parser.add_argument("--vit-patch-size", type=int, default=5, help="Patch size used inside the ViT encoder.")
    parser.add_argument("--batch-size", type=int, default=32)
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
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
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
    args = parser.parse_args()
    if args.run_name is None:
        args.run_name = args.legacy_run_name
    return args


def build_datamodule(args: argparse.Namespace):
    try:
        from eps_seg.config.datasets import BetaSegDatasetConfig
        from eps_seg.config.train import TrainConfig
        from eps_seg.dataloaders.datamodules import EPSSegDataModule
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "eps_seg is not installed in this environment. Install it first so train.py can load BetaSeg2D."
        ) from exc

    dataset_config = BetaSegDatasetConfig(
        dim=2,
        name="betaseg_2d",
        fold=args.fold,
        max_folds=args.max_folds,
        data_dir=str(args.data_dir),
        cache_dir=str(args.cache_root),
        enable_cache=True,
        train_keys=list(args.train_keys),
        test_keys=list(args.test_keys),
        seed=args.seed,
        patch_size=args.patch_size,
        samples_per_class_training={0: 100, 1: 200, 2: 100, 3: 100},
        samples_per_class_validation={0: 100, 1: 200, 2: 100, 3: 100},
    )
    train_config = TrainConfig(
        model_name="vit_autoencoder",
        batch_size=args.batch_size,
        batches_per_pseudoepoch=args.batches_per_pseudoepoch,
    )
    datamodule = EPSSegDataModule(dataset_config, train_cfg=train_config)
    datamodule.prepare_data()
    datamodule.setup("fit")
    datamodule.setup("test")
    return datamodule


def get_loader(datamodule, stage: str) -> DataLoader:
    if stage == "train":
        return datamodule.train_dataloader()
    if stage == "val":
        if hasattr(datamodule, "val_dataloader"):
            return datamodule.val_dataloader()
        return datamodule.test_dataloader()
    if stage == "test":
        return datamodule.test_dataloader()
    raise ValueError(f"Unknown stage: {stage}")


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


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    vit_patch_size: int,
    mask_ratio: float = 0.0,
    cls_loss_weight: float = 1.0,
    optimizer: AdamW | None = None,
    stage_name: str = "eval",
    epoch: int | None = None,
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
            if batch_idx == 1 and epoch is not None:
                print(
                    f"[{datetime.now(timezone.utc).isoformat()}] epoch={epoch:03d} stage={stage_name} started",
                    flush=True,
                )
            patches, segments = require_segments(batch)
            patches = patches.to(device, non_blocking=True)
            segments = segments.to(device=device, non_blocking=True)
            aux = model.forward_with_aux(patches, mask_ratio=mask_ratio)
            mse_loss = F.mse_loss(aux.reconstruction, patches)
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
        "patch_size": args.patch_size,
        "vit_patch_size": args.vit_patch_size,
        "embed_dim": args.embed_dim,
        "token_embed_dim": args.token_embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mask_ratio": args.mask_ratio,
        "cls_loss_weight": args.cls_loss_weight,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
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
            "data_dir": str(args.data_dir),
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

    datamodule = build_datamodule(args)
    train_loader = get_loader(datamodule, "train")
    val_loader = get_loader(datamodule, "val")

    in_channels = infer_in_channels(train_loader)
    device = torch.device(args.device)
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
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_timestamp = datetime.now(timezone.utc).isoformat()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            vit_patch_size=args.vit_patch_size,
            mask_ratio=args.mask_ratio,
            cls_loss_weight=args.cls_loss_weight,
            optimizer=optimizer,
            stage_name="train",
            epoch=epoch,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            device,
            vit_patch_size=args.vit_patch_size,
            mask_ratio=args.mask_ratio,
            cls_loss_weight=args.cls_loss_weight,
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
