import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models_vit import ViTAutoencoder
from plotting.training import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ViT autoencoder on BetaSeg2D patches.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the betaseg dataset root.")
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/vit_ae"))
    parser.add_argument("--train-keys", nargs="+", default=["high_c1", "high_c2", "high_c3"])
    parser.add_argument("--test-keys", nargs="+", default=["high_c4"])
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=32, help="BetaSeg2D patch size returned by the dataloader.")
    parser.add_argument("--vit-patch-size", type=int, default=8, help="Patch size used inside the ViT encoder.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches-per-pseudoepoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-every", type=int, default=5)
    return parser.parse_args()


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
        samples_per_class_training={0: 5, 1: 10, 2: 5, 3: 5},
        samples_per_class_validation={0: 5, 1: 10, 2: 5, 3: 5},
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


def unpack_batch(batch) -> torch.Tensor:
    patches = batch[0] if isinstance(batch, (tuple, list)) else batch
    return patches.float()


def infer_in_channels(loader: Iterable) -> int:
    sample_batch = next(iter(loader))
    patches = unpack_batch(sample_batch)
    return int(patches.shape[1])


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        patches = unpack_batch(batch).to(device, non_blocking=True)
        recon = model(patches)
        loss = F.mse_loss(recon, patches)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = patches.shape[0]
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


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
        "depth": args.depth,
        "num_heads": args.num_heads,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "fold": args.fold,
        "max_folds": args.max_folds,
        "train_keys": list(args.train_keys),
        "test_keys": list(args.test_keys),
    }


def save_checkpoint(
    output_dir: Path,
    model: ViTAutoencoder,
    optimizer: AdamW,
    args: argparse.Namespace,
    epoch: int,
    val_loss: float,
    is_best: bool,
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
        },
        "training_config": build_training_config(args),
    }
    torch.save(state, output_dir / "last.pt")
    if is_best:
        torch.save(state, output_dir / "best.pt")


def save_embeddings_preview(
    model: ViTAutoencoder,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    batch = next(iter(loader))
    patches = unpack_batch(batch).to(device, non_blocking=True)
    embeddings = model.extract_embeddings(patches, pooling="mean").cpu()
    torch.save(embeddings, output_dir / "test_embeddings_preview.pt")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    datamodule = build_datamodule(args)
    train_loader = get_loader(datamodule, "train")
    val_loader = get_loader(datamodule, "val")
    test_loader = get_loader(datamodule, "test")

    in_channels = infer_in_channels(train_loader)
    device = torch.device(args.device)
    model = ViTAutoencoder(
        image_size=args.patch_size,
        patch_size=args.vit_patch_size,
        in_channels=in_channels,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss = run_epoch(model, val_loader, device)
        test_loss = run_epoch(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            }
        )
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        if is_best or epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args.output_dir, model, optimizer, args, epoch, val_loss, is_best)

        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f} test_mse={test_loss:.6f}"
        )

    save_embeddings_preview(model, test_loader, device, args.output_dir)
    with open(args.output_dir / "history.json", "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    plot_training_history(history, args.output_dir)


if __name__ == "__main__":
    main()
