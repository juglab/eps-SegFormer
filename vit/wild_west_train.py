import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train as base_train
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
from models_vit import ViTAutoencoder, ViTAutoencoderAuxOutput, ViTEncoderOutput
from models_muvit_v2 import WildWestMuViTV2, token_centers_from_bbox
from plotting.training import plot_training_history
from reproducibility import configure_reproducibility
from wild_west_dataloader import (
    WildWestCoordDataset,
    WildWestMultiResDataset,
    build_level_bboxes,
    build_pretrain_loader,
    build_train_val_loaders,
    build_train_val_multires_loaders,
    run_balanced_batch_sampler_self_tests,
    sample_bbox_from_slice,
)
from vit.label_utils import NUM_CLASSES

ALLOWED_UNLABELED_MIX_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)


class WildWestViTAutoencoder(ViTAutoencoder):
    def _apply_random_mask(
        self,
        tokens: torch.Tensor,
        mask_ratio: float,
        protect_center: torch.Tensor | None = None,
        force_mask_center: torch.Tensor | None = None,
        random_mask_exclusion_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in the range [0.0, 1.0).")

        batch_size, num_tokens, _ = tokens.shape
        center_index = (self.grid_size // 2) * self.grid_size + (self.grid_size // 2)
        visible_mask = torch.ones(
            batch_size,
            num_tokens,
            device=tokens.device,
            dtype=torch.bool,
        )
        masked = tokens.clone()

        if protect_center is not None:
            if protect_center.ndim != 1 or protect_center.shape[0] != batch_size:
                raise ValueError(
                    f"Expected protect_center with shape [{batch_size}], got {tuple(protect_center.shape)}."
                )
            protect_center = protect_center.to(device=tokens.device, dtype=torch.bool)

        if force_mask_center is not None:
            if force_mask_center.ndim != 1 or force_mask_center.shape[0] != batch_size:
                raise ValueError(
                    f"Expected force_mask_center with shape [{batch_size}], got {tuple(force_mask_center.shape)}."
                )
            force_mask_center = force_mask_center.to(device=tokens.device, dtype=torch.bool)
            masked[force_mask_center, center_index] = self.mask_token[0, 0]
            visible_mask[force_mask_center, center_index] = False

        if random_mask_exclusion_mask is not None:
            if random_mask_exclusion_mask.shape != (batch_size, num_tokens):
                raise ValueError(
                    "Expected random_mask_exclusion_mask with shape "
                    f"[{batch_size}, {num_tokens}], got {tuple(random_mask_exclusion_mask.shape)}."
                )
            random_mask_exclusion_mask = random_mask_exclusion_mask.to(device=tokens.device, dtype=torch.bool)

        num_tokens = tokens.shape[1]
        num_masked = int(num_tokens * mask_ratio)
        if num_masked <= 0:
            return masked, visible_mask

        noise = torch.rand(batch_size, num_tokens, device=tokens.device)
        unavailable = ~visible_mask
        if protect_center is not None:
            unavailable[:, center_index] |= protect_center
        if random_mask_exclusion_mask is not None:
            unavailable |= random_mask_exclusion_mask
        noise[unavailable] = float("inf")
        mask_indices = noise.argsort(dim=1)

        for batch_index in range(batch_size):
            available_count = int((~unavailable[batch_index]).sum().item())
            random_mask_count = min(num_masked, available_count)
            if random_mask_count <= 0:
                continue
            selected = mask_indices[batch_index, :random_mask_count]
            masked[batch_index, selected] = self.mask_token[0, 0]
            visible_mask[batch_index, selected] = False
        return masked, visible_mask

    def encode(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
        protect_center: torch.Tensor | None = None,
        force_mask_center: torch.Tensor | None = None,
        random_mask_exclusion_mask: torch.Tensor | None = None,
    ) -> ViTEncoderOutput:
        tokens = self.patch_embed(x)
        tokens = tokens.flatten(2).transpose(1, 2)
        tokens, visible_mask = self._apply_random_mask(
            tokens,
            mask_ratio=mask_ratio,
            protect_center=protect_center,
            force_mask_center=force_mask_center,
            random_mask_exclusion_mask=random_mask_exclusion_mask,
        )
        tokens = self._encode_masked_tokens(tokens, visible_mask)
        feature_map = tokens.transpose(1, 2).reshape(
            x.shape[0], self.embed_dim, self.grid_size, self.grid_size
        )
        pooled = tokens.mean(dim=1)
        return ViTEncoderOutput(
            tokens=tokens,
            feature_map=feature_map,
            pooled=pooled,
            visible_mask=visible_mask,
        )

    def forward_with_aux(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
        protect_center: torch.Tensor | None = None,
        force_mask_center: torch.Tensor | None = None,
        random_mask_exclusion_mask: torch.Tensor | None = None,
    ) -> ViTAutoencoderAuxOutput:
        encoded = self.encode(
            x,
            mask_ratio=mask_ratio,
            protect_center=protect_center,
            force_mask_center=force_mask_center,
            random_mask_exclusion_mask=random_mask_exclusion_mask,
        )
        reconstruction = self.decoder(encoded.feature_map)
        token_logits = self.classify_tokens(encoded.tokens)
        return ViTAutoencoderAuxOutput(
            reconstruction=reconstruction,
            token_logits=token_logits,
            visible_mask=encoded.visible_mask,
            encoded=encoded,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ViT autoencoder on mixed labeled/unlabeled BetaSeg2D patches.")
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
    parser.add_argument(
        "--training-seed",
        type=int,
        default=None,
        help=(
            "Optional seed for model initialization and training-time PyTorch randomness. "
            "When omitted, --seed is used for both data and training randomness."
        ),
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed Python/NumPy/PyTorch and prefer deterministic PyTorch/cuDNN algorithms.",
    )
    parser.add_argument(
        "--model-style",
        choices=("vit", "muvit"),
        default="vit",
        help="Training/model path to use. Default keeps the existing single-resolution Wild West ViT behavior.",
    )
    parser.add_argument("--patch-size", type=int, default=25, help="BetaSeg2D patch size returned by the dataloader.")
    parser.add_argument("--vit-patch-size", type=int, default=5, help="Patch size used inside the ViT encoder.")
    parser.add_argument(
        "--resolution-scales",
        nargs="+",
        type=float,
        default=[1.0, 2.0],
        help="MuViT mode crop scales relative to patch_size.",
    )
    parser.add_argument("--rope-base", type=float, default=10000.0, help="MuViT mode RoPE frequency base.")
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
    parser.add_argument(
        "--mask-end-ratio",
        type=float,
        default=None,
        help="If set, enables mask ratio decay schedule. Floor value the ratio decays toward.",
    )
    parser.add_argument(
        "--mask-decay-epochs",
        type=int,
        default=5,
        help="Decrease mask ratio every N epochs when decay schedule is active.",
    )
    parser.add_argument(
        "--mask-decay-step",
        type=float,
        default=0.15,
        help="Amount to subtract from mask ratio per decay step.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=None,
        help="Dirichlet concentration parameter for per-level mask ratio sampling (MuViT only). "
             "None = uniform masking across levels. Lower values = more variance (paper uses 0.5).",
    )
    parser.add_argument(
        "--masking-mode",
        choices=("token", "drop"),
        default="token",
        help="How masked patches are handled: token keeps learned mask tokens in the encoder, drop excludes masked patches from the encoder.",
    )
    parser.add_argument(
        "--mask-selection-mode",
        choices=("any", "unused_by_head"),
        default="any",
        help="Which patches random masking may select in Wild West training.",
    )
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=0,
        help="Epochs of reconstruction-only pretraining on random patches before full training. 0 = disabled.",
    )
    parser.add_argument(
        "--pretrain-samples-per-epoch",
        type=int,
        default=None,
        help="Number of random patches per pretrain epoch. Defaults to len(train_dataset).",
    )
    parser.add_argument(
        "--loss-mode",
        choices=base_train.LOSS_MODES,
        default="ce_reconstruct_all",
        help="Reconstruction term combined with visible-token cross-entropy.",
    )
    parser.add_argument(
        "--normalize-patches",
        action="store_true",
        default=False,
        help="Normalize each patch token to zero mean and unit variance before MSE loss (MAE-style).",
    )
    parser.add_argument(
        "--fft-loss-weight",
        type=float,
        default=0.0,
        help="Weight for FFT frequency-domain L1 loss added to MSE. 0.0 = disabled. MuViT uses 0.01.",
    )
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--segmentation-head",
        choices=("linear", "neighbor_concat", "neighbor_concat_v2"),
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
    parser.add_argument(
        "--unlabeled-mix-ratio",
        type=float,
        default=0.0,
        help="Reconstruction-only sample count relative to labeled CSV sample count.",
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
    args.auto_run_name = args.run_name is None and args.legacy_run_name is None
    if args.run_name is None:
        args.run_name = args.legacy_run_name
    if args.unlabeled_mix_ratio not in ALLOWED_UNLABELED_MIX_RATIOS:
        raise SystemExit(
            "--unlabeled-mix-ratio must be one of "
            f"{', '.join(str(value) for value in ALLOWED_UNLABELED_MIX_RATIOS)}."
        )
    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        raise SystemExit("--patch-size must be a positive odd integer.")
    if args.vit_patch_size <= 0:
        raise SystemExit("--vit-patch-size must be positive.")
    if args.patch_size % args.vit_patch_size != 0:
        raise SystemExit("--patch-size must be divisible by --vit-patch-size.")
    if args.model_style == "muvit" and any(scale <= 0 for scale in args.resolution_scales):
        raise SystemExit("--resolution-scales must contain only positive values.")
    if args.segmentation_head == "neighbor_concat_v2" and args.classifier_context_kernel_size < 3:
        raise SystemExit("--classifier-context-kernel-size must be at least 3 for neighbor_concat_v2.")
    return args


def build_loaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, tuple[float, float], DataLoader | None]:
    if args.model_style == "muvit":
        train_loader, val_loader, data_stats = build_train_val_multires_loaders(
            dataset_root=args.dataset_root,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            resolution_scales=args.resolution_scales,
            num_workers=args.num_workers,
            train_coords_csv=args.train_coords_csv,
            val_coords_csv=args.val_coords_csv,
            unlabeled_mix_ratio=args.unlabeled_mix_ratio,
            seed=args.seed,
        )
    else:
        train_loader, val_loader, data_stats = build_train_val_loaders(
            dataset_root=args.dataset_root,
            dataset_size=args.dataset_size,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            num_workers=args.num_workers,
            train_coords_csv=args.train_coords_csv,
            val_coords_csv=args.val_coords_csv,
            unlabeled_mix_ratio=args.unlabeled_mix_ratio,
            seed=args.seed,
        )

    pretrain_loader = None
    if args.pretrain_epochs > 0:
        mean, std = data_stats
        names = sorted({s.name for s in train_loader.dataset.labeled_samples})
        num_samples = args.pretrain_samples_per_epoch or len(train_loader.dataset)
        pretrain_loader = build_pretrain_loader(
            dataset_root=args.dataset_root,
            names=names,
            patch_size=args.patch_size,
            normalize_mean=mean,
            normalize_std=std,
            num_samples=num_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
            resolution_scales=args.resolution_scales if args.model_style == "muvit" else None,
        )

    return train_loader, val_loader, data_stats, pretrain_loader


def infer_wild_west_in_channels(loader: DataLoader) -> int:
    sample_batch = next(iter(loader))
    if isinstance(sample_batch, dict):
        return int(sample_batch["img"].shape[2])
    return base_train.infer_in_channels([sample_batch])


def compute_center_patch_classification_metrics(
    token_logits: torch.Tensor,
    center_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if token_logits.ndim != 3:
        raise ValueError(f"Expected token_logits with shape [B, N, C], got {tuple(token_logits.shape)}.")
    if center_labels.ndim != 1:
        raise ValueError(f"Expected center_labels with shape [B], got {tuple(center_labels.shape)}.")

    token_count = token_logits.shape[1]
    grid_size = int(token_count ** 0.5)
    if grid_size * grid_size != token_count:
        raise ValueError(f"Expected a square token grid, got token_count={token_count}.")

    center_index = (grid_size // 2) * grid_size + (grid_size // 2)
    center_logits = token_logits[:, center_index, :]
    valid = (center_labels >= 0) & (center_labels < center_logits.shape[-1])
    if not valid.any():
        zero = center_logits.new_zeros(())
        return zero, zero, 0

    supervised_logits = center_logits[valid]
    supervised_targets = center_labels[valid]
    ce_loss = F.cross_entropy(supervised_logits, supervised_targets)
    predictions = supervised_logits.argmax(dim=-1)
    accuracy = (predictions == supervised_targets).float().mean()
    return ce_loss, accuracy, int(supervised_targets.numel())


def compute_center_logits_classification_metrics(
    center_logits: torch.Tensor,
    center_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if center_logits.ndim != 2:
        raise ValueError(f"Expected center_logits with shape [B, C], got {tuple(center_logits.shape)}.")
    if center_labels.ndim != 1:
        raise ValueError(f"Expected center_labels with shape [B], got {tuple(center_labels.shape)}.")
    valid = (center_labels >= 0) & (center_labels < center_logits.shape[-1])
    if not valid.any():
        zero = center_logits.new_zeros(())
        return zero, zero, 0
    supervised_logits = center_logits[valid]
    supervised_targets = center_labels[valid]
    ce_loss = F.cross_entropy(supervised_logits, supervised_targets)
    accuracy = (supervised_logits.argmax(dim=-1) == supervised_targets).float().mean()
    return ce_loss, accuracy, int(supervised_targets.numel())


def center_head_token_mask(
    grid_size: int,
    segmentation_head: str,
    kernel_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(grid_size * grid_size, device=device, dtype=torch.bool)
    center_row = grid_size // 2
    center_col = grid_size // 2
    center_index = center_row * grid_size + center_col

    if segmentation_head == "linear":
        mask[center_index] = True
        return mask

    if segmentation_head not in ("neighbor_concat", "neighbor_concat_v2"):
        raise ValueError(f"Unsupported segmentation_head '{segmentation_head}'.")

    radius = kernel_size // 2
    for row in range(center_row - radius, center_row + radius + 1):
        for col in range(center_col - radius, center_col + radius + 1):
            if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
                continue
            index = row * grid_size + col
            if segmentation_head == "neighbor_concat_v2" and index == center_index:
                continue
            mask[index] = True
    return mask


def build_random_mask_exclusion_mask(
    model,
    labeled_mask: torch.Tensor | None,
    mask_selection_mode: str,
) -> torch.Tensor | None:
    if mask_selection_mode == "any" or labeled_mask is None:
        return None
    if mask_selection_mode != "unused_by_head":
        raise ValueError("mask_selection_mode must be one of 'any' or 'unused_by_head'.")

    head_mask = center_head_token_mask(
        model.grid_size,
        model.segmentation_head,
        model.classifier_context_kernel_size,
        device=labeled_mask.device,
    )
    if getattr(model, "num_levels", 1) > 1:
        full_head_mask = torch.zeros(
            model.num_levels * model.num_patches,
            device=labeled_mask.device,
            dtype=torch.bool,
        )
        full_head_mask[: model.num_patches] = head_mask
        head_mask = full_head_mask
    exclusion_mask = head_mask.unsqueeze(0).expand(labeled_mask.shape[0], -1).clone()
    exclusion_mask &= labeled_mask.to(device=labeled_mask.device, dtype=torch.bool).unsqueeze(1)
    return exclusion_mask


def run_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    vit_patch_size: int,
    mask_ratio: float = 0.0,
    mask_selection_mode: str = "any",
    cls_loss_weight: float = 1.0,
    loss_mode: str = "ce_reconstruct_all",
    normalize_patches: bool = False,
    fft_loss_weight: float = 0.0,
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
    total_supervised_samples = 0
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

            if isinstance(batch, dict):
                imgs = batch["img"].float().to(device, non_blocking=True)
                bbox = batch["bbox"].to(device=device, non_blocking=True)
                center_labels = batch["center_label"].to(device=device, non_blocking=True)
                is_labeled = batch["is_labeled"].to(device=device, non_blocking=True)
                protect_center = None
                force_mask_center = None
                random_mask_exclusion_mask = build_random_mask_exclusion_mask(
                    model,
                    is_labeled,
                    mask_selection_mode,
                )
                if model.segmentation_head == "neighbor_concat_v2":
                    force_mask_center = is_labeled
                else:
                    protect_center = is_labeled

                aux = model.forward_with_aux(
                    imgs,
                    bbox=bbox,
                    mask_ratio=mask_ratio,
                    protect_center=protect_center,
                    force_mask_center=force_mask_center,
                    random_mask_exclusion_mask=random_mask_exclusion_mask,
                )
                batch_size, _num_levels, _channels, _height, _width = imgs.shape
                mse_loss = base_train.compute_reconstruction_loss(
                    aux.reconstruction,
                    imgs[:, 0],
                    visible_mask=aux.visible_mask[:, : model.num_patches],
                    vit_patch_size=vit_patch_size,
                    loss_mode=loss_mode,
                    normalize_patches=normalize_patches,
                    fft_loss_weight=fft_loss_weight,
                )
                ce_loss, cls_acc, supervised_samples = compute_center_patch_classification_metrics(
                    aux.token_logits,
                    center_labels,
                )
                loss = mse_loss + cls_loss_weight * ce_loss

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item() * batch_size
                total_mse += mse_loss.item() * batch_size
                total_ce += ce_loss.item() * supervised_samples
                total_items += batch_size
                total_supervised_samples += supervised_samples
                total_correct += cls_acc.item() * supervised_samples
                continue

            patches = batch[0].float()
            center_labels = batch[1]
            is_labeled = batch[4] if len(batch) > 4 else None
            patches = patches.to(device, non_blocking=True)
            center_labels = center_labels.to(device=device, non_blocking=True)
            labeled_mask = is_labeled.to(device=device, non_blocking=True) if is_labeled is not None else None
            protect_center = None
            force_mask_center = None
            random_mask_exclusion_mask = build_random_mask_exclusion_mask(
                model,
                labeled_mask,
                mask_selection_mode,
            )
            if labeled_mask is not None:
                if model.segmentation_head == "neighbor_concat_v2":
                    force_mask_center = labeled_mask
                else:
                    protect_center = labeled_mask

            aux = model.forward_with_aux(
                patches,
                mask_ratio=mask_ratio,
                protect_center=protect_center,
                force_mask_center=force_mask_center,
                random_mask_exclusion_mask=random_mask_exclusion_mask,
            )
            mse_loss = base_train.compute_reconstruction_loss(
                aux.reconstruction,
                patches,
                visible_mask=aux.visible_mask,
                vit_patch_size=vit_patch_size,
                loss_mode=loss_mode,
                normalize_patches=normalize_patches,
                fft_loss_weight=fft_loss_weight,
            )
            ce_loss, cls_acc, supervised_samples = compute_center_patch_classification_metrics(
                aux.token_logits,
                center_labels,
            )
            loss = mse_loss + cls_loss_weight * ce_loss

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = patches.shape[0]
            total_loss += loss.item() * batch_size
            total_mse += mse_loss.item() * batch_size
            total_ce += ce_loss.item() * supervised_samples
            total_items += batch_size
            total_supervised_samples += supervised_samples
            total_correct += cls_acc.item() * supervised_samples

    mean_items = max(total_items, 1)
    cls_acc_value = total_correct / max(total_supervised_samples, 1)
    return {
        "total_loss": total_loss / mean_items,
        "mse_loss": total_mse / mean_items,
        "ce_loss": total_ce / max(total_supervised_samples, 1),
        "cls_acc": cls_acc_value,
        "supervised_tokens": float(total_supervised_samples),
    }


def extract_dataset_composition(loader: DataLoader) -> dict[str, object]:
    dataset = loader.dataset
    if not isinstance(dataset, (WildWestCoordDataset, WildWestMultiResDataset)):
        return {}
    composition = {
        "split": dataset.split,
        "sample_counts": dict(dataset.sample_counts),
        "labeled_class_counts": dict(dataset.class_counts),
        "unlabeled_class_counts": dict(dataset.unlabeled_class_counts),
        "unlabeled_mix_ratio": dataset.unlabeled_mix_ratio if dataset.include_unlabeled else 0.0,
        "batch_sampler": dataset.batch_sampler_metadata,
    }
    if isinstance(dataset, WildWestMultiResDataset):
        composition["resolution_scales"] = list(dataset.resolution_scales)
        composition["max_half_extent"] = dataset.max_half_extent
    return composition


def run_wild_west_masking_self_tests() -> None:
    model = WildWestViTAutoencoder(
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
        model.mask_token.fill_(3.0)
        model.pos_embed.copy_(
            torch.arange(model.num_patches * model.embed_dim).view(
                1,
                model.num_patches,
                model.embed_dim,
            )
        )

    tokens = torch.randn(2, model.num_patches, model.embed_dim)
    force_mask_center = torch.tensor([True, False])
    masked, visible_mask = model._apply_random_mask(
        tokens,
        mask_ratio=0.0,
        force_mask_center=force_mask_center,
    )
    center_index = (model.grid_size // 2) * model.grid_size + (model.grid_size // 2)

    assert not visible_mask[0, center_index]
    assert visible_mask[1, center_index]
    assert torch.allclose(masked[0, center_index], model.mask_token[0, 0])

    token_inputs = masked + model.pos_embed_scale * model.pos_embed
    assert not torch.allclose(token_inputs[0, center_index], torch.zeros_like(token_inputs[0, center_index]))

    drop_tokens = torch.arange(
        2 * model.num_patches * model.embed_dim,
        dtype=torch.float32,
    ).view(2, model.num_patches, model.embed_dim)
    drop_masked, drop_visible_mask = model._apply_random_mask(
        drop_tokens,
        mask_ratio=0.0,
        force_mask_center=force_mask_center,
    )
    captured_encoder_inputs: list[torch.Tensor] = []

    def capture_encoder_input(_, inputs):
        captured_encoder_inputs.append(inputs[0].detach().clone())

    handle = model.encoder.register_forward_pre_hook(capture_encoder_input)
    encoded = model._encode_masked_tokens(drop_masked, drop_visible_mask, masking_mode="drop")
    handle.remove()

    center_input = drop_masked[0, center_index] + model.pos_embed_scale * model.pos_embed[0, center_index]
    first_sample_input = captured_encoder_inputs[-1][0, : int(drop_visible_mask[0].sum().item())]
    assert not torch.isclose(first_sample_input, center_input).all(dim=1).any()
    assert torch.allclose(encoded[0, center_index], model.mask_token[0, 0])

    head_mask = center_head_token_mask(
        model.grid_size,
        "neighbor_concat",
        3,
        device=tokens.device,
    )
    labeled_mask = torch.tensor([True, False], device=tokens.device)
    exclusion_mask = head_mask.unsqueeze(0).expand(2, -1).clone()
    exclusion_mask &= labeled_mask.unsqueeze(1)
    torch.manual_seed(0)
    _, restricted_visible = model._apply_random_mask(
        tokens,
        mask_ratio=0.75,
        random_mask_exclusion_mask=exclusion_mask,
    )
    assert restricted_visible[0, head_mask].all()
    assert not restricted_visible[1, head_mask].all()

    v2_head_mask = center_head_token_mask(
        model.grid_size,
        "neighbor_concat_v2",
        3,
        device=tokens.device,
    )
    v2_exclusion_mask = v2_head_mask.unsqueeze(0).expand(2, -1).clone()
    v2_exclusion_mask &= labeled_mask.unsqueeze(1)
    torch.manual_seed(1)
    _, v2_visible = model._apply_random_mask(
        tokens,
        mask_ratio=0.75,
        force_mask_center=labeled_mask,
        random_mask_exclusion_mask=v2_exclusion_mask,
    )
    assert not v2_visible[0, center_index]
    assert v2_visible[0, v2_head_mask].all()
    assert not v2_visible[1, v2_head_mask].all()


def run_muvit_self_tests() -> None:
    bbox = build_level_bboxes(50, 80, patch_size=25, resolution_scales=(1.0, 2.0))
    assert bbox.shape == (2, 2, 2)
    centers = bbox.mean(dim=1)
    assert torch.allclose(centers[0], centers[1])
    extents = bbox[:, 1] - bbox[:, 0]
    assert torch.allclose(extents[:, 0], torch.tensor([25.0, 50.0]))
    assert torch.allclose(extents[:, 1], torch.tensor([25.0, 50.0]))

    bbox_27 = build_level_bboxes(50, 80, patch_size=27, resolution_scales=(1.0, 2.0))
    extents_27 = bbox_27[:, 1] - bbox_27[:, 0]
    assert torch.allclose(extents_27[:, 0], torch.tensor([27.0, 54.0]))

    image = torch.arange(10000.0).view(100, 100)
    fine_sample = sample_bbox_from_slice(image, bbox[0], out_size=25)
    coarse_sample = sample_bbox_from_slice(image, bbox[1], out_size=25)
    assert fine_sample.shape == (1, 25, 25)
    assert coarse_sample.shape == (1, 25, 25)

    batched_bbox = bbox.view(1, 2, 2, 2)
    coords = token_centers_from_bbox(batched_bbox, grid_h=5, grid_w=5)
    assert coords.shape == (1, 2, 25, 2)
    center_index = 2 * 5 + 2
    assert torch.allclose(coords[0, 0, center_index], coords[0, 1, center_index])

    coords_27 = token_centers_from_bbox(bbox_27.view(1, 2, 2, 2), grid_h=9, grid_w=9)
    assert coords_27.shape == (1, 2, 81, 2)
    assert torch.allclose(coords_27[0, 0, 4 * 9 + 4], coords_27[0, 1, 4 * 9 + 4])

    model = WildWestMuViTV2(
        image_size=25,
        patch_size=5,
        in_channels=1,
        embed_dim=16,
        depth=1,
        num_heads=1,
        mlp_ratio=2.0,
        decoder_channels=(8,),
        num_classes=NUM_CLASSES,
        num_levels=2,
    )
    imgs = torch.randn(2, 2, 1, 25, 25)
    bbox_batch = bbox.view(1, 2, 2, 2).repeat(2, 1, 1, 1)
    protect_center = torch.tensor([True, False])
    torch.manual_seed(0)
    aux = model.forward_with_aux(
        imgs,
        bbox=bbox_batch,
        mask_ratio=0.75,
        protect_center=protect_center,
    )
    assert aux.reconstruction.shape == (2, 1, 25, 25)
    assert aux.token_logits.shape == (2, 25, NUM_CLASSES)
    assert aux.center_logits.shape == (2, NUM_CLASSES)
    assert aux.visible_mask.shape == (2, 50)
    assert aux.finest_tokens.shape == (2, 25, 16)
    assert torch.equal(aux.visible_mask.view(2, 2, 25).sum(dim=2), torch.full((2, 2), 7))
    assert aux.visible_mask[0, center_index]

    center_labels = torch.tensor([1, -1])
    ce_loss, _, supervised_samples = compute_center_patch_classification_metrics(aux.token_logits, center_labels)
    assert supervised_samples == 1
    recon_loss = base_train.compute_reconstruction_loss(
        aux.reconstruction,
        imgs[:, 0],
        visible_mask=aux.visible_mask[:, : model.num_patches],
        vit_patch_size=5,
        loss_mode="ce_reconstruct_all",
    )
    assert recon_loss.ndim == 0

    optimizer = AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    (recon_loss + ce_loss).backward()
    optimizer.step()

    flex_model = WildWestMuViTV2(
        image_size=27,
        patch_size=3,
        in_channels=1,
        embed_dim=16,
        depth=1,
        num_heads=1,
        mlp_ratio=2.0,
        decoder_channels=(8,),
        num_classes=NUM_CLASSES,
        num_levels=2,
    )
    flex_imgs = torch.randn(1, 2, 1, 27, 27)
    flex_aux = flex_model.forward_with_aux(flex_imgs, bbox=bbox_27.view(1, 2, 2, 2), mask_ratio=0.0)
    assert flex_aux.reconstruction.shape == (1, 1, 27, 27)
    assert flex_aux.finest_tokens.shape[1] == 81

    neighbor_model = WildWestMuViTV2(
        image_size=25,
        patch_size=5,
        in_channels=1,
        embed_dim=16,
        depth=1,
        num_heads=1,
        mlp_ratio=2.0,
        decoder_channels=(8,),
        num_classes=NUM_CLASSES,
        num_levels=2,
        segmentation_head="neighbor_concat",
        classifier_context_kernel_size=3,
    )
    neighbor_aux = neighbor_model.forward_with_aux(imgs, bbox=bbox_batch, mask_ratio=0.0)
    assert neighbor_aux.token_logits.shape == (2, 25, NUM_CLASSES)

    v2_model = WildWestMuViTV2(
        image_size=25,
        patch_size=5,
        in_channels=1,
        embed_dim=16,
        depth=1,
        num_heads=1,
        mlp_ratio=2.0,
        decoder_channels=(8,),
        num_classes=NUM_CLASSES,
        num_levels=2,
        segmentation_head="neighbor_concat_v2",
        classifier_context_kernel_size=3,
    )
    labeled_mask = torch.tensor([True, False])
    head_exclusion = build_random_mask_exclusion_mask(v2_model, labeled_mask, "unused_by_head")
    assert head_exclusion.shape == (2, 50)
    assert not head_exclusion[0, center_index]
    assert head_exclusion[0, :25].any()
    assert not head_exclusion[0, 25:].any()
    torch.manual_seed(1)
    v2_aux = v2_model.forward_with_aux(
        imgs,
        bbox=bbox_batch,
        mask_ratio=0.75,
        force_mask_center=labeled_mask,
        random_mask_exclusion_mask=head_exclusion,
    )
    assert v2_aux.token_logits.shape == (2, 25, NUM_CLASSES)
    assert not v2_aux.visible_mask[0, center_index]
    assert v2_aux.visible_mask[0, head_exclusion[0]].all()
    assert v2_aux.visible_mask.view(2, 2, 25)[1].sum(dim=1).tolist() == [7, 7]

    drop_model = WildWestMuViTV2(
        image_size=25,
        patch_size=5,
        in_channels=1,
        embed_dim=16,
        depth=1,
        num_heads=1,
        mlp_ratio=2.0,
        decoder_channels=(8,),
        num_classes=NUM_CLASSES,
        num_levels=2,
        masking_mode="drop",
    )
    drop_aux = drop_model.forward_with_aux(
        imgs,
        bbox=bbox_batch,
        mask_ratio=0.5,
        protect_center=labeled_mask,
    )
    assert drop_aux.reconstruction.shape == (2, 1, 25, 25)
    assert drop_aux.token_logits.shape == (2, 25, NUM_CLASSES)
    assert torch.equal(drop_aux.visible_mask.view(2, 2, 25).sum(dim=2), torch.full((2, 2), 13))


def build_training_config(
    args: argparse.Namespace,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
) -> dict[str, object]:
    config = base_train.build_training_config(args)
    config["data_seed"] = args.seed
    config["training_seed"] = args.training_seed if args.training_seed is not None else args.seed
    config["model_style"] = args.model_style
    config["resolution_scales"] = list(args.resolution_scales)
    config["rope_base"] = args.rope_base
    config["unlabeled_mix_ratio"] = args.unlabeled_mix_ratio
    config["mask_selection_mode"] = args.mask_selection_mode
    if train_loader is not None:
        config["train_dataset_composition"] = extract_dataset_composition(train_loader)
    if val_loader is not None:
        config["val_dataset_composition"] = extract_dataset_composition(val_loader)
    return config


def init_wandb(
    args: argparse.Namespace,
    run_name: str,
    run_output_dir: Path,
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
):
    wandb_settings = base_train.load_wandb_settings(args.wandb_config)
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

    config = build_training_config(args, train_loader=train_loader, val_loader=val_loader)
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


def save_checkpoint(
    output_dir: Path,
    model: ViTAutoencoder,
    optimizer: AdamW,
    args: argparse.Namespace,
    run_name: str,
    epoch: int,
    val_loss: float,
    is_best: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    wandb_run=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.model_style == "muvit":
        model_config = {
            "model_style": "muvit",
            "image_size": model.image_size,
            "patch_size": model.patch_size,
            "in_channels": model.in_channels,
            "embed_dim": model.embed_dim,
            "num_classes": model.num_classes,
            "num_levels": model.num_levels,
            "resolution_scales": list(args.resolution_scales),
            "rope_base": model.rope_base,
            "masking_mode": model.masking_mode,
            "segmentation_head": model.segmentation_head,
            "classifier_context_kernel_size": model.classifier_context_kernel_size,
            "classifier_hidden_dim": model.classifier_hidden_dim,
        }
    else:
        model_config = {
            "model_style": "vit",
            "image_size": model.image_size,
            "patch_size": model.patch_size,
            "in_channels": model.in_channels,
            "embed_dim": model.embed_dim,
            "token_embed_dim": model.token_embed_dim,
            "num_classes": model.num_classes,
            "masking_mode": model.masking_mode,
            "segmentation_head": model.segmentation_head,
            "classifier_context_kernel_size": model.classifier_context_kernel_size,
            "classifier_hidden_dim": model.classifier_hidden_dim,
        }
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model_config,
        "run_name": run_name,
        "run_output_dir": str(output_dir.resolve()),
        "training_config": build_training_config(args, train_loader=train_loader, val_loader=val_loader),
        "wandb": base_train.wandb_run_metadata(wandb_run),
    }
    torch.save(state, output_dir / "last.pt")
    if is_best:
        torch.save(state, output_dir / "best.pt")


@torch.inference_mode()
def save_wild_west_embeddings_preview(
    model,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    split_name: str,
    model_style: str,
) -> None:
    if model_style == "vit":
        base_train.save_embeddings_preview(model, loader, device, output_dir, split_name=split_name)
        return

    batch = next(iter(loader))
    imgs = batch["img"].float().to(device, non_blocking=True)
    bbox = batch["bbox"].to(device=device, non_blocking=True)
    model.eval()
    aux = model.forward_with_aux(imgs, bbox=bbox, mask_ratio=0.0)
    embeddings = aux.finest_tokens.mean(dim=1).cpu()
    torch.save(embeddings, output_dir / f"{split_name}_embeddings_preview.pt")


def compute_scheduled_mask_ratio(
    epoch: int,
    mask_start: float,
    mask_end: float,
    decay_epochs: int,
    decay_step: float,
) -> float:
    steps_taken = (epoch - 1) // decay_epochs
    ratio = mask_start + steps_taken * decay_step
    if mask_end > mask_start:
        return min(ratio, mask_end)
    return max(ratio, mask_end)


def main() -> None:
    args = parse_args()
    if args.self_test:
        base_train.run_loss_mode_self_tests()
        base_train.run_masking_self_tests()
        run_wild_west_masking_self_tests()
        run_muvit_self_tests()
        run_balanced_batch_sampler_self_tests()
        print("wild_west_train.py self-tests passed.", flush=True)
        return
    if args.data_dir == TRAIN_DEFAULT_DATA_DIR:
        args.data_dir = args.dataset_root / "datasets" / "betaseg"
    if args.loss_mode == "ce_inpaint" and args.mask_ratio == 0.0 and args.unlabeled_mix_ratio > 0.0:
        print(
            "WARNING: ce_inpaint with mask_ratio=0 and unlabeled_mix_ratio>0 gives unlabeled samples "
            "no reconstruction loss and no CE; they only change batch composition and optimization dynamics.",
            flush=True,
        )
    configure_reproducibility(args.seed, deterministic=args.deterministic)
    run_name, run_output_dir = base_train.resolve_run_output_dir(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] run_name={run_name} output_dir={run_output_dir}",
        flush=True,
    )

    train_loader, val_loader, data_stats, pretrain_loader = build_loaders(args)

    train_dataset_composition = extract_dataset_composition(train_loader)
    val_dataset_composition = extract_dataset_composition(val_loader)
    with (run_output_dir / "dataset_composition.json").open("w", encoding="utf-8") as fp:
        json.dump(
            {
                "train": train_dataset_composition,
                "val": val_dataset_composition,
            },
            fp,
            indent=2,
        )

    wandb_run = init_wandb(
        args,
        run_name,
        run_output_dir,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    wandb_metadata = base_train.wandb_run_metadata(wandb_run)
    if wandb_metadata is not None:
        with (run_output_dir / "wandb_run.json").open("w", encoding="utf-8") as fp:
            json.dump(wandb_metadata, fp, indent=2)

    in_channels = infer_wild_west_in_channels(train_loader)
    device = torch.device(args.device)
    training_seed = args.training_seed if args.training_seed is not None else args.seed
    configure_reproducibility(training_seed, deterministic=args.deterministic)
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] data_seed={args.seed} training_seed={training_seed}",
        flush=True,
    )
    if args.model_style == "muvit":
        model = WildWestMuViTV2(
            image_size=args.patch_size,
            patch_size=args.vit_patch_size,
            in_channels=in_channels,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            num_levels=len(args.resolution_scales),
            rope_base=args.rope_base,
            segmentation_head=args.segmentation_head,
            classifier_context_kernel_size=args.classifier_context_kernel_size,
            classifier_hidden_dim=args.classifier_hidden_dim,
            masking_mode=args.masking_mode,
            dirichlet_alpha=args.dirichlet_alpha,
        ).to(device)
    else:
        model = WildWestViTAutoencoder(
            image_size=args.patch_size,
            patch_size=args.vit_patch_size,
            in_channels=in_channels,
            embed_dim=args.embed_dim,
            token_embed_dim=args.token_embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            masking_mode=args.masking_mode,
            segmentation_head=args.segmentation_head,
            classifier_context_kernel_size=args.classifier_context_kernel_size,
            classifier_hidden_dim=args.classifier_hidden_dim,
        ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    history: list[dict[str, float]] = []
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] train_loader_ready "
        f"train_mean={data_stats[0]:.6f} train_std={data_stats[1]:.6f} "
        f"unlabeled_mix_ratio={args.unlabeled_mix_ratio:.2f}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        epoch_timestamp = datetime.now(timezone.utc).isoformat()
        is_pretraining = epoch <= args.pretrain_epochs
        phase = "pretrain" if is_pretraining else "finetune"
        current_loader = pretrain_loader if is_pretraining else train_loader
        current_cls_loss_weight = 0.0 if is_pretraining else args.cls_loss_weight
        if args.mask_end_ratio is not None:
            current_mask_ratio = compute_scheduled_mask_ratio(
                epoch,
                mask_start=args.mask_ratio,
                mask_end=args.mask_end_ratio,
                decay_epochs=args.mask_decay_epochs,
                decay_step=args.mask_decay_step,
            )
        else:
            current_mask_ratio = args.mask_ratio
        train_metrics = run_epoch(
            model,
            current_loader,
            device,
            vit_patch_size=args.vit_patch_size,
            mask_ratio=current_mask_ratio,
            mask_selection_mode=args.mask_selection_mode,
            cls_loss_weight=current_cls_loss_weight,
            loss_mode=args.loss_mode,
            normalize_patches=args.normalize_patches,
            fft_loss_weight=args.fft_loss_weight,
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
            mask_ratio=current_mask_ratio,
            mask_selection_mode=args.mask_selection_mode,
            cls_loss_weight=current_cls_loss_weight,
            loss_mode=args.loss_mode,
            normalize_patches=args.normalize_patches,
            fft_loss_weight=args.fft_loss_weight,
            stage_name="val",
            epoch=epoch,
        )
        history.append(
            {
                "epoch": epoch,
                "phase": phase,
                "timestamp_utc": epoch_timestamp,
                "mask_ratio": current_mask_ratio,
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
        if is_pretraining and epoch == args.pretrain_epochs:
            torch.save(model.state_dict(), run_output_dir / "pretrain.pt")
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
                train_loader=train_loader,
                val_loader=val_loader,
                wandb_run=wandb_run,
            )

        print(
            f"[{epoch_timestamp}] epoch={epoch:03d} phase={phase} cls_w={current_cls_loss_weight:.4f} "
            f"mask_ratio={current_mask_ratio:.4f} "
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
                    "phase": phase,
                    "mask_ratio": current_mask_ratio,
                    "cls_loss_weight": current_cls_loss_weight,
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

    save_wild_west_embeddings_preview(
        model,
        val_loader,
        device,
        run_output_dir,
        split_name="val",
        model_style=args.model_style,
    )
    with (run_output_dir / "history.json").open("w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)

    plot_training_history(history, run_output_dir)
    if wandb_run is not None:
        wandb_run.summary["best_val_total"] = best_val
        wandb_run.finish()


if __name__ == "__main__":
    main()
