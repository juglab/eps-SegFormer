from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from dataloader import (
    AUTO_FILTER_BORDER_PATCH_SIZES,
    CoordRow,
    _csv_path_for_split,
    _read_slice,
    _resolve_volume_paths,
    _volume_shape,
    compute_source_stats,
    filter_coordinate_rows_for_patch_size,
    load_coordinate_rows,
    validate_coordinate_rows,
)
from reproducibility import make_torch_generator, seed_worker
from vit.label_utils import NUM_CLASSES, UNRECOGNIZED_LABEL


@dataclass(frozen=True)
class WildWestSample:
    name: str
    z: int
    y: int
    x: int
    is_labeled: bool
    hidden_class: int | None


def _group_rows_by_slice(rows: Iterable[CoordRow]) -> dict[tuple[str, int], list[CoordRow]]:
    grouped: dict[tuple[str, int], list[CoordRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.name, row.z)].append(row)
    return grouped


def _round_robin_quotas(
    total: int,
    keys: list[tuple[str, int]] | list[int],
    capacities: dict[tuple[str, int], int] | dict[int, int] | None = None,
) -> dict[tuple[str, int], int] | dict[int, int]:
    quotas = {key: 0 for key in keys}
    if total <= 0 or not keys:
        return quotas

    active = [key for key in keys if capacities is None or capacities.get(key, 0) > 0]
    while total > 0 and active:
        next_active: list[tuple[str, int]] | list[int] = []
        for key in active:
            if total <= 0:
                break
            capacity = None if capacities is None else capacities.get(key, 0)
            if capacity is not None and quotas[key] >= capacity:
                continue
            quotas[key] += 1
            total -= 1
            if capacity is None or quotas[key] < capacity:
                next_active.append(key)
        active = next_active
    return quotas


def build_level_bboxes(
    center_y: float,
    center_x: float,
    patch_size: int,
    resolution_scales: Sequence[float],
) -> torch.Tensor:
    half_extent = patch_size / 2.0
    boxes = []
    for scale in resolution_scales:
        if scale <= 0:
            raise ValueError(f"resolution scales must be positive, got {scale}.")
        extent = half_extent * float(scale)
        boxes.append(
            [
                [float(center_y) - extent, float(center_x) - extent],
                [float(center_y) + extent, float(center_x) + extent],
            ]
        )
    return torch.tensor(boxes, dtype=torch.float32)


def sample_bbox_from_slice(
    image_2d: np.ndarray | torch.Tensor,
    bbox: torch.Tensor,
    out_size: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    if out_size <= 0:
        raise ValueError(f"out_size must be positive, got {out_size}.")
    if mode not in ("bilinear", "nearest"):
        raise ValueError("mode must be one of 'bilinear' or 'nearest'.")

    image = torch.as_tensor(image_2d)
    if image.ndim != 2:
        raise ValueError(f"Expected image_2d with shape [H, W], got {tuple(image.shape)}.")
    image = image.to(dtype=torch.float32)
    bbox = torch.as_tensor(bbox, dtype=torch.float32, device=image.device)
    if bbox.shape != (2, 2):
        raise ValueError(f"Expected bbox with shape [2, 2], got {tuple(bbox.shape)}.")

    height, width = image.shape
    y0, x0 = bbox[0]
    y1, x1 = bbox[1]
    ys = y0 + (torch.arange(out_size, device=image.device, dtype=torch.float32) + 0.5) * ((y1 - y0) / out_size)
    xs = x0 + (torch.arange(out_size, device=image.device, dtype=torch.float32) + 0.5) * ((x1 - x0) / out_size)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = ((yy + 0.5) / height) * 2.0 - 1.0
    grid_x = ((xx + 0.5) / width) * 2.0 - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    sampled = F.grid_sample(
        image.view(1, 1, height, width),
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )
    return sampled.squeeze(0)


def _filter_rows_for_continuous_extent(
    rows: Iterable[CoordRow],
    volume_shapes: dict[str, tuple[int, int, int]],
    half_extent: float,
) -> tuple[list[CoordRow], int]:
    kept_rows: list[CoordRow] = []
    dropped = 0
    for row in rows:
        if row.name not in volume_shapes:
            raise ValueError(f"Unknown dataset '{row.name}' referenced by row {row.row_index}.")
        depth, height, width = volume_shapes[row.name]
        is_valid = (
            0 <= row.z < depth
            and row.y - half_extent >= 0.0
            and row.x - half_extent >= 0.0
            and row.y + half_extent <= float(height)
            and row.x + half_extent <= float(width)
        )
        if is_valid:
            kept_rows.append(row)
        else:
            dropped += 1
    return kept_rows, dropped


class WildWestCoordDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        size: str = "low",
        patch_size: int = 25,
        normalize_mean: float | None = None,
        normalize_std: float | None = None,
        csv_path: Path | None = None,
        unlabeled_mix_ratio: float = 0.0,
        seed: int = 42,
        include_unlabeled: bool = False,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}.")
        if unlabeled_mix_ratio < 0.0:
            raise ValueError(f"unlabeled_mix_ratio must be non-negative, got {unlabeled_mix_ratio}.")

        self.dataset_root = dataset_root
        self.split = split
        self.size = size
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.csv_path = csv_path or _csv_path_for_split(dataset_root, size=size, split=split)
        self.unlabeled_mix_ratio = unlabeled_mix_ratio
        self.seed = seed
        self.include_unlabeled = include_unlabeled and unlabeled_mix_ratio > 0.0

        self.rows = load_coordinate_rows(self.csv_path)
        self._rows_by_slice = _group_rows_by_slice(self.rows)
        self._source_paths: dict[str, Path] = {}
        self._label_paths: dict[str, Path] = {}
        volume_shapes: dict[str, tuple[int, int, int]] = {}
        for name in sorted({row.name for row in self.rows}):
            source_path, label_path = _resolve_volume_paths(dataset_root, name)
            self._source_paths[name] = source_path
            self._label_paths[name] = label_path
            volume_shapes[name] = _volume_shape(source_path)

        if patch_size in AUTO_FILTER_BORDER_PATCH_SIZES:
            filtered_rows, dropped = filter_coordinate_rows_for_patch_size(
                self.rows,
                volume_shapes,
                patch_size=patch_size,
            )
            if not filtered_rows:
                raise ValueError(
                    f"Filtering {self.csv_path} for patch_size={patch_size} removed all rows."
                )
            if dropped > 0:
                print(
                    f"Filtered {dropped} border-invalid rows from {self.csv_path} for patch_size={patch_size}. "
                    f"Keeping {len(filtered_rows)} rows.",
                    flush=True,
                )
            self.rows = filtered_rows
            self._rows_by_slice = _group_rows_by_slice(self.rows)
        else:
            validate_coordinate_rows(self.rows, volume_shapes, patch_size=patch_size, csv_path=self.csv_path)

        self.class_counts = self._compute_center_class_counts(self.rows)
        self._labeled_lookup_by_slice = {
            slice_key: {(row.y, row.x) for row in slice_rows}
            for slice_key, slice_rows in self._rows_by_slice.items()
        }
        self.labeled_samples = [
            WildWestSample(
                name=row.name,
                z=row.z,
                y=row.y,
                x=row.x,
                is_labeled=True,
                hidden_class=None,
            )
            for row in self.rows
        ]
        self.unlabeled_samples: list[WildWestSample] = []
        self.unlabeled_class_counts: dict[int, int] = {}

        if self.include_unlabeled:
            self.unlabeled_samples = self._sample_unlabeled_samples()
            self.unlabeled_class_counts = self._class_histogram_for_samples(self.unlabeled_samples)

        self.samples = [*self.labeled_samples, *self.unlabeled_samples]
        self.sample_counts = {
            "labeled": len(self.labeled_samples),
            "unlabeled": len(self.unlabeled_samples),
            "total": len(self.samples),
        }
        self.batch_sampler_metadata: dict[str, object] | None = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        image_slice = _read_slice(self._source_paths[sample.name], sample.z, np.float32)

        y0 = sample.y - self.half
        y1 = sample.y + self.half + 1
        x0 = sample.x - self.half
        x1 = sample.x + self.half + 1

        image_patch = np.asarray(image_slice[y0:y1, x0:x1], dtype=np.float32)
        if self.normalize_mean is not None and self.normalize_std is not None:
            image_patch = (image_patch - self.normalize_mean) / self.normalize_std

        if sample.is_labeled:
            label_slice = _read_slice(self._label_paths[sample.name], sample.z, np.int64)
            segment_patch = np.asarray(label_slice[y0:y1, x0:x1], dtype=np.int64)
            center_label = int(label_slice[sample.y, sample.x])
        else:
            segment_patch = np.full((self.patch_size, self.patch_size), UNRECOGNIZED_LABEL, dtype=np.int64)
            center_label = UNRECOGNIZED_LABEL

        patch_tensor = torch.from_numpy(image_patch).unsqueeze(0).to(dtype=torch.float32)
        center_tensor = torch.tensor(center_label, dtype=torch.long)
        segment_tensor = torch.from_numpy(segment_patch).unsqueeze(0).to(dtype=torch.long)
        coords_tensor = torch.tensor((sample.z, sample.y, sample.x), dtype=torch.long)
        is_labeled_tensor = torch.tensor(sample.is_labeled, dtype=torch.bool)
        return patch_tensor, center_tensor, segment_tensor, coords_tensor, is_labeled_tensor

    def _compute_center_class_counts(self, rows: Iterable[CoordRow]) -> dict[int, int]:
        counts: Counter[int] = Counter()
        grouped_rows = _group_rows_by_slice(rows)
        for (name, z_index), slice_rows in grouped_rows.items():
            label_slice = _read_slice(self._label_paths[name], z_index, np.int64)
            for row in slice_rows:
                counts[int(label_slice[row.y, row.x])] += 1
        return {int(label): int(count) for label, count in sorted(counts.items())}

    def _class_histogram_for_samples(self, samples: Iterable[WildWestSample]) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for sample in samples:
            if sample.hidden_class is not None and 0 <= sample.hidden_class < NUM_CLASSES:
                counts[int(sample.hidden_class)] += 1
        return {int(label): int(count) for label, count in sorted(counts.items())}

    def _sample_unlabeled_samples(self) -> list[WildWestSample]:
        target_count = int(round(len(self.labeled_samples) * self.unlabeled_mix_ratio))
        if target_count <= 0:
            return []

        rng = np.random.default_rng(self.seed)
        slice_keys = list(self._rows_by_slice.keys())
        rng.shuffle(slice_keys)
        slice_quotas = _round_robin_quotas(target_count, slice_keys)
        used_unlabeled_by_slice: dict[tuple[str, int], set[tuple[int, int]]] = defaultdict(set)
        sampled: list[WildWestSample] = []

        for slice_key in slice_keys:
            selected = self._sample_from_slice(
                slice_key=slice_key,
                target_count=slice_quotas[slice_key],
                rng=rng,
                used_coords=used_unlabeled_by_slice[slice_key],
            )
            sampled.extend(selected)

        remaining = target_count - len(sampled)
        if remaining > 0:
            for slice_key in slice_keys:
                if remaining <= 0:
                    break
                selected = self._sample_from_slice(
                    slice_key=slice_key,
                    target_count=remaining,
                    rng=rng,
                    used_coords=used_unlabeled_by_slice[slice_key],
                )
                sampled.extend(selected)
                remaining = target_count - len(sampled)

        return sampled

    def _sample_from_slice(
        self,
        slice_key: tuple[str, int],
        target_count: int,
        rng: np.random.Generator,
        used_coords: set[tuple[int, int]],
    ) -> list[WildWestSample]:
        if target_count <= 0:
            return []

        name, z_index = slice_key
        label_slice = _read_slice(self._label_paths[name], z_index, np.int64)
        height, width = label_slice.shape
        if height <= 2 * self.half or width <= 2 * self.half:
            return []

        core_labels = label_slice[self.half : height - self.half, self.half : width - self.half]
        available = np.ones(core_labels.shape, dtype=bool)

        for y, x in self._labeled_lookup_by_slice[slice_key]:
            available[y - self.half, x - self.half] = False
        for y, x in used_coords:
            available[y - self.half, x - self.half] = False

        unique_classes = [
            int(class_id)
            for class_id in np.unique(core_labels)
            if 0 <= int(class_id) < NUM_CLASSES
        ]
        rng.shuffle(unique_classes)
        class_positions: dict[int, np.ndarray] = {}
        capacities: dict[int, int] = {}
        for class_id in unique_classes:
            class_mask = available & (core_labels == class_id)
            positions = np.argwhere(class_mask)
            if positions.size == 0:
                continue
            class_positions[class_id] = positions
            capacities[class_id] = int(len(positions))

        if not class_positions:
            return []

        class_order = list(class_positions.keys())
        class_targets = _round_robin_quotas(
            min(target_count, sum(capacities.values())),
            class_order,
            capacities=capacities,
        )

        sampled: list[WildWestSample] = []
        for class_id in class_order:
            class_target = class_targets[class_id]
            if class_target <= 0:
                continue
            positions = class_positions[class_id]
            selected_indices = rng.choice(len(positions), size=class_target, replace=False)
            for selected_index in np.atleast_1d(selected_indices):
                local_y, local_x = positions[int(selected_index)]
                y = int(local_y) + self.half
                x = int(local_x) + self.half
                used_coords.add((y, x))
                sampled.append(
                    WildWestSample(
                        name=name,
                        z=z_index,
                        y=y,
                        x=x,
                        is_labeled=False,
                        hidden_class=class_id,
                    )
                )

        return sampled


class WildWestMultiResDataset(WildWestCoordDataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        size: str = "low",
        patch_size: int = 25,
        resolution_scales: Sequence[float] = (1.0, 2.0),
        normalize_mean: float | None = None,
        normalize_std: float | None = None,
        csv_path: Path | None = None,
        unlabeled_mix_ratio: float = 0.0,
        seed: int = 42,
        include_unlabeled: bool = False,
    ) -> None:
        if not resolution_scales:
            raise ValueError("resolution_scales must contain at least one scale.")
        if any(scale <= 0 for scale in resolution_scales):
            raise ValueError(f"resolution_scales must be positive, got {tuple(resolution_scales)}.")
        self.resolution_scales = tuple(float(scale) for scale in resolution_scales)
        self.max_half_extent = (patch_size / 2.0) * max(self.resolution_scales)
        super().__init__(
            dataset_root=dataset_root,
            split=split,
            size=size,
            patch_size=patch_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            csv_path=csv_path,
            unlabeled_mix_ratio=unlabeled_mix_ratio,
            seed=seed,
            include_unlabeled=False,
        )

        volume_shapes = {name: _volume_shape(path) for name, path in self._source_paths.items()}
        filtered_rows, dropped = _filter_rows_for_continuous_extent(
            self.rows,
            volume_shapes,
            half_extent=self.max_half_extent,
        )
        if not filtered_rows:
            raise ValueError(
                f"Filtering {self.csv_path} for max MuViT half extent {self.max_half_extent:g} removed all rows."
            )
        if dropped > 0:
            print(
                f"Filtered {dropped} border-invalid rows from {self.csv_path} for "
                f"max MuViT half extent {self.max_half_extent:g}. Keeping {len(filtered_rows)} rows.",
                flush=True,
            )
        self.rows = filtered_rows
        self._rows_by_slice = _group_rows_by_slice(self.rows)
        self.class_counts = self._compute_center_class_counts(self.rows)
        self._labeled_lookup_by_slice = {
            slice_key: {(row.y, row.x) for row in slice_rows}
            for slice_key, slice_rows in self._rows_by_slice.items()
        }
        self.labeled_samples = [
            WildWestSample(
                name=row.name,
                z=row.z,
                y=row.y,
                x=row.x,
                is_labeled=True,
                hidden_class=None,
            )
            for row in self.rows
        ]
        self.include_unlabeled = include_unlabeled and unlabeled_mix_ratio > 0.0
        self.unlabeled_samples = []
        self.unlabeled_class_counts = {}
        if self.include_unlabeled:
            self.unlabeled_samples = self._sample_unlabeled_samples()
            self.unlabeled_class_counts = self._class_histogram_for_samples(self.unlabeled_samples)
        self.samples = [*self.labeled_samples, *self.unlabeled_samples]
        self.sample_counts = {
            "labeled": len(self.labeled_samples),
            "unlabeled": len(self.unlabeled_samples),
            "total": len(self.samples),
        }
        self.batch_sampler_metadata = None

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        image_slice = _read_slice(self._source_paths[sample.name], sample.z, np.float32)
        if self.normalize_mean is not None and self.normalize_std is not None:
            image_slice = (image_slice - self.normalize_mean) / self.normalize_std

        bbox = build_level_bboxes(
            sample.y,
            sample.x,
            patch_size=self.patch_size,
            resolution_scales=self.resolution_scales,
        )
        image_levels = torch.stack(
            [
                sample_bbox_from_slice(image_slice, level_bbox, out_size=self.patch_size, mode="bilinear")
                for level_bbox in bbox
            ],
            dim=0,
        )

        y0 = sample.y - self.half
        y1 = sample.y + self.half + 1
        x0 = sample.x - self.half
        x1 = sample.x + self.half + 1
        if sample.is_labeled:
            label_slice = _read_slice(self._label_paths[sample.name], sample.z, np.int64)
            segment_patch = np.asarray(label_slice[y0:y1, x0:x1], dtype=np.int64)
            center_label = int(label_slice[sample.y, sample.x])
        else:
            segment_patch = np.full((self.patch_size, self.patch_size), UNRECOGNIZED_LABEL, dtype=np.int64)
            center_label = UNRECOGNIZED_LABEL

        return {
            "img": image_levels.to(dtype=torch.float32),
            "bbox": bbox,
            "center_label": torch.tensor(center_label, dtype=torch.long),
            "segment": torch.from_numpy(segment_patch).unsqueeze(0).to(dtype=torch.long),
            "coords": torch.tensor((sample.z, sample.y, sample.x), dtype=torch.long),
            "is_labeled": torch.tensor(sample.is_labeled, dtype=torch.bool),
        }

    def _sample_from_slice(
        self,
        slice_key: tuple[str, int],
        target_count: int,
        rng: np.random.Generator,
        used_coords: set[tuple[int, int]],
    ) -> list[WildWestSample]:
        if target_count <= 0:
            return []

        name, z_index = slice_key
        label_slice = _read_slice(self._label_paths[name], z_index, np.int64)
        height, width = label_slice.shape
        min_y = int(math.ceil(self.max_half_extent))
        max_y = int(math.floor(height - self.max_half_extent))
        min_x = int(math.ceil(self.max_half_extent))
        max_x = int(math.floor(width - self.max_half_extent))
        if min_y > max_y or min_x > max_x:
            return []

        core_labels = label_slice[min_y : max_y + 1, min_x : max_x + 1]
        available = np.ones(core_labels.shape, dtype=bool)
        for y, x in self._labeled_lookup_by_slice[slice_key]:
            if min_y <= y <= max_y and min_x <= x <= max_x:
                available[y - min_y, x - min_x] = False
        for y, x in used_coords:
            if min_y <= y <= max_y and min_x <= x <= max_x:
                available[y - min_y, x - min_x] = False

        unique_classes = [
            int(class_id)
            for class_id in np.unique(core_labels)
            if 0 <= int(class_id) < NUM_CLASSES
        ]
        rng.shuffle(unique_classes)
        class_positions: dict[int, np.ndarray] = {}
        capacities: dict[int, int] = {}
        for class_id in unique_classes:
            positions = np.argwhere(available & (core_labels == class_id))
            if positions.size == 0:
                continue
            class_positions[class_id] = positions
            capacities[class_id] = int(len(positions))
        if not class_positions:
            return []

        class_order = list(class_positions.keys())
        class_targets = _round_robin_quotas(
            min(target_count, sum(capacities.values())),
            class_order,
            capacities=capacities,
        )
        sampled: list[WildWestSample] = []
        for class_id in class_order:
            class_target = class_targets[class_id]
            if class_target <= 0:
                continue
            positions = class_positions[class_id]
            selected_indices = rng.choice(len(positions), size=class_target, replace=False)
            for selected_index in np.atleast_1d(selected_indices):
                local_y, local_x = positions[int(selected_index)]
                y = int(local_y) + min_y
                x = int(local_x) + min_x
                used_coords.add((y, x))
                sampled.append(
                    WildWestSample(
                        name=name,
                        z=z_index,
                        y=y,
                        x=x,
                        is_labeled=False,
                        hidden_class=class_id,
                    )
                )
        return sampled


class WildWestBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labeled_indices: Sequence[int],
        unlabeled_indices: Sequence[int],
        batch_size: int,
        unlabeled_mix_ratio: float,
        seed: int = 42,
        batches_per_epoch: int | None = None,
    ) -> None:
        if batch_size <= 1:
            raise ValueError("Balanced labeled/unlabeled batches require batch_size > 1.")
        if unlabeled_mix_ratio <= 0.0:
            raise ValueError("unlabeled_mix_ratio must be > 0 for balanced batches.")
        if not labeled_indices:
            raise ValueError("Balanced batches require at least one labeled sample.")
        if not unlabeled_indices:
            raise ValueError("Balanced batches require at least one unlabeled sample.")

        unlabeled_fraction = unlabeled_mix_ratio / (1.0 + unlabeled_mix_ratio)
        unlabeled_per_batch = int(round(batch_size * unlabeled_fraction))
        unlabeled_per_batch = max(1, min(batch_size - 1, unlabeled_per_batch))
        labeled_per_batch = batch_size - unlabeled_per_batch

        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = list(unlabeled_indices)
        self.batch_size = batch_size
        self.unlabeled_mix_ratio = unlabeled_mix_ratio
        self.unlabeled_fraction = unlabeled_fraction
        self.labeled_per_batch = labeled_per_batch
        self.unlabeled_per_batch = unlabeled_per_batch
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch or math.ceil(
            (len(self.labeled_indices) + len(self.unlabeled_indices)) / batch_size
        )
        self._epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        labeled_pool = self._shuffled_pool(rng, self.labeled_indices)
        unlabeled_pool = self._shuffled_pool(rng, self.unlabeled_indices)
        labeled_cursor = 0
        unlabeled_cursor = 0

        for _ in range(self.batches_per_epoch):
            labeled_batch, labeled_pool, labeled_cursor = self._take(
                rng,
                source=self.labeled_indices,
                pool=labeled_pool,
                cursor=labeled_cursor,
                count=self.labeled_per_batch,
            )
            unlabeled_batch, unlabeled_pool, unlabeled_cursor = self._take(
                rng,
                source=self.unlabeled_indices,
                pool=unlabeled_pool,
                cursor=unlabeled_cursor,
                count=self.unlabeled_per_batch,
            )
            batch = [*labeled_batch, *unlabeled_batch]
            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch

    @staticmethod
    def _shuffled_pool(rng: np.random.Generator, source: Sequence[int]) -> list[int]:
        return rng.permutation(np.asarray(source, dtype=np.int64)).astype(int).tolist()

    @classmethod
    def _take(
        cls,
        rng: np.random.Generator,
        source: Sequence[int],
        pool: list[int],
        cursor: int,
        count: int,
    ) -> tuple[list[int], list[int], int]:
        selected: list[int] = []
        while len(selected) < count:
            if cursor >= len(pool):
                pool = cls._shuffled_pool(rng, source)
                cursor = 0
            take_count = min(count - len(selected), len(pool) - cursor)
            selected.extend(pool[cursor : cursor + take_count])
            cursor += take_count
        return selected, pool, cursor

    def metadata(self) -> dict[str, object]:
        return {
            "sampler": "WildWestBalancedBatchSampler",
            "seed": self.seed,
            "batch_size": self.batch_size,
            "batches_per_epoch": self.batches_per_epoch,
            "unlabeled_mix_ratio": self.unlabeled_mix_ratio,
            "unlabeled_fraction": self.unlabeled_fraction,
            "labeled_per_batch": self.labeled_per_batch,
            "unlabeled_per_batch": self.unlabeled_per_batch,
            "labeled_pool_size": len(self.labeled_indices),
            "unlabeled_pool_size": len(self.unlabeled_indices),
        }


def build_wild_west_train_batch_sampler(
    dataset: WildWestCoordDataset,
    batch_size: int,
    unlabeled_mix_ratio: float,
    seed: int,
) -> WildWestBalancedBatchSampler | None:
    if unlabeled_mix_ratio <= 0.0 or not dataset.unlabeled_samples:
        return None

    labeled_count = len(dataset.labeled_samples)
    unlabeled_count = len(dataset.unlabeled_samples)
    return WildWestBalancedBatchSampler(
        labeled_indices=range(labeled_count),
        unlabeled_indices=range(labeled_count, labeled_count + unlabeled_count),
        batch_size=batch_size,
        unlabeled_mix_ratio=unlabeled_mix_ratio,
        seed=seed,
    )


def log_dataset_composition(train_dataset: WildWestCoordDataset, val_dataset: WildWestCoordDataset) -> None:
    print(
        "train dataset composition: "
        f"labeled={train_dataset.sample_counts['labeled']} "
        f"unlabeled={train_dataset.sample_counts['unlabeled']} "
        f"total={train_dataset.sample_counts['total']} "
        f"labeled_class_counts={train_dataset.class_counts} "
        f"unlabeled_class_counts={train_dataset.unlabeled_class_counts}",
        flush=True,
    )
    print(
        "val dataset composition: "
        f"labeled={val_dataset.sample_counts['labeled']} "
        f"unlabeled={val_dataset.sample_counts['unlabeled']} "
        f"total={val_dataset.sample_counts['total']} "
        f"labeled_class_counts={val_dataset.class_counts}",
        flush=True,
    )


def build_train_val_loaders(
    dataset_root: Path,
    dataset_size: str,
    batch_size: int,
    patch_size: int,
    num_workers: int,
    train_coords_csv: Path | None = None,
    val_coords_csv: Path | None = None,
    unlabeled_mix_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, tuple[float, float]]:
    train_csv_path = train_coords_csv or _csv_path_for_split(dataset_root, size=dataset_size, split="train")
    val_csv_path = val_coords_csv or _csv_path_for_split(dataset_root, size=dataset_size, split="val")
    train_rows = load_coordinate_rows(train_csv_path)
    train_mean, train_std = compute_source_stats(dataset_root, names=(row.name for row in train_rows))

    train_dataset = WildWestCoordDataset(
        dataset_root=dataset_root,
        split="train",
        size=dataset_size,
        patch_size=patch_size,
        normalize_mean=train_mean,
        normalize_std=train_std,
        csv_path=train_csv_path,
        unlabeled_mix_ratio=unlabeled_mix_ratio,
        seed=seed,
        include_unlabeled=True,
    )
    val_dataset = WildWestCoordDataset(
        dataset_root=dataset_root,
        split="val",
        size=dataset_size,
        patch_size=patch_size,
        normalize_mean=train_mean,
        normalize_std=train_std,
        csv_path=val_csv_path,
        unlabeled_mix_ratio=0.0,
        seed=seed,
        include_unlabeled=False,
    )

    log_dataset_composition(train_dataset, val_dataset)

    train_batch_sampler = build_wild_west_train_batch_sampler(
        train_dataset,
        batch_size=batch_size,
        unlabeled_mix_ratio=unlabeled_mix_ratio,
        seed=seed,
    )
    if train_batch_sampler is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
            generator=make_torch_generator(seed),
        )
    else:
        train_dataset.batch_sampler_metadata = train_batch_sampler.metadata()
        print(
            "train balanced batch sampler: "
            f"labeled_per_batch={train_batch_sampler.labeled_per_batch} "
            f"unlabeled_per_batch={train_batch_sampler.unlabeled_per_batch} "
            f"batches_per_epoch={len(train_batch_sampler)} "
            f"seed={train_batch_sampler.seed}",
            flush=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
            generator=make_torch_generator(seed),
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed + 1),
    )
    return train_loader, val_loader, (train_mean, train_std)


def build_train_val_multires_loaders(
    dataset_root: Path,
    dataset_size: str,
    batch_size: int,
    patch_size: int,
    resolution_scales: Sequence[float],
    num_workers: int,
    train_coords_csv: Path | None = None,
    val_coords_csv: Path | None = None,
    unlabeled_mix_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, tuple[float, float]]:
    train_csv_path = train_coords_csv or _csv_path_for_split(dataset_root, size=dataset_size, split="train")
    val_csv_path = val_coords_csv or _csv_path_for_split(dataset_root, size=dataset_size, split="val")
    train_rows = load_coordinate_rows(train_csv_path)
    train_mean, train_std = compute_source_stats(dataset_root, names=(row.name for row in train_rows))

    train_dataset = WildWestMultiResDataset(
        dataset_root=dataset_root,
        split="train",
        size=dataset_size,
        patch_size=patch_size,
        resolution_scales=resolution_scales,
        normalize_mean=train_mean,
        normalize_std=train_std,
        csv_path=train_csv_path,
        unlabeled_mix_ratio=unlabeled_mix_ratio,
        seed=seed,
        include_unlabeled=True,
    )
    val_dataset = WildWestMultiResDataset(
        dataset_root=dataset_root,
        split="val",
        size=dataset_size,
        patch_size=patch_size,
        resolution_scales=resolution_scales,
        normalize_mean=train_mean,
        normalize_std=train_std,
        csv_path=val_csv_path,
        unlabeled_mix_ratio=0.0,
        seed=seed,
        include_unlabeled=False,
    )

    log_dataset_composition(train_dataset, val_dataset)

    train_batch_sampler = build_wild_west_train_batch_sampler(
        train_dataset,
        batch_size=batch_size,
        unlabeled_mix_ratio=unlabeled_mix_ratio,
        seed=seed,
    )
    if train_batch_sampler is None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
            generator=make_torch_generator(seed),
        )
    else:
        train_dataset.batch_sampler_metadata = train_batch_sampler.metadata()
        print(
            "train balanced batch sampler: "
            f"labeled_per_batch={train_batch_sampler.labeled_per_batch} "
            f"unlabeled_per_batch={train_batch_sampler.unlabeled_per_batch} "
            f"batches_per_epoch={len(train_batch_sampler)} "
            f"seed={train_batch_sampler.seed}",
            flush=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
            generator=make_torch_generator(seed),
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed + 1),
    )
    return train_loader, val_loader, (train_mean, train_std)


class PretrainDataset(Dataset):
    """Randomly samples patches from full 3D volumes for unsupervised pretraining.

    Unlike BetaSegCoordDataset, this uses no coordinate CSV — coordinates are
    pre-sampled uniformly at random from all valid positions in the given volumes.
    All samples are unlabeled (center_label = UNRECOGNIZED_LABEL).
    """

    def __init__(
        self,
        dataset_root: Path,
        names: list[str],
        patch_size: int,
        normalize_mean: float,
        normalize_std: float,
        num_samples: int,
        seed: int = 42,
    ) -> None:
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.num_samples = num_samples

        self.source_paths: dict[str, Path] = {}
        self.volume_shapes: dict[str, tuple[int, int, int]] = {}
        for name in names:
            source_path, _ = _resolve_volume_paths(dataset_root, name)
            self.source_paths[name] = source_path
            self.volume_shapes[name] = _volume_shape(source_path)

        self.coords = self._sample_coords(names, seed)

    def _sample_coords(self, names: list[str], seed: int) -> list[tuple[str, int, int, int]]:
        rng = np.random.default_rng(seed)
        coords = []
        for _ in range(self.num_samples):
            name = names[rng.integers(len(names))]
            depth, height, width = self.volume_shapes[name]
            z = int(rng.integers(0, depth))
            y = int(rng.integers(self.half, height - self.half))
            x = int(rng.integers(self.half, width - self.half))
            coords.append((name, z, y, x))
        return coords

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        name, z, y, x = self.coords[idx]
        slice_2d = _read_slice(self.source_paths[name], z, dtype=np.float32)
        patch = slice_2d[y - self.half: y + self.half + 1, x - self.half: x + self.half + 1]
        patch_tensor = torch.from_numpy((patch - self.normalize_mean) / self.normalize_std).unsqueeze(0)
        segment_tensor = torch.full((1, self.patch_size, self.patch_size), UNRECOGNIZED_LABEL, dtype=torch.long)
        return (
            patch_tensor,
            torch.tensor(UNRECOGNIZED_LABEL, dtype=torch.long),
            segment_tensor,
            torch.tensor([z, y, x], dtype=torch.long),
            torch.tensor(False, dtype=torch.bool),
        )


class PretrainMultiResDataset(PretrainDataset):
    """Multi-resolution version of PretrainDataset for MuViT pretraining.

    Returns the same dict format as WildWestMultiResDataset.
    """

    def __init__(
        self,
        dataset_root: Path,
        names: list[str],
        patch_size: int,
        normalize_mean: float,
        normalize_std: float,
        num_samples: int,
        resolution_scales: Sequence[float],
        seed: int = 42,
    ) -> None:
        self.resolution_scales = list(resolution_scales)
        self.max_half_extent = math.ceil((patch_size / 2.0) * max(resolution_scales))
        super().__init__(dataset_root, names, patch_size, normalize_mean, normalize_std, num_samples, seed)

    def _sample_coords(self, names: list[str], seed: int) -> list[tuple[str, int, int, int]]:
        rng = np.random.default_rng(seed)
        coords = []
        margin = self.max_half_extent
        for _ in range(self.num_samples):
            name = names[rng.integers(len(names))]
            depth, height, width = self.volume_shapes[name]
            z = int(rng.integers(0, depth))
            y = int(rng.integers(margin, height - margin))
            x = int(rng.integers(margin, width - margin))
            coords.append((name, z, y, x))
        return coords

    def __getitem__(self, idx: int) -> dict:
        name, z, y, x = self.coords[idx]
        slice_2d = _read_slice(self.source_paths[name], z, dtype=np.float32)
        bbox = build_level_bboxes(y, x, self.patch_size, self.resolution_scales)
        imgs = torch.stack([
            sample_bbox_from_slice(slice_2d, bbox[level], out_size=self.patch_size)
            for level in range(len(self.resolution_scales))
        ])
        imgs = (imgs - self.normalize_mean) / self.normalize_std
        segment_tensor = torch.full((1, self.patch_size, self.patch_size), UNRECOGNIZED_LABEL, dtype=torch.long)
        return {
            "img": imgs,
            "bbox": bbox,
            "center_label": torch.tensor(UNRECOGNIZED_LABEL, dtype=torch.long),
            "segment": segment_tensor,
            "coords": torch.tensor([z, y, x], dtype=torch.long),
            "is_labeled": torch.tensor(False, dtype=torch.bool),
        }


def build_pretrain_loader(
    dataset_root: Path,
    names: list[str],
    patch_size: int,
    normalize_mean: float,
    normalize_std: float,
    num_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    resolution_scales: Sequence[float] | None = None,
) -> DataLoader:
    if resolution_scales is not None:
        dataset: Dataset = PretrainMultiResDataset(
            dataset_root=dataset_root,
            names=names,
            patch_size=patch_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            num_samples=num_samples,
            resolution_scales=resolution_scales,
            seed=seed,
        )
    else:
        dataset = PretrainDataset(
            dataset_root=dataset_root,
            names=names,
            patch_size=patch_size,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            num_samples=num_samples,
            seed=seed,
        )
    print(f"pretrain dataset: {len(dataset)} random patches from volumes {names}", flush=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed),
    )


def run_balanced_batch_sampler_self_tests() -> None:
    labeled_indices = list(range(100))
    unlabeled_indices = list(range(100, 125))
    sampler = WildWestBalancedBatchSampler(
        labeled_indices=labeled_indices,
        unlabeled_indices=unlabeled_indices,
        batch_size=32,
        unlabeled_mix_ratio=0.25,
        seed=7,
        batches_per_epoch=4,
    )
    first_batches = list(iter(sampler))
    assert len(first_batches) == 4
    for batch in first_batches:
        labeled_count = sum(index < 100 for index in batch)
        unlabeled_count = sum(index >= 100 for index in batch)
        assert len(batch) == 32
        assert labeled_count == 26
        assert unlabeled_count == 6

    same_seed_batches = list(
        iter(
            WildWestBalancedBatchSampler(
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                batch_size=32,
                unlabeled_mix_ratio=0.25,
                seed=7,
                batches_per_epoch=4,
            )
        )
    )
    different_seed_batches = list(
        iter(
            WildWestBalancedBatchSampler(
                labeled_indices=labeled_indices,
                unlabeled_indices=unlabeled_indices,
                batch_size=32,
                unlabeled_mix_ratio=0.25,
                seed=8,
                batches_per_epoch=4,
            )
        )
    )
    assert first_batches == same_seed_batches
    assert first_batches != different_seed_batches

    class DummyDataset:
        labeled_samples = [object()]
        unlabeled_samples: list[object] = []

    assert build_wild_west_train_batch_sampler(DummyDataset(), 32, 0.0, 7) is None
