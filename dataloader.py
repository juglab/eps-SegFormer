from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import DataLoader, Dataset


EXPECTED_COLUMNS = ("name", "z", "y", "x")
DATASETS_RELATIVE_DIR = Path("datasets") / "betaseg"
COORDS_RELATIVE_DIR = Path("baseline_coords")
VALID_SPLITS = ("train", "val", "test")
AUTO_FILTER_BORDER_PATCH_SIZES = frozenset({65})

@dataclass(frozen=True)
class CoordRow:
    name: str
    z: int
    y: int
    x: int
    row_index: int


def _csv_path_for_split(dataset_root: Path, size: str, split: str) -> Path:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Use one of {VALID_SPLITS}.")
    return dataset_root / COORDS_RELATIVE_DIR / f"2D_{size}_{split}.csv"


def _resolve_volume_paths(dataset_root: Path, name: str) -> tuple[Path, Path]:
    data_root = dataset_root / DATASETS_RELATIVE_DIR
    source_path = data_root / name / f"{name}_source.tif"
    label_path = data_root / name / f"{name}_gt.tif"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source TIFF for '{name}': {source_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing GT TIFF for '{name}': {label_path}")
    return source_path, label_path


def _volume_shape(path: Path) -> tuple[int, int, int]:
    with tiff.TiffFile(path) as tiff_file:
        shape = tiff_file.series[0].shape
    if len(shape) != 3:
        raise ValueError(f"Expected 3D TIFF volume at {path}, got shape {shape}.")
    return int(shape[0]), int(shape[1]), int(shape[2])


def _read_slice(path: Path, z_index: int, dtype: np.dtype) -> np.ndarray:
    with tiff.TiffFile(path) as tiff_file:
        return tiff_file.pages[z_index].asarray().astype(dtype, copy=False)


def load_coordinate_rows(csv_path: Path) -> list[CoordRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        if tuple(reader.fieldnames) != EXPECTED_COLUMNS:
            raise ValueError(
                f"CSV {csv_path} must have columns {EXPECTED_COLUMNS}, got {tuple(reader.fieldnames)}."
            )

        rows: list[CoordRow] = []
        for row_index, row in enumerate(reader, start=2):
            rows.append(
                CoordRow(
                    name=str(row["name"]),
                    z=int(row["z"]),
                    y=int(row["y"]),
                    x=int(row["x"]),
                    row_index=row_index,
                )
            )
    if not rows:
        raise ValueError(f"CSV contains no samples: {csv_path}")
    return rows


def validate_coordinate_rows(
    rows: Iterable[CoordRow],
    volume_shapes: dict[str, tuple[int, int, int]],
    patch_size: int,
    csv_path: Path,
) -> None:
    half = patch_size // 2
    for row in rows:
        if row.name not in volume_shapes:
            raise ValueError(f"CSV {csv_path} row {row.row_index} references unknown dataset '{row.name}'.")
        depth, height, width = volume_shapes[row.name]
        if not 0 <= row.z < depth:
            raise ValueError(
                f"CSV {csv_path} row {row.row_index} has z={row.z} outside [0, {depth - 1}] for '{row.name}'."
            )
        if row.y - half < 0 or row.y + half >= height:
            raise ValueError(
                f"CSV {csv_path} row {row.row_index} has y={row.y} outside valid centered range "
                f"[{half}, {height - half - 1}] for patch_size={patch_size} and '{row.name}'."
            )
        if row.x - half < 0 or row.x + half >= width:
            raise ValueError(
                f"CSV {csv_path} row {row.row_index} has x={row.x} outside valid centered range "
                f"[{half}, {width - half - 1}] for patch_size={patch_size} and '{row.name}'."
            )


def filter_coordinate_rows_for_patch_size(
    rows: Iterable[CoordRow],
    volume_shapes: dict[str, tuple[int, int, int]],
    patch_size: int,
) -> tuple[list[CoordRow], int]:
    half = patch_size // 2
    kept_rows: list[CoordRow] = []
    dropped = 0

    for row in rows:
        if row.name not in volume_shapes:
            raise ValueError(f"Unknown dataset '{row.name}' referenced by row {row.row_index}.")
        depth, height, width = volume_shapes[row.name]
        is_valid = (
            0 <= row.z < depth
            and half <= row.y <= height - half - 1
            and half <= row.x <= width - half - 1
        )
        if is_valid:
            kept_rows.append(row)
        else:
            dropped += 1

    return kept_rows, dropped


def compute_source_stats(dataset_root: Path, names: Iterable[str]) -> tuple[float, float]:
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for name in sorted(set(names)):
        source_path, _ = _resolve_volume_paths(dataset_root, name)
        volume = tiff.imread(source_path).astype(np.float32, copy=False)
        total_sum += float(np.sum(volume, dtype=np.float64))
        total_sq_sum += float(np.sum(np.square(volume, dtype=np.float64), dtype=np.float64))
        total_count += int(volume.size)

    if total_count == 0:
        raise ValueError("Cannot compute normalization statistics from zero source pixels.")

    mean = total_sum / total_count
    variance = max(total_sq_sum / total_count - mean * mean, 0.0)
    std = variance ** 0.5
    if std == 0.0:
        raise ValueError("Training source volumes have zero standard deviation; normalization would divide by zero.")
    return float(mean), float(std)


class BetaSegCoordDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split: str,
        size: str = "low",
        patch_size: int = 25,
        normalize_mean: float | None = None,
        normalize_std: float | None = None,
        csv_path: Path | None = None,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or patch_size % 2 == 0:
            raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}.")

        self.dataset_root = dataset_root
        self.split = split
        self.size = size
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.csv_path = csv_path or _csv_path_for_split(dataset_root, size=size, split=split)
        self.rows = load_coordinate_rows(self.csv_path)

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
        else:
            validate_coordinate_rows(self.rows, volume_shapes, patch_size=patch_size, csv_path=self.csv_path)
        self.class_counts = self._compute_class_counts()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_slice = _read_slice(self._source_paths[row.name], row.z, np.float32)
        label_slice = _read_slice(self._label_paths[row.name], row.z, np.int64)

        y0 = row.y - self.half
        y1 = row.y + self.half + 1
        x0 = row.x - self.half
        x1 = row.x + self.half + 1

        image_patch = np.asarray(image_slice[y0:y1, x0:x1], dtype=np.float32)
        if self.normalize_mean is not None and self.normalize_std is not None:
            image_patch = (image_patch - self.normalize_mean) / self.normalize_std
        segment_patch = np.asarray(label_slice[y0:y1, x0:x1], dtype=np.int64)

        patch_tensor = torch.from_numpy(image_patch).unsqueeze(0).to(dtype=torch.float32)
        segment_tensor = torch.from_numpy(segment_patch).unsqueeze(0).to(dtype=torch.long)
        center_label = torch.tensor(int(label_slice[row.y, row.x]), dtype=torch.long)
        coords_tensor = torch.tensor((row.z, row.y, row.x), dtype=torch.long)
        return patch_tensor, center_label, segment_tensor, coords_tensor

    def _compute_class_counts(self) -> dict[int, int]:
        counts: Counter[int] = Counter()
        for row in self.rows:
            label_slice = _read_slice(self._label_paths[row.name], row.z, np.int64)
            counts[int(label_slice[row.y, row.x])] += 1
        return {int(label): int(count) for label, count in sorted(counts.items())}


def log_dataset_class_counts(train_dataset: BetaSegCoordDataset, val_dataset: BetaSegCoordDataset) -> None:
    print(
        f"train dataset samples: total={len(train_dataset)} class_counts={train_dataset.class_counts}",
        flush=True,
    )
    print(
        f"val dataset samples: total={len(val_dataset)} class_counts={val_dataset.class_counts}",
        flush=True,
    )


def build_split_dataloader(
    dataset_root: Path,
    split: str,
    size: str,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    normalize_mean: float | None,
    normalize_std: float | None,
    shuffle: bool,
    csv_path: Path | None = None,
) -> DataLoader:
    dataset = BetaSegCoordDataset(
        dataset_root=dataset_root,
        split=split,
        size=size,
        patch_size=patch_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        csv_path=csv_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def build_train_val_loaders(
    dataset_root: Path,
    dataset_size: str,
    batch_size: int,
    patch_size: int,
    num_workers: int,
    train_coords_csv: Path | None = None,
    val_coords_csv: Path | None = None,
) -> tuple[DataLoader, DataLoader, tuple[float, float]]:
    train_csv_path = train_coords_csv or _csv_path_for_split(dataset_root, size=dataset_size, split="train")
    val_csv_path = val_coords_csv or _csv_path_for_split(dataset_root, size=dataset_size, split="val")
    train_rows = load_coordinate_rows(train_csv_path)
    train_mean, train_std = compute_source_stats(dataset_root, names=(row.name for row in train_rows))

    train_dataset = BetaSegCoordDataset(
        dataset_root=dataset_root,
        split="train",
        size=dataset_size,
        patch_size=patch_size,
        normalize_mean=train_mean,
        normalize_std=train_std,
        csv_path=train_csv_path,
    )
    val_dataset = BetaSegCoordDataset(
        dataset_root=dataset_root,
        split="val",
        size=dataset_size,
        patch_size=patch_size,
        normalize_mean=train_mean,
        normalize_std=train_std,
        csv_path=val_csv_path,
    )

    log_dataset_class_counts(train_dataset, val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, (train_mean, train_std)
