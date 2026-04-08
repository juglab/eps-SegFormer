from __future__ import annotations

import numpy as np
import torch

NUM_CLASSES = 4
UNRECOGNIZED_LABEL = -1


def remap_label_array(labels: np.ndarray) -> np.ndarray:
    return labels.astype(np.int64, copy=True)


def remap_label_tensor(labels: torch.Tensor) -> torch.Tensor:
    return labels.long().clone()


def valid_class_mask(labels: torch.Tensor) -> torch.Tensor:
    return (labels >= 0) & (labels < NUM_CLASSES)
