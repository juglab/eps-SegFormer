from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .common import append_timestamp


def plot_training_history(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [x['epoch'] for x in history]
    train_loss = [x['train_loss'] for x in history]
    val_loss = [x['val_loss'] for x in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='train_mse', marker='o', linewidth=1)
    plt.plot(epochs, val_loss, label='val_mse', marker='o', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Progression')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = append_timestamp(output_dir / 'loss_curve.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('saved loss curve to:', output_path)
