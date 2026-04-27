from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .common import append_timestamp


def plot_training_history(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [x['epoch'] for x in history]
    train_loss = [x['train_loss'] for x in history]
    val_loss = [x['val_loss'] for x in history]
    train_mse = [x.get('train_mse', x['train_loss']) for x in history]
    val_mse = [x.get('val_mse', x['val_loss']) for x in history]
    train_ce = [x.get('train_ce', 0.0) for x in history]
    val_ce = [x.get('val_ce', 0.0) for x in history]
    train_acc = [x.get('train_cls_acc', 0.0) for x in history]
    val_acc = [x.get('val_cls_acc', 0.0) for x in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(epochs, train_loss, label='train_total', marker='o', linewidth=1)
    axes[0].plot(epochs, val_loss, label='val_total', marker='o', linewidth=1)
    axes[0].plot(epochs, train_mse, label='train_mse', linestyle='--', linewidth=1)
    axes[0].plot(epochs, val_mse, label='val_mse', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total And Reconstruction Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.4)

    axes[1].plot(epochs, train_ce, label='train_ce', marker='o', linewidth=1)
    axes[1].plot(epochs, val_ce, label='val_ce', marker='o', linewidth=1)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cross-Entropy')
    axes[1].set_title('Center-Token Classification Loss')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.4)

    axes[2].plot(epochs, train_acc, label='train_acc', marker='o', linewidth=1)
    axes[2].plot(epochs, val_acc, label='val_acc', marker='o', linewidth=1)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Center-Token Classification Accuracy')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('Training Progression')
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = append_timestamp(output_dir / 'loss_curve.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved loss curve to:', output_path)
