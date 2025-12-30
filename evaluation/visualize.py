"""
Visualization Utilities for MRI Classification
===============================================
Provides plotting functions for training metrics, predictions, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dict with keys like 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', color='#2E86AB')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', color='#E94F37')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'train_accuracy' in history:
        axes[1].plot(history['train_accuracy'], label='Train Acc', color='#2E86AB')
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Val Acc', color='#E94F37')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        confusion_matrix: NxN numpy array
        class_names: List of class names
        save_path: Optional path to save figure
        figsize: Figure size
        normalize: If True, normalize rows to sum to 1
    """
    if normalize:
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = confusion_matrix.astype(float) / row_sums
    else:
        cm = confusion_matrix.astype(float)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap
    colors = ['#FFFFFF', '#2E86AB']
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Labels
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted Label',
        ylabel='True Label',
        title='Confusion Matrix'
    )
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Dict[str, float]:
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: One-hot encoded true labels (N, num_classes)
        y_scores: Predicted probabilities (N, num_classes)
        class_names: List of class names
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Dictionary of AUC values per class
    """
    from sklearn.metrics import roc_curve, auc
    
    n_classes = len(class_names)
    colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    auc_values = {}
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values[class_name] = roc_auc
        
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return auc_values


def plot_mri_slices(
    volume: np.ndarray,
    prediction: Optional[str] = None,
    true_label: Optional[str] = None,
    probability: Optional[float] = None,
    save_path: Optional[str] = None,
    num_slices: int = 5
) -> None:
    """
    Plot slices from an MRI volume with prediction info.
    
    Args:
        volume: 3D numpy array (D, H, W)
        prediction: Predicted class name
        true_label: True class name
        probability: Prediction probability
        save_path: Optional path to save figure
        num_slices: Number of slices to show
    """
    depth = volume.shape[0]
    slice_indices = np.linspace(depth // 4, 3 * depth // 4, num_slices, dtype=int)
    
    fig, axes = plt.subplots(1, num_slices, figsize=(num_slices * 3, 3))
    
    for i, (ax, idx) in enumerate(zip(axes, slice_indices)):
        ax.imshow(volume[idx], cmap='gray')
        ax.set_title(f'Slice {idx}')
        ax.axis('off')
    
    # Add prediction info as suptitle
    title_parts = []
    if prediction:
        title_parts.append(f"Predicted: {prediction}")
    if true_label:
        title_parts.append(f"True: {true_label}")
    if probability is not None:
        title_parts.append(f"Confidence: {probability:.2%}")
    
    if title_parts:
        color = 'green' if prediction == true_label else 'red'
        fig.suptitle(' | '.join(title_parts), fontsize=12, color=color)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved MRI slices to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sample_predictions(
    volumes: List[np.ndarray],
    predictions: List[str],
    true_labels: List[str],
    probabilities: List[float],
    save_dir: str,
    num_samples: int = 6
) -> None:
    """
    Plot multiple sample predictions.
    
    Args:
        volumes: List of 3D volumes
        predictions: List of predicted class names
        true_labels: List of true class names
        probabilities: List of prediction probabilities
        save_dir: Directory to save figures
        num_samples: Number of samples to plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(num_samples, len(volumes))):
        save_path = os.path.join(save_dir, f'sample_{i}.png')
        plot_mri_slices(
            volumes[i],
            prediction=predictions[i],
            true_label=true_labels[i],
            probability=probabilities[i],
            save_path=save_path
        )


def plot_learning_rate_schedule(
    learning_rates: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot learning rate over epochs.
    
    Args:
        learning_rates: List of learning rates per epoch
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 4))
    plt.plot(learning_rates, color='#2E86AB', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved learning rate schedule to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_evaluation_report(
    metrics: Dict[str, float],
    confusion_matrix: np.ndarray,
    class_names: List[str],
    history: Dict[str, List[float]],
    output_dir: str
) -> None:
    """
    Create a complete evaluation report with all visualizations.
    
    Args:
        metrics: Dictionary of evaluation metrics
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        history: Training history dictionary
        output_dir: Directory to save all outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        confusion_matrix,
        class_names,
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    # Create text report
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("MRI Classification Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Classification Metrics:\n")
        f.write("-" * 30 + "\n")
        for metric_name, value in metrics.items():
            f.write(f"  {metric_name}: {value:.4f}\n")
        
        f.write("\n\nConfusion Matrix:\n")
        f.write("-" * 30 + "\n")
        f.write("  " + "  ".join(f"{name[:8]:>8}" for name in class_names) + "\n")
        for i, row in enumerate(confusion_matrix):
            f.write(f"{class_names[i][:8]:>8}: " + "  ".join(f"{int(v):>8}" for v in row) + "\n")
    
    print(f"Evaluation report saved to {output_dir}")
