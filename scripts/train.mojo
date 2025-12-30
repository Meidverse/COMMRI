"""
MRI Classification Training Script
====================================
Main entry point for training the 3D CNN on MRI data.

Usage (from WSL2):
    cd /mnt/c/mri
    source .venv/bin/activate
    mojo run scripts/train.mojo --data_dir ./data/mri_dataset --epochs 100

This script uses Python interop for data loading and Mojo for training.
"""

from python import Python
from models.tensor3d import Tensor3D, Shape5D
from models.network import VGG3D, create_model
from training.trainer import Trainer, TrainingConfig
from training.loss import to_one_hot
from evaluation.metrics import MetricsTracker


fn load_config(config_path: String) -> TrainingConfig:
    """Load configuration from YAML file (uses Python)."""
    # Default config
    return TrainingConfig(
        epochs=100,
        batch_size=4,
        learning_rate=0.001,
        weight_decay=0.0001,
        early_stopping_patience=15,
        checkpoint_dir="./checkpoints",
        log_frequency=10
    )


fn create_synthetic_batch(
    batch_size: Int,
    input_shape: Tuple[Int, Int, Int],
    num_classes: Int,
    seed: Int
) -> Tuple[Tensor3D, Tensor3D]:
    """Create synthetic batch for testing."""
    let d = input_shape[0]
    let h = input_shape[1]
    let w = input_shape[2]
    
    # Random input
    var x = Tensor3D.randn(Shape5D(batch_size, 1, d, h, w), seed)
    
    # Random labels
    var y = Tensor3D.zeros(Shape5D(batch_size, num_classes, 1, 1, 1))
    for b in range(batch_size):
        let class_idx = (seed + b) % num_classes
        y[b, class_idx, 0, 0, 0] = 1.0
    
    return (x, y)


fn main():
    """Main training entry point."""
    print("=" * 60)
    print("MRI Classification System - Training")
    print("=" * 60)
    
    # Configuration
    let config = load_config("config/config.yaml")
    let num_classes = 2
    let input_shape = (64, 64, 64)  # D, H, W
    
    print("\nConfiguration:")
    print("  Epochs:", config.epochs)
    print("  Batch Size:", config.batch_size)
    print("  Learning Rate:", config.learning_rate)
    print("  Input Shape:", input_shape[0], "x", input_shape[1], "x", input_shape[2])
    print("  Num Classes:", num_classes)
    
    # Create model
    print("\nInitializing model...")
    var model = VGG3D(
        in_channels=1,
        num_classes=num_classes,
        base_filters=32,
        dropout_rate=0.5
    )
    print("  Model initialized with", model.num_parameters(), "parameters")
    
    # Create trainer
    var trainer = Trainer(model, config)
    
    # For demonstration, create synthetic data
    # In production, load real data using Python data module
    print("\nPreparing data (synthetic for demo)...")
    var train_batches = List[Tuple[Tensor3D, Tensor3D]]()
    var val_batches = List[Tuple[Tensor3D, Tensor3D]]()
    
    let num_train_batches = 10
    let num_val_batches = 3
    
    for i in range(num_train_batches):
        let batch = create_synthetic_batch(config.batch_size, input_shape, num_classes, i * 100)
        train_batches.append(batch)
    
    for i in range(num_val_batches):
        let batch = create_synthetic_batch(config.batch_size, input_shape, num_classes, (num_train_batches + i) * 100)
        val_batches.append(batch)
    
    print("  Train batches:", num_train_batches)
    print("  Val batches:", num_val_batches)
    
    # Train
    print("\nStarting training...")
    let final_metrics = trainer.fit(train_batches, val_batches)
    
    # Final report
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nFinal Metrics:")
    print("  Best Val Loss:", final_metrics.best_val_loss)
    print("  Final Train Loss:", final_metrics.train_loss)
    print("  Final Train Accuracy:", final_metrics.train_accuracy)
    print("  Final Val Accuracy:", final_metrics.val_accuracy)
    print("  Epochs Trained:", final_metrics.epoch + 1)
    
    print("\nModel saved to:", config.checkpoint_dir)
    print("=" * 60)
