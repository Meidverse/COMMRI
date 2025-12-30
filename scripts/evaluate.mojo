"""
MRI Classification Evaluation Script
======================================
Evaluate a trained model on test data and generate metrics.

Usage:
    mojo run scripts/evaluate.mojo --model_path ./checkpoints/best_model.bin --data_dir ./data/test
"""

from models.tensor3d import Tensor3D, Shape5D
from models.network import VGG3D
from evaluation.metrics import MetricsTracker, ConfusionMatrix, ClassificationMetrics


fn evaluate_model(
    model: VGG3D,
    test_batches: List[Tuple[Tensor3D, Tensor3D]],
    num_classes: Int
) -> ClassificationMetrics:
    """
    Evaluate model on test data.
    
    Args:
        model: Trained VGG3D model
        test_batches: List of (input, label) tuples
        num_classes: Number of classes
        
    Returns:
        Computed metrics
    """
    var tracker = MetricsTracker(num_classes)
    var model_mut = model
    model_mut.eval()
    
    for i in range(len(test_batches)):
        let batch = test_batches[i]
        let x = batch[0]
        let y = batch[1]
        
        # Forward pass
        let predictions = model_mut.forward(x)
        
        # Update metrics (loss placeholder)
        tracker.update(predictions, y, 0.0)
    
    tracker.print_report()
    return tracker.compute_metrics()


fn create_test_batch(
    batch_size: Int,
    input_shape: Tuple[Int, Int, Int],
    num_classes: Int,
    seed: Int
) -> Tuple[Tensor3D, Tensor3D]:
    """Create synthetic test batch."""
    let d = input_shape[0]
    let h = input_shape[1]
    let w = input_shape[2]
    
    var x = Tensor3D.randn(Shape5D(batch_size, 1, d, h, w), seed)
    
    var y = Tensor3D.zeros(Shape5D(batch_size, num_classes, 1, 1, 1))
    for b in range(batch_size):
        let class_idx = (seed + b) % num_classes
        y[b, class_idx, 0, 0, 0] = 1.0
    
    return (x, y)


fn main():
    """Main evaluation entry point."""
    print("=" * 60)
    print("MRI Classification System - Evaluation")
    print("=" * 60)
    
    # Configuration
    let num_classes = 2
    let input_shape = (64, 64, 64)
    let batch_size = 4
    
    # Initialize model (in production, load from checkpoint)
    print("\nLoading model...")
    var model = VGG3D(
        in_channels=1,
        num_classes=num_classes,
        base_filters=32,
        dropout_rate=0.5
    )
    model.eval()
    print("  Model loaded with", model.num_parameters(), "parameters")
    
    # Create synthetic test data
    print("\nPreparing test data (synthetic for demo)...")
    var test_batches = List[Tuple[Tensor3D, Tensor3D]]()
    let num_test_batches = 5
    
    for i in range(num_test_batches):
        let batch = create_test_batch(batch_size, input_shape, num_classes, i * 1000)
        test_batches.append(batch)
    
    print("  Test samples:", num_test_batches * batch_size)
    
    # Evaluate
    print("\nRunning evaluation...")
    let metrics = evaluate_model(model, test_batches, num_classes)
    
    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print("\nTest Metrics:")
    print("  Accuracy:", metrics.accuracy)
    print("  Precision:", metrics.precision)
    print("  Recall:", metrics.recall)
    print("  F1 Score:", metrics.f1_score)
    print("=" * 60)
