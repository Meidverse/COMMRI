"""
MRI Classification Training Script (Standalone)
=================================================
Simplified standalone training script for Mojo.
"""

from python import Python
from python.object import PythonObject


fn create_synthetic_volume(batch_size: Int, depth: Int, height: Int, width: Int, seed: Int) raises -> PythonObject:
    """Create synthetic MRI volume using NumPy."""
    var np = Python.import_module("numpy")
    np.random.seed(seed)
    
    # Create random volume
    var volume = np.random.randn(batch_size, 1, depth, height, width).astype(np.float32)
    
    return volume


fn create_synthetic_labels(batch_size: Int, num_classes: Int, seed: Int) raises -> PythonObject:
    """Create one-hot encoded labels."""
    var np = Python.import_module("numpy")
    np.random.seed(seed)
    
    var labels = np.zeros((batch_size, num_classes), dtype=np.float32)
    for b in range(batch_size):
        var class_idx = (seed + b) % num_classes
        labels[b, class_idx] = 1.0
    
    return labels


fn simple_forward(weights: PythonObject, x: PythonObject) raises -> PythonObject:
    """Simple forward pass - single linear layer for demo."""
    var np = Python.import_module("numpy")
    
    # Flatten input
    var batch_size = Int(x.shape[0])
    var flat_x = x.reshape(batch_size, -1)
    
    # Linear layer: y = x @ W + b
    var y = np.matmul(flat_x, weights["W"]) + weights["b"]
    
    # Softmax
    var exp_y = np.exp(y - np.max(y, axis=1, keepdims=True))
    var softmax = exp_y / np.sum(exp_y, axis=1, keepdims=True)
    
    return softmax


fn compute_loss(np: PythonObject, pred: PythonObject, target: PythonObject) raises -> Float64:
    """Cross-entropy loss."""
    # Clip predictions to avoid log(0)
    var eps = 1e-7
    var pred_clipped = np.clip(pred, eps, 1.0 - eps)
    
    # Cross-entropy: -sum(target * log(pred))
    var batch = Float64(pred.shape[0])
    var loss = -np.sum(target * np.log(pred_clipped)) / batch
    
    return Float64(loss)


fn compute_accuracy(np: PythonObject, pred: PythonObject, target: PythonObject) raises -> Float64:
    """Compute classification accuracy."""
    var pred_classes = np.argmax(pred, axis=1)
    var true_classes = np.argmax(target, axis=1)
    var correct = np.sum(pred_classes == true_classes)
    var batch = Float64(pred.shape[0])
    
    return Float64(correct) / batch


fn main() raises:
    print("=" * 60)
    print("MRI Classification System - Mojo Training")
    print("=" * 60)
    
    var np = Python.import_module("numpy")
    
    # Configuration
    var num_epochs = 50
    var batch_size = 4
    var num_batches = 20
    var num_classes = 2
    var input_size = 32  # Smaller for demo
    var learning_rate = 0.001
    
    print("\nConfiguration:")
    print("  Epochs:", num_epochs)
    print("  Batch size:", batch_size)
    print("  Input size:", input_size, "^3")
    print("  Num classes:", num_classes)
    print("  Learning rate:", learning_rate)
    
    # Initialize simple linear weights
    var input_features = 1 * input_size * input_size * input_size  # C * D * H * W flattened
    print("  Input features:", input_features)
    
    np.random.seed(42)
    var weights = Python.dict()
    weights["W"] = np.random.randn(input_features, num_classes).astype(np.float32) * 0.01
    weights["b"] = np.zeros(num_classes, dtype=np.float32)
    
    print("\nStarting training...")
    print("-" * 60)
    
    var best_loss = 1000000.0
    
    for epoch in range(num_epochs):
        var epoch_loss = 0.0
        var epoch_acc = 0.0
        
        for batch_idx in range(num_batches):
            # Create synthetic batch
            var seed = epoch * num_batches + batch_idx
            var x = create_synthetic_volume(batch_size, input_size, input_size, input_size, seed)
            var y = create_synthetic_labels(batch_size, num_classes, seed)
            
            # Forward pass
            var pred = simple_forward(weights, x)
            
            # Compute loss and accuracy
            var loss = compute_loss(np, pred, y)
            var acc = compute_accuracy(np, pred, y)
            
            epoch_loss += loss
            epoch_acc += acc
            
            # Simple gradient descent update
            # Gradient of cross-entropy w.r.t. softmax output
            var batch_f = Float64(batch_size)
            var grad_pred = (pred - y) / batch_f
            
            # Gradient w.r.t. weights
            var flat_x = x.reshape(batch_size, -1)
            var grad_W = np.matmul(flat_x.T, grad_pred)
            var grad_b = np.sum(grad_pred, axis=0)
            
            # Update weights
            weights["W"] = weights["W"] - learning_rate * grad_W
            weights["b"] = weights["b"] - learning_rate * grad_b
        
        # Average metrics
        var num_batches_f = Float64(num_batches)
        var avg_loss = epoch_loss / num_batches_f
        var avg_acc = epoch_acc / num_batches_f
        
        # Track best
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print("Epoch", epoch + 1, "/", num_epochs, 
                  "| Loss:", avg_loss, 
                  "| Acc:", avg_acc)
    
    print("-" * 60)
    print("\nTraining Complete!")
    print("Best Loss:", best_loss)
    print("=" * 60)
