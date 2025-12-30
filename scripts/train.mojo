"""
MRI Classification Training Script (Standalone)
=================================================
Simplified standalone training script for Mojo.
"""

from python import Python


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
    var weights_W = np.random.randn(input_features, num_classes).astype(np.float32) * 0.01
    var weights_b = np.zeros(num_classes, dtype=np.float32)
    
    print("\nStarting training...")
    print("-" * 60)
    
    var best_loss = 1000000.0
    
    for epoch in range(num_epochs):
        var epoch_loss = 0.0
        var epoch_acc = 0.0
        
        for batch_idx in range(num_batches):
            # Create synthetic batch
            var seed = epoch * num_batches + batch_idx
            np.random.seed(seed)
            
            # Create synthetic volume
            var x = np.random.randn(batch_size, 1, input_size, input_size, input_size).astype(np.float32)
            
            # Create one-hot labels
            var y = np.zeros((batch_size, num_classes), dtype=np.float32)
            for b in range(batch_size):
                var class_idx = (seed + b) % num_classes
                y[b, class_idx] = 1.0
            
            # Forward pass
            var flat_x = x.reshape(batch_size, -1)
            var logits = np.matmul(flat_x, weights_W) + weights_b
            
            # Softmax
            var exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            var pred = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Cross-entropy loss
            var eps = 1e-7
            var pred_clipped = np.clip(pred, eps, 1.0 - eps)
            var loss_val = -np.sum(y * np.log(pred_clipped)) / batch_size
            var loss = Float64(loss_val)
            
            # Accuracy
            var pred_classes = np.argmax(pred, axis=1)
            var true_classes = np.argmax(y, axis=1)
            var correct = np.sum(pred_classes == true_classes)
            var acc = Float64(correct) / Float64(batch_size)
            
            epoch_loss += loss
            epoch_acc += acc
            
            # Gradient descent update
            var grad_pred = (pred - y) / batch_size
            var grad_W = np.matmul(flat_x.T, grad_pred)
            var grad_b = np.sum(grad_pred, axis=0)
            
            # Update weights
            weights_W = weights_W - learning_rate * grad_W
            weights_b = weights_b - learning_rate * grad_b
        
        # Average metrics
        var avg_loss = epoch_loss / Float64(num_batches)
        var avg_acc = epoch_acc / Float64(num_batches)
        
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
