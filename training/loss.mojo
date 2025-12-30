"""
Loss Functions for Classification
==================================
Implements cross-entropy and related loss functions for training.
"""

from models.tensor3d import Tensor3D, Shape5D, FloatDType
from math import log


fn cross_entropy_loss(pred: Tensor3D, target: Tensor3D) -> Scalar[FloatDType]:
    """
    Cross-entropy loss for multi-class classification.
    
    Args:
        pred: Softmax probabilities (batch, num_classes, 1, 1, 1)
        target: One-hot encoded targets (batch, num_classes, 1, 1, 1)
        
    Returns:
        Scalar loss value
    """
    let batch = pred.shape.batch
    let num_classes = pred.shape.channels
    var total_loss: Scalar[FloatDType] = 0.0
    
    for b in range(batch):
        for c in range(num_classes):
            let p = pred[b, c, 0, 0, 0]
            let t = target[b, c, 0, 0, 0]
            # Only add loss for true class (t > 0)
            if t > 0.5:
                # Clip prediction to prevent log(0)
                let p_clipped = max(p, 1e-7)
                total_loss += -log(p_clipped)
    
    return total_loss / Float32(batch)


fn cross_entropy_loss_backward(pred: Tensor3D, target: Tensor3D) -> Tensor3D:
    """
    Gradient of cross-entropy loss w.r.t. softmax output.
    
    For softmax + cross-entropy, the gradient simplifies to: pred - target
    
    Args:
        pred: Softmax probabilities
        target: One-hot encoded targets
        
    Returns:
        Gradient tensor (same shape as pred)
    """
    let batch = pred.shape.batch
    var grad = pred.sub(target)
    
    # Scale by 1/batch for mean reduction
    grad.mul_scalar(1.0 / Float32(batch))
    
    return grad


fn binary_cross_entropy(pred: Tensor3D, target: Tensor3D) -> Scalar[FloatDType]:
    """
    Binary cross-entropy loss.
    
    Args:
        pred: Sigmoid probabilities
        target: Binary targets (0 or 1)
        
    Returns:
        Scalar loss value
    """
    var total_loss: Scalar[FloatDType] = 0.0
    let n = pred.numel()
    
    for i in range(n):
        let p = pred.data.load(i)
        let t = target.data.load(i)
        
        # Clip to prevent log(0)
        let p_clipped = max(min(p, 1.0 - 1e-7), 1e-7)
        total_loss += -(t * log(p_clipped) + (1.0 - t) * log(1.0 - p_clipped))
    
    return total_loss / Float32(n)


fn binary_cross_entropy_backward(pred: Tensor3D, target: Tensor3D) -> Tensor3D:
    """
    Gradient of binary cross-entropy.
    
    dL/dp = (p - t) / (p * (1 - p))
    
    For sigmoid + BCE, simplified to: pred - target
    """
    return pred.sub(target)


fn mse_loss(pred: Tensor3D, target: Tensor3D) -> Scalar[FloatDType]:
    """
    Mean squared error loss.
    
    Returns:
        Scalar MSE loss
    """
    var total_loss: Scalar[FloatDType] = 0.0
    let n = pred.numel()
    
    for i in range(n):
        let diff = pred.data.load(i) - target.data.load(i)
        total_loss += diff * diff
    
    return total_loss / Float32(n)


fn mse_loss_backward(pred: Tensor3D, target: Tensor3D) -> Tensor3D:
    """Gradient of MSE loss: 2 * (pred - target) / n"""
    let n = pred.numel()
    var grad = pred.sub(target)
    grad.mul_scalar(2.0 / Float32(n))
    return grad


fn max(a: Scalar[FloatDType], b: Scalar[FloatDType]) -> Scalar[FloatDType]:
    if a > b:
        return a
    return b


fn min(a: Scalar[FloatDType], b: Scalar[FloatDType]) -> Scalar[FloatDType]:
    if a < b:
        return a
    return b


# ============ Label Encoding ============

fn to_one_hot(labels: Tensor3D, num_classes: Int) -> Tensor3D:
    """
    Convert class indices to one-hot encoded tensor.
    
    Args:
        labels: Tensor with class indices (batch, 1, 1, 1, 1)
        num_classes: Number of classes
        
    Returns:
        One-hot encoded tensor (batch, num_classes, 1, 1, 1)
    """
    let batch = labels.shape.batch
    var one_hot = Tensor3D.zeros(Shape5D(batch, num_classes, 1, 1, 1))
    
    for b in range(batch):
        let class_idx = Int(labels[b, 0, 0, 0, 0])
        if class_idx >= 0 and class_idx < num_classes:
            one_hot[b, class_idx, 0, 0, 0] = 1.0
    
    return one_hot
