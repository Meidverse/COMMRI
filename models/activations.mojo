"""
Activation Functions for Neural Networks
=========================================
Implements common activation functions with forward and backward passes.
"""

from .tensor3d import Tensor3D, Shape5D, FloatDType, simd_width
from algorithm import vectorize
from math import exp, log


struct ReLU:
    """Rectified Linear Unit activation."""
    var _input_cache: Tensor3D
    
    fn __init__(inout self):
        self._input_cache = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """ReLU(x) = max(0, x)"""
        self._input_cache = x
        return x.relu()
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Gradient is 1 where x > 0, else 0."""
        return self._input_cache.relu_backward(grad)


struct LeakyReLU:
    """Leaky ReLU activation."""
    var alpha: Scalar[FloatDType]
    var _input_cache: Tensor3D
    
    fn __init__(inout self, alpha: Float32 = 0.01):
        self.alpha = alpha
        self._input_cache = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """LeakyReLU(x) = x if x > 0 else alpha * x"""
        self._input_cache = x
        var output = Tensor3D.zeros(x.shape)
        
        for i in range(x.numel()):
            let val = x.data.load(i)
            if val > 0:
                output.data.store(i, val)
            else:
                output.data.store(i, self.alpha * val)
        
        return output
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Gradient is 1 where x > 0, else alpha."""
        var input_grad = Tensor3D.zeros(grad.shape)
        
        for i in range(grad.numel()):
            let x_val = self._input_cache.data.load(i)
            let g_val = grad.data.load(i)
            if x_val > 0:
                input_grad.data.store(i, g_val)
            else:
                input_grad.data.store(i, self.alpha * g_val)
        
        return input_grad


struct Sigmoid:
    """Sigmoid activation."""
    var _output_cache: Tensor3D
    
    fn __init__(inout self):
        self._output_cache = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """Sigmoid(x) = 1 / (1 + exp(-x))"""
        var output = Tensor3D.zeros(x.shape)
        
        for i in range(x.numel()):
            let val = x.data.load(i)
            # Clip to prevent overflow
            let clipped = max(min(val, 88.0), -88.0)
            output.data.store(i, 1.0 / (1.0 + exp(-clipped)))
        
        self._output_cache = output
        return output
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Gradient: sigmoid(x) * (1 - sigmoid(x))"""
        var input_grad = Tensor3D.zeros(grad.shape)
        
        for i in range(grad.numel()):
            let s = self._output_cache.data.load(i)
            let g = grad.data.load(i)
            input_grad.data.store(i, g * s * (1.0 - s))
        
        return input_grad


struct Softmax:
    """
    Softmax activation for classification.
    Applied over channels dimension (dim=1).
    """
    var _output_cache: Tensor3D
    
    fn __init__(inout self):
        self._output_cache = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """
        Softmax over channels: exp(x_c) / sum(exp(x_i))
        Input/Output shape: (batch, channels, 1, 1, 1) or (batch, channels, D, H, W)
        """
        let batch = x.shape.batch
        let c = x.shape.channels
        let d = x.shape.depth
        let h = x.shape.height
        let w = x.shape.width
        
        var output = Tensor3D.zeros(x.shape)
        
        for b in range(batch):
            for dd in range(d):
                for hh in range(h):
                    for ww in range(w):
                        # Find max for numerical stability
                        var max_val = x[b, 0, dd, hh, ww]
                        for ch in range(1, c):
                            let val = x[b, ch, dd, hh, ww]
                            if val > max_val:
                                max_val = val
                        
                        # Compute exp(x - max) and sum
                        var sum_exp: Scalar[FloatDType] = 0.0
                        for ch in range(c):
                            let e = exp(x[b, ch, dd, hh, ww] - max_val)
                            output[b, ch, dd, hh, ww] = e
                            sum_exp += e
                        
                        # Normalize
                        for ch in range(c):
                            output[b, ch, dd, hh, ww] = output[b, ch, dd, hh, ww] / sum_exp
        
        self._output_cache = output
        return output
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """
        Softmax backward pass.
        For cross-entropy loss, this is typically simplified.
        """
        let batch = grad.shape.batch
        let c = grad.shape.channels
        let d = grad.shape.depth
        let h = grad.shape.height
        let w = grad.shape.width
        
        var input_grad = Tensor3D.zeros(grad.shape)
        
        for b in range(batch):
            for dd in range(d):
                for hh in range(h):
                    for ww in range(w):
                        # Compute dot product of softmax output and gradient
                        var dot_prod: Scalar[FloatDType] = 0.0
                        for ch in range(c):
                            dot_prod += self._output_cache[b, ch, dd, hh, ww] * grad[b, ch, dd, hh, ww]
                        
                        # Compute gradient: s_i * (g_i - dot(s, g))
                        for ch in range(c):
                            let s = self._output_cache[b, ch, dd, hh, ww]
                            let g = grad[b, ch, dd, hh, ww]
                            input_grad[b, ch, dd, hh, ww] = s * (g - dot_prod)
        
        return input_grad


fn max(a: Scalar[FloatDType], b: Scalar[FloatDType]) -> Scalar[FloatDType]:
    """Max of two values."""
    if a > b:
        return a
    return b


fn min(a: Scalar[FloatDType], b: Scalar[FloatDType]) -> Scalar[FloatDType]:
    """Min of two values."""
    if a < b:
        return a
    return b
