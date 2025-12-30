"""
Neural Network Layers for 3D Medical Imaging
=============================================
Implements Conv3D, BatchNorm3D, Pooling, and Linear layers.
Each layer supports forward and backward passes for training.
"""

from .tensor3d import Tensor3D, Shape5D, FloatDType, simd_width
from memory import memset_zero, memcpy
from algorithm import vectorize, parallelize
from math import sqrt


# ============ Abstract Layer Trait ============

trait Layer:
    """Interface for neural network layers."""
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        ...
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        ...


# ============ Convolutional Layers ============

struct Conv3D:
    """
    3D Convolutional layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolving kernel (D, H, W)
        stride: Stride of convolution
        padding: Zero-padding added to input
    """
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var stride: Int
    var padding: Int
    
    # Parameters
    var weight: Tensor3D  # Shape: (out_channels, in_channels, kD, kH, kW)
    var bias: Tensor3D    # Shape: (out_channels, 1, 1, 1, 1)
    
    # Gradients
    var weight_grad: Tensor3D
    var bias_grad: Tensor3D
    
    # Cache for backward
    var _input_cache: Tensor3D
    
    fn __init__(
        inout self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int = 3,
        stride: Int = 1,
        padding: Int = 1
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights with Kaiming/He initialization
        let fan_in = in_channels * kernel_size * kernel_size * kernel_size
        self.weight = Tensor3D.kaiming_normal(
            Shape5D(out_channels, in_channels, kernel_size, kernel_size, kernel_size),
            fan_in
        )
        self.bias = Tensor3D.zeros(Shape5D(out_channels, 1, 1, 1, 1))
        
        # Initialize gradients
        self.weight_grad = Tensor3D.zeros(self.weight.shape)
        self.bias_grad = Tensor3D.zeros(self.bias.shape)
        
        # Placeholder for input cache
        self._input_cache = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """
        Forward pass of 3D convolution.
        
        Input: (batch, in_channels, D, H, W)
        Output: (batch, out_channels, D_out, H_out, W_out)
        """
        # Cache input for backward
        self._input_cache = x
        
        let batch = x.shape.batch
        let d_in = x.shape.depth
        let h_in = x.shape.height
        let w_in = x.shape.width
        
        # Calculate output dimensions
        let d_out = (d_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        let h_out = (h_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        let w_out = (w_in + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        var output = Tensor3D.zeros(Shape5D(batch, self.out_channels, d_out, h_out, w_out))
        
        # Convolution (naive implementation - can be optimized with im2col)
        for b in range(batch):
            for oc in range(self.out_channels):
                for od in range(d_out):
                    for oh in range(h_out):
                        for ow in range(w_out):
                            var sum_val: Scalar[FloatDType] = 0.0
                            
                            for ic in range(self.in_channels):
                                for kd in range(self.kernel_size):
                                    for kh in range(self.kernel_size):
                                        for kw in range(self.kernel_size):
                                            let id = od * self.stride + kd - self.padding
                                            let ih = oh * self.stride + kh - self.padding
                                            let iw = ow * self.stride + kw - self.padding
                                            
                                            if id >= 0 and id < d_in and ih >= 0 and ih < h_in and iw >= 0 and iw < w_in:
                                                sum_val += x[b, ic, id, ih, iw] * self.weight[oc, ic, kd, kh, kw]
                            
                            # Add bias
                            sum_val += self.bias[oc, 0, 0, 0, 0]
                            output[b, oc, od, oh, ow] = sum_val
        
        return output
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """
        Backward pass of 3D convolution.
        Computes gradients w.r.t. input, weights, and bias.
        """
        let x = self._input_cache
        let batch = x.shape.batch
        
        # Reset gradients
        self.weight_grad = Tensor3D.zeros(self.weight.shape)
        self.bias_grad = Tensor3D.zeros(self.bias.shape)
        
        var input_grad = Tensor3D.zeros(x.shape)
        
        let d_out = grad.shape.depth
        let h_out = grad.shape.height
        let w_out = grad.shape.width
        
        # Compute gradients
        for b in range(batch):
            for oc in range(self.out_channels):
                for od in range(d_out):
                    for oh in range(h_out):
                        for ow in range(w_out):
                            let g = grad[b, oc, od, oh, ow]
                            
                            # Bias gradient
                            self.bias_grad[oc, 0, 0, 0, 0] += g
                            
                            for ic in range(self.in_channels):
                                for kd in range(self.kernel_size):
                                    for kh in range(self.kernel_size):
                                        for kw in range(self.kernel_size):
                                            let id = od * self.stride + kd - self.padding
                                            let ih = oh * self.stride + kh - self.padding
                                            let iw = ow * self.stride + kw - self.padding
                                            
                                            if id >= 0 and id < x.shape.depth and ih >= 0 and ih < x.shape.height and iw >= 0 and iw < x.shape.width:
                                                # Weight gradient
                                                self.weight_grad[oc, ic, kd, kh, kw] += g * x[b, ic, id, ih, iw]
                                                # Input gradient
                                                input_grad[b, ic, id, ih, iw] += g * self.weight[oc, ic, kd, kh, kw]
        
        return input_grad
    
    fn parameters(self) -> List[Tensor3D]:
        """Return list of layer parameters."""
        var params = List[Tensor3D]()
        params.append(self.weight)
        params.append(self.bias)
        return params
    
    fn gradients(self) -> List[Tensor3D]:
        """Return list of parameter gradients."""
        var grads = List[Tensor3D]()
        grads.append(self.weight_grad)
        grads.append(self.bias_grad)
        return grads


# ============ Batch Normalization ============

struct BatchNorm3D:
    """
    3D Batch Normalization layer.
    
    Normalizes over (N, D, H, W) dimensions, keeping C dimension.
    """
    var num_features: Int
    var eps: Scalar[FloatDType]
    var momentum: Scalar[FloatDType]
    var training: Bool
    
    # Learnable parameters
    var gamma: Tensor3D  # Scale
    var beta: Tensor3D   # Shift
    
    # Running statistics
    var running_mean: Tensor3D
    var running_var: Tensor3D
    
    # Gradients
    var gamma_grad: Tensor3D
    var beta_grad: Tensor3D
    
    # Cache for backward
    var _x_norm: Tensor3D
    var _std_inv: Tensor3D
    var _input_cache: Tensor3D
    
    fn __init__(inout self, num_features: Int, eps: Float32 = 1e-5, momentum: Float32 = 0.1):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = True
        
        # Initialize parameters
        self.gamma = Tensor3D.ones(Shape5D(1, num_features, 1, 1, 1))
        self.beta = Tensor3D.zeros(Shape5D(1, num_features, 1, 1, 1))
        
        self.running_mean = Tensor3D.zeros(Shape5D(1, num_features, 1, 1, 1))
        self.running_var = Tensor3D.ones(Shape5D(1, num_features, 1, 1, 1))
        
        self.gamma_grad = Tensor3D.zeros(Shape5D(1, num_features, 1, 1, 1))
        self.beta_grad = Tensor3D.zeros(Shape5D(1, num_features, 1, 1, 1))
        
        # Placeholders
        self._x_norm = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
        self._std_inv = Tensor3D.zeros(Shape5D(1, num_features, 1, 1, 1))
        self._input_cache = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """Forward pass with batch normalization."""
        self._input_cache = x
        
        let batch = x.shape.batch
        let c = x.shape.channels
        let d = x.shape.depth
        let h = x.shape.height
        let w = x.shape.width
        let spatial_size = Float32(batch * d * h * w)
        
        var output = Tensor3D.zeros(x.shape)
        
        if self.training:
            # Compute batch mean and variance per channel
            for ch in range(c):
                var mean_val: Scalar[FloatDType] = 0.0
                var var_val: Scalar[FloatDType] = 0.0
                
                # Compute mean
                for b in range(batch):
                    for dd in range(d):
                        for hh in range(h):
                            for ww in range(w):
                                mean_val += x[b, ch, dd, hh, ww]
                mean_val /= spatial_size
                
                # Compute variance
                for b in range(batch):
                    for dd in range(d):
                        for hh in range(h):
                            for ww in range(w):
                                let diff = x[b, ch, dd, hh, ww] - mean_val
                                var_val += diff * diff
                var_val /= spatial_size
                
                let std_inv = 1.0 / sqrt(var_val + self.eps)
                self._std_inv[0, ch, 0, 0, 0] = std_inv
                
                # Update running stats
                self.running_mean[0, ch, 0, 0, 0] = (1.0 - self.momentum) * self.running_mean[0, ch, 0, 0, 0] + self.momentum * mean_val
                self.running_var[0, ch, 0, 0, 0] = (1.0 - self.momentum) * self.running_var[0, ch, 0, 0, 0] + self.momentum * var_val
                
                # Normalize and scale
                for b in range(batch):
                    for dd in range(d):
                        for hh in range(h):
                            for ww in range(w):
                                let x_norm = (x[b, ch, dd, hh, ww] - mean_val) * std_inv
                                output[b, ch, dd, hh, ww] = self.gamma[0, ch, 0, 0, 0] * x_norm + self.beta[0, ch, 0, 0, 0]
        else:
            # Use running statistics during inference
            for ch in range(c):
                let mean_val = self.running_mean[0, ch, 0, 0, 0]
                let var_val = self.running_var[0, ch, 0, 0, 0]
                let std_inv = 1.0 / sqrt(var_val + self.eps)
                
                for b in range(batch):
                    for dd in range(d):
                        for hh in range(h):
                            for ww in range(w):
                                let x_norm = (x[b, ch, dd, hh, ww] - mean_val) * std_inv
                                output[b, ch, dd, hh, ww] = self.gamma[0, ch, 0, 0, 0] * x_norm + self.beta[0, ch, 0, 0, 0]
        
        self._x_norm = output  # Cache for backward
        return output
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Backward pass for batch normalization."""
        let x = self._input_cache
        let batch = x.shape.batch
        let c = x.shape.channels
        let d = x.shape.depth
        let h = x.shape.height
        let w = x.shape.width
        let N = Float32(batch * d * h * w)
        
        var input_grad = Tensor3D.zeros(x.shape)
        
        # Reset gradients
        self.gamma_grad = Tensor3D.zeros(self.gamma.shape)
        self.beta_grad = Tensor3D.zeros(self.beta.shape)
        
        for ch in range(c):
            let std_inv = self._std_inv[0, ch, 0, 0, 0]
            let gamma_val = self.gamma[0, ch, 0, 0, 0]
            
            var dgamma: Scalar[FloatDType] = 0.0
            var dbeta: Scalar[FloatDType] = 0.0
            var dmean: Scalar[FloatDType] = 0.0
            var dvar: Scalar[FloatDType] = 0.0
            
            # First pass: compute dgamma, dbeta
            for b in range(batch):
                for dd in range(d):
                    for hh in range(h):
                        for ww in range(w):
                            let g = grad[b, ch, dd, hh, ww]
                            let x_norm = (self._x_norm[b, ch, dd, hh, ww] - self.beta[0, ch, 0, 0, 0]) / gamma_val
                            dgamma += g * x_norm
                            dbeta += g
            
            self.gamma_grad[0, ch, 0, 0, 0] = dgamma
            self.beta_grad[0, ch, 0, 0, 0] = dbeta
            
            # Input gradient (simplified)
            for b in range(batch):
                for dd in range(d):
                    for hh in range(h):
                        for ww in range(w):
                            input_grad[b, ch, dd, hh, ww] = grad[b, ch, dd, hh, ww] * gamma_val * std_inv
        
        return input_grad
    
    fn train(inout self):
        """Set to training mode."""
        self.training = True
    
    fn eval(inout self):
        """Set to evaluation mode."""
        self.training = False


# ============ Pooling Layers ============

struct MaxPool3D:
    """3D Max Pooling layer."""
    var kernel_size: Int
    var stride: Int
    
    var _max_indices: Tensor3D  # Cache for backward
    var _input_shape: Shape5D
    
    fn __init__(inout self, kernel_size: Int = 2, stride: Int = 2):
        self.kernel_size = kernel_size
        self.stride = stride
        self._max_indices = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
        self._input_shape = Shape5D(1, 1, 1, 1, 1)
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """Forward pass of max pooling."""
        self._input_shape = x.shape
        
        let batch = x.shape.batch
        let c = x.shape.channels
        let d_out = x.shape.depth // self.stride
        let h_out = x.shape.height // self.stride
        let w_out = x.shape.width // self.stride
        
        var output = Tensor3D.zeros(Shape5D(batch, c, d_out, h_out, w_out))
        self._max_indices = Tensor3D.zeros(Shape5D(batch, c, d_out, h_out, w_out))
        
        for b in range(batch):
            for ch in range(c):
                for od in range(d_out):
                    for oh in range(h_out):
                        for ow in range(w_out):
                            var max_val = x[b, ch, od * self.stride, oh * self.stride, ow * self.stride]
                            var max_idx: Int = 0
                            
                            for kd in range(self.kernel_size):
                                for kh in range(self.kernel_size):
                                    for kw in range(self.kernel_size):
                                        let val = x[b, ch, 
                                            od * self.stride + kd,
                                            oh * self.stride + kh,
                                            ow * self.stride + kw]
                                        if val > max_val:
                                            max_val = val
                                            max_idx = kd * self.kernel_size * self.kernel_size + kh * self.kernel_size + kw
                            
                            output[b, ch, od, oh, ow] = max_val
                            self._max_indices[b, ch, od, oh, ow] = Float32(max_idx)
        
        return output
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Backward pass - route gradients to max positions."""
        var input_grad = Tensor3D.zeros(self._input_shape)
        
        let batch = grad.shape.batch
        let c = grad.shape.channels
        let d_out = grad.shape.depth
        let h_out = grad.shape.height
        let w_out = grad.shape.width
        
        for b in range(batch):
            for ch in range(c):
                for od in range(d_out):
                    for oh in range(h_out):
                        for ow in range(w_out):
                            let max_idx = Int(self._max_indices[b, ch, od, oh, ow])
                            let kd = max_idx // (self.kernel_size * self.kernel_size)
                            let kh = (max_idx % (self.kernel_size * self.kernel_size)) // self.kernel_size
                            let kw = max_idx % self.kernel_size
                            
                            input_grad[b, ch,
                                od * self.stride + kd,
                                oh * self.stride + kh,
                                ow * self.stride + kw] = grad[b, ch, od, oh, ow]
        
        return input_grad


struct GlobalAvgPool3D:
    """Global Average Pooling - reduces spatial dims to 1x1x1."""
    var _input_shape: Shape5D
    
    fn __init__(inout self):
        self._input_shape = Shape5D(1, 1, 1, 1, 1)
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """Average all spatial positions per channel."""
        self._input_shape = x.shape
        
        let batch = x.shape.batch
        let c = x.shape.channels
        let spatial_size = Float32(x.shape.depth * x.shape.height * x.shape.width)
        
        var output = Tensor3D.zeros(Shape5D(batch, c, 1, 1, 1))
        
        for b in range(batch):
            for ch in range(c):
                var sum_val: Scalar[FloatDType] = 0.0
                for d in range(x.shape.depth):
                    for h in range(x.shape.height):
                        for w in range(x.shape.width):
                            sum_val += x[b, ch, d, h, w]
                output[b, ch, 0, 0, 0] = sum_val / spatial_size
        
        return output
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Distribute gradient equally to all positions."""
        let spatial_size = Float32(self._input_shape.depth * self._input_shape.height * self._input_shape.width)
        var input_grad = Tensor3D.zeros(self._input_shape)
        
        for b in range(self._input_shape.batch):
            for ch in range(self._input_shape.channels):
                let g = grad[b, ch, 0, 0, 0] / spatial_size
                for d in range(self._input_shape.depth):
                    for h in range(self._input_shape.height):
                        for w in range(self._input_shape.width):
                            input_grad[b, ch, d, h, w] = g
        
        return input_grad


# ============ Fully Connected Layer ============

struct Linear:
    """Fully connected (dense) layer."""
    var in_features: Int
    var out_features: Int
    
    var weight: Tensor3D  # (out_features, in_features, 1, 1, 1)
    var bias: Tensor3D    # (out_features, 1, 1, 1, 1)
    
    var weight_grad: Tensor3D
    var bias_grad: Tensor3D
    
    var _input_cache: Tensor3D
    
    fn __init__(inout self, in_features: Int, out_features: Int):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier/Glorot initialization
        let std = sqrt(2.0 / Float32(in_features + out_features))
        self.weight = Tensor3D.randn(Shape5D(out_features, in_features, 1, 1, 1))
        self.weight.mul_scalar(std)
        self.bias = Tensor3D.zeros(Shape5D(out_features, 1, 1, 1, 1))
        
        self.weight_grad = Tensor3D.zeros(self.weight.shape)
        self.bias_grad = Tensor3D.zeros(self.bias.shape)
        
        self._input_cache = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """
        Forward pass: y = xW^T + b
        Input: (batch, in_features, 1, 1, 1)
        Output: (batch, out_features, 1, 1, 1)
        """
        self._input_cache = x
        let batch = x.shape.batch
        
        var output = Tensor3D.zeros(Shape5D(batch, self.out_features, 1, 1, 1))
        
        for b in range(batch):
            for o in range(self.out_features):
                var sum_val: Scalar[FloatDType] = 0.0
                for i in range(self.in_features):
                    sum_val += x[b, i, 0, 0, 0] * self.weight[o, i, 0, 0, 0]
                output[b, o, 0, 0, 0] = sum_val + self.bias[o, 0, 0, 0, 0]
        
        return output
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Backward pass: compute gradients."""
        let x = self._input_cache
        let batch = x.shape.batch
        
        # Reset gradients
        self.weight_grad = Tensor3D.zeros(self.weight.shape)
        self.bias_grad = Tensor3D.zeros(self.bias.shape)
        
        var input_grad = Tensor3D.zeros(x.shape)
        
        for b in range(batch):
            for o in range(self.out_features):
                let g = grad[b, o, 0, 0, 0]
                self.bias_grad[o, 0, 0, 0, 0] += g
                
                for i in range(self.in_features):
                    self.weight_grad[o, i, 0, 0, 0] += g * x[b, i, 0, 0, 0]
                    input_grad[b, i, 0, 0, 0] += g * self.weight[o, i, 0, 0, 0]
        
        return input_grad


# ============ Dropout ============

struct Dropout3D:
    """3D Dropout layer for regularization."""
    var p: Scalar[FloatDType]  # Dropout probability
    var training: Bool
    var _mask: Tensor3D
    
    fn __init__(inout self, p: Float32 = 0.5):
        self.p = p
        self.training = True
        self._mask = Tensor3D.zeros(Shape5D(1, 1, 1, 1, 1))
    
    fn forward(inout self, x: Tensor3D) -> Tensor3D:
        """Apply dropout during training."""
        if not self.training:
            return x
        
        self._mask = Tensor3D.zeros(x.shape)
        var output = Tensor3D.zeros(x.shape)
        let scale = 1.0 / (1.0 - self.p)
        
        # Simple random dropout (using hash-based pseudo-random)
        var state: Int = 12345
        for i in range(x.numel()):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            let rand_val = Float32(state) / Float32(0x7FFFFFFF)
            
            if rand_val > self.p:
                self._mask.data.store(i, scale)
                output.data.store(i, x.data.load(i) * scale)
        
        return output
    
    fn backward(inout self, grad: Tensor3D) -> Tensor3D:
        """Backward pass - apply same mask."""
        return grad.mul(self._mask)
    
    fn train(inout self):
        self.training = True
    
    fn eval(inout self):
        self.training = False
