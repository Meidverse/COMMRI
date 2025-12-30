"""
Optimizers for Neural Network Training
=======================================
Implements SGD (with momentum) and Adam optimizer.
"""

from models.tensor3d import Tensor3D, Shape5D, FloatDType
from memory import memset_zero
from math import sqrt


struct SGD:
    """
    Stochastic Gradient Descent with momentum.
    
    v_t = momentum * v_{t-1} + lr * grad
    param = param - v_t
    """
    var lr: Scalar[FloatDType]
    var momentum: Scalar[FloatDType]
    var weight_decay: Scalar[FloatDType]
    var velocities: List[Tensor3D]
    var initialized: Bool
    
    fn __init__(
        inout self,
        lr: Float32 = 0.001,
        momentum: Float32 = 0.9,
        weight_decay: Float32 = 0.0
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = List[Tensor3D]()
        self.initialized = False
    
    fn step(inout self, params: List[Tensor3D], grads: List[Tensor3D]):
        """
        Perform one optimization step.
        
        Args:
            params: List of parameter tensors
            grads: List of gradient tensors (same order as params)
        """
        # Initialize velocities on first call
        if not self.initialized:
            for i in range(len(params)):
                self.velocities.append(Tensor3D.zeros(params[i].shape))
            self.initialized = True
        
        # Update each parameter
        for i in range(len(params)):
            let param = params[i]
            let grad = grads[i]
            
            # Apply weight decay (L2 regularization)
            var effective_grad = grad
            if self.weight_decay > 0:
                for j in range(param.numel()):
                    let g = grad.data.load(j)
                    let p = param.data.load(j)
                    effective_grad.data.store(j, g + self.weight_decay * p)
            
            # Update velocity
            for j in range(param.numel()):
                let v = self.velocities[i].data.load(j)
                let g = effective_grad.data.load(j)
                let new_v = self.momentum * v + self.lr * g
                self.velocities[i].data.store(j, new_v)
                
                # Update parameter
                let p = param.data.load(j)
                param.data.store(j, p - new_v)
    
    fn zero_grad(inout self):
        """Reset velocities to zero."""
        for i in range(len(self.velocities)):
            for j in range(self.velocities[i].numel()):
                self.velocities[i].data.store(j, 0.0)


struct Adam:
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(v_hat) + eps)
    """
    var lr: Scalar[FloatDType]
    var beta1: Scalar[FloatDType]
    var beta2: Scalar[FloatDType]
    var eps: Scalar[FloatDType]
    var weight_decay: Scalar[FloatDType]
    
    var m: List[Tensor3D]  # First moment (mean of gradients)
    var v: List[Tensor3D]  # Second moment (variance of gradients)
    var t: Int             # Timestep
    var initialized: Bool
    
    fn __init__(
        inout self,
        lr: Float32 = 0.001,
        beta1: Float32 = 0.9,
        beta2: Float32 = 0.999,
        eps: Float32 = 1e-8,
        weight_decay: Float32 = 0.0
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = List[Tensor3D]()
        self.v = List[Tensor3D]()
        self.t = 0
        self.initialized = False
    
    fn step(inout self, params: List[Tensor3D], grads: List[Tensor3D]):
        """
        Perform one optimization step.
        """
        self.t += 1
        
        # Initialize on first call
        if not self.initialized:
            for i in range(len(params)):
                self.m.append(Tensor3D.zeros(params[i].shape))
                self.v.append(Tensor3D.zeros(params[i].shape))
            self.initialized = True
        
        # Bias correction terms
        let beta1_t = self._pow(self.beta1, self.t)
        let beta2_t = self._pow(self.beta2, self.t)
        let bias_correction1 = 1.0 - beta1_t
        let bias_correction2 = 1.0 - beta2_t
        
        # Update each parameter
        for i in range(len(params)):
            let param = params[i]
            let grad = grads[i]
            
            for j in range(param.numel()):
                var g = grad.data.load(j)
                let p = param.data.load(j)
                
                # Apply weight decay (AdamW style)
                if self.weight_decay > 0:
                    g = g + self.weight_decay * p
                
                # Update biased first moment
                let m_old = self.m[i].data.load(j)
                let m_new = self.beta1 * m_old + (1.0 - self.beta1) * g
                self.m[i].data.store(j, m_new)
                
                # Update biased second moment
                let v_old = self.v[i].data.load(j)
                let v_new = self.beta2 * v_old + (1.0 - self.beta2) * g * g
                self.v[i].data.store(j, v_new)
                
                # Bias-corrected estimates
                let m_hat = m_new / bias_correction1
                let v_hat = v_new / bias_correction2
                
                # Update parameter
                let update = self.lr * m_hat / (sqrt(v_hat) + self.eps)
                param.data.store(j, p - update)
    
    fn _pow(self, base: Scalar[FloatDType], exp: Int) -> Scalar[FloatDType]:
        """Compute base^exp."""
        var result: Scalar[FloatDType] = 1.0
        for _ in range(exp):
            result *= base
        return result
    
    fn zero_grad(inout self):
        """Reset moments (typically not needed for Adam)."""
        pass


struct AdamW:
    """
    AdamW optimizer (Adam with decoupled weight decay).
    
    Applies weight decay directly to parameters, not to gradients.
    """
    var lr: Scalar[FloatDType]
    var beta1: Scalar[FloatDType]
    var beta2: Scalar[FloatDType]
    var eps: Scalar[FloatDType]
    var weight_decay: Scalar[FloatDType]
    
    var m: List[Tensor3D]
    var v: List[Tensor3D]
    var t: Int
    var initialized: Bool
    
    fn __init__(
        inout self,
        lr: Float32 = 0.001,
        beta1: Float32 = 0.9,
        beta2: Float32 = 0.999,
        eps: Float32 = 1e-8,
        weight_decay: Float32 = 0.01
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = List[Tensor3D]()
        self.v = List[Tensor3D]()
        self.t = 0
        self.initialized = False
    
    fn step(inout self, params: List[Tensor3D], grads: List[Tensor3D]):
        """Perform AdamW optimization step."""
        self.t += 1
        
        if not self.initialized:
            for i in range(len(params)):
                self.m.append(Tensor3D.zeros(params[i].shape))
                self.v.append(Tensor3D.zeros(params[i].shape))
            self.initialized = True
        
        let beta1_t = self._pow(self.beta1, self.t)
        let beta2_t = self._pow(self.beta2, self.t)
        let bias_correction1 = 1.0 - beta1_t
        let bias_correction2 = 1.0 - beta2_t
        
        for i in range(len(params)):
            let param = params[i]
            let grad = grads[i]
            
            for j in range(param.numel()):
                let g = grad.data.load(j)
                var p = param.data.load(j)
                
                # Decoupled weight decay
                p = p - self.lr * self.weight_decay * p
                
                # Update moments
                let m_old = self.m[i].data.load(j)
                let m_new = self.beta1 * m_old + (1.0 - self.beta1) * g
                self.m[i].data.store(j, m_new)
                
                let v_old = self.v[i].data.load(j)
                let v_new = self.beta2 * v_old + (1.0 - self.beta2) * g * g
                self.v[i].data.store(j, v_new)
                
                # Bias-corrected estimates
                let m_hat = m_new / bias_correction1
                let v_hat = v_new / bias_correction2
                
                # Update parameter
                p = p - self.lr * m_hat / (sqrt(v_hat) + self.eps)
                param.data.store(j, p)
    
    fn _pow(self, base: Scalar[FloatDType], exp: Int) -> Scalar[FloatDType]:
        var result: Scalar[FloatDType] = 1.0
        for _ in range(exp):
            result *= base
        return result
