"""
Learning Rate Schedulers
=========================
Implements common learning rate scheduling strategies.
"""

from math import cos


alias PI: Float32 = 3.14159265358979


struct StepLR:
    """
    Step learning rate decay.
    Reduces LR by gamma every step_size epochs.
    """
    var base_lr: Float32
    var step_size: Int
    var gamma: Float32
    var current_lr: Float32
    
    fn __init__(inout self, base_lr: Float32, step_size: Int = 30, gamma: Float32 = 0.1):
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma
        self.current_lr = base_lr
    
    fn step(inout self, epoch: Int) -> Float32:
        """Get learning rate for given epoch."""
        let num_decays = epoch // self.step_size
        self.current_lr = self.base_lr * self._pow(self.gamma, num_decays)
        return self.current_lr
    
    fn _pow(self, base: Float32, exp: Int) -> Float32:
        var result: Float32 = 1.0
        for _ in range(exp):
            result *= base
        return result


struct CosineAnnealingLR:
    """
    Cosine annealing learning rate scheduler.
    LR follows a cosine curve from base_lr to min_lr over T_max epochs.
    """
    var base_lr: Float32
    var min_lr: Float32
    var T_max: Int
    var warmup_epochs: Int
    var current_lr: Float32
    
    fn __init__(
        inout self,
        base_lr: Float32,
        T_max: Int,
        min_lr: Float32 = 1e-6,
        warmup_epochs: Int = 0
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.current_lr = base_lr
    
    fn step(inout self, epoch: Int) -> Float32:
        """Get learning rate for given epoch."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            self.current_lr = self.base_lr * Float32(epoch + 1) / Float32(self.warmup_epochs)
        else:
            # Cosine annealing
            let adjusted_epoch = epoch - self.warmup_epochs
            let adjusted_T_max = self.T_max - self.warmup_epochs
            let cos_value = cos(PI * Float32(adjusted_epoch) / Float32(adjusted_T_max))
            self.current_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cos_value)
        
        return self.current_lr


struct ReduceOnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    """
    var factor: Float32
    var patience: Int
    var min_lr: Float32
    var current_lr: Float32
    var best_value: Float32
    var counter: Int
    var mode: String  # "min" or "max"
    
    fn __init__(
        inout self,
        base_lr: Float32,
        factor: Float32 = 0.1,
        patience: Int = 10,
        min_lr: Float32 = 1e-7,
        mode: String = "min"
    ):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.current_lr = base_lr
        self.best_value = Float32.MAX if mode == "min" else Float32.MIN
        self.counter = 0
        self.mode = mode
    
    fn step(inout self, metric: Float32) -> Float32:
        """
        Update scheduler with new metric value.
        
        Args:
            metric: Current metric value (e.g., validation loss)
            
        Returns:
            Current learning rate
        """
        var improved = False
        
        if self.mode == "min":
            if metric < self.best_value:
                self.best_value = metric
                self.counter = 0
                improved = True
        else:  # max
            if metric > self.best_value:
                self.best_value = metric
                self.counter = 0
                improved = True
        
        if not improved:
            self.counter += 1
            if self.counter >= self.patience:
                let new_lr = self.current_lr * self.factor
                if new_lr >= self.min_lr:
                    self.current_lr = new_lr
                else:
                    self.current_lr = self.min_lr
                self.counter = 0
        
        return self.current_lr


struct WarmupCosine:
    """
    Linear warmup followed by cosine annealing.
    Common schedule for transformer-style training.
    """
    var base_lr: Float32
    var warmup_steps: Int
    var total_steps: Int
    var min_lr: Float32
    var current_step: Int
    var current_lr: Float32
    
    fn __init__(
        inout self,
        base_lr: Float32,
        warmup_steps: Int,
        total_steps: Int,
        min_lr: Float32 = 0.0
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0
        self.current_lr = 0.0
    
    fn step(inout self) -> Float32:
        """Get learning rate for current step and increment."""
        self.current_step += 1
        
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            self.current_lr = self.base_lr * Float32(self.current_step) / Float32(self.warmup_steps)
        else:
            # Cosine decay
            let decay_steps = self.total_steps - self.warmup_steps
            let current_decay_step = self.current_step - self.warmup_steps
            let cos_value = cos(PI * Float32(current_decay_step) / Float32(decay_steps))
            self.current_lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + cos_value)
        
        return self.current_lr
    
    fn reset(inout self):
        """Reset step counter."""
        self.current_step = 0
