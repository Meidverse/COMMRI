"""
Training Loop Implementation
=============================
Complete training pipeline with validation, checkpointing, and early stopping.
"""

from models.tensor3d import Tensor3D, Shape5D, FloatDType
from models.network import VGG3D
from training.loss import cross_entropy_loss, cross_entropy_loss_backward, to_one_hot
from training.optimizers import Adam, SGD
from training.scheduler import CosineAnnealingLR, ReduceOnPlateau


@value
struct TrainingConfig:
    """Configuration for training."""
    var epochs: Int
    var batch_size: Int
    var learning_rate: Float32
    var weight_decay: Float32
    var early_stopping_patience: Int
    var checkpoint_dir: String
    var log_frequency: Int
    
    fn __init__(
        inout self,
        epochs: Int = 100,
        batch_size: Int = 4,
        learning_rate: Float32 = 0.001,
        weight_decay: Float32 = 0.0001,
        early_stopping_patience: Int = 15,
        checkpoint_dir: String = "./checkpoints",
        log_frequency: Int = 10
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = checkpoint_dir
        self.log_frequency = log_frequency


@value
struct TrainingMetrics:
    """Metrics tracked during training."""
    var train_loss: Float32
    var train_accuracy: Float32
    var val_loss: Float32
    var val_accuracy: Float32
    var epoch: Int
    var best_val_loss: Float32
    var epochs_without_improvement: Int


struct EarlyStopping:
    """Early stopping to prevent overfitting."""
    var patience: Int
    var min_delta: Float32
    var best_loss: Float32
    var counter: Int
    var should_stop: Bool
    
    fn __init__(inout self, patience: Int = 15, min_delta: Float32 = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = Float32.MAX
        self.counter = 0
        self.should_stop = False
    
    fn step(inout self, val_loss: Float32) -> Bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


struct Trainer:
    """
    Training orchestrator for MRI classification.
    
    Handles the full training loop including:
    - Forward/backward passes
    - Optimization steps
    - Validation
    - Checkpointing
    - Early stopping
    - Learning rate scheduling
    """
    var model: VGG3D
    var optimizer: Adam
    var scheduler: CosineAnnealingLR
    var early_stopping: EarlyStopping
    var config: TrainingConfig
    var metrics: TrainingMetrics
    
    fn __init__(
        inout self,
        model: VGG3D,
        config: TrainingConfig
    ):
        self.model = model
        self.config = config
        
        # Initialize optimizer
        self.optimizer = Adam(
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            base_lr=config.learning_rate,
            T_max=config.epochs,
            warmup_epochs=5
        )
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience
        )
        
        # Initialize metrics
        self.metrics = TrainingMetrics(
            train_loss=0.0,
            train_accuracy=0.0,
            val_loss=0.0,
            val_accuracy=0.0,
            epoch=0,
            best_val_loss=Float32.MAX,
            epochs_without_improvement=0
        )
    
    fn train_step(
        inout self,
        batch_x: Tensor3D,
        batch_y: Tensor3D
    ) -> Tuple[Float32, Float32]:
        """
        Single training step on a batch.
        
        Args:
            batch_x: Input batch (B, C, D, H, W)
            batch_y: Labels (B, num_classes, 1, 1, 1)
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Set model to training mode
        self.model.train()
        
        # Forward pass
        let predictions = self.model.forward(batch_x)
        
        # Compute loss
        let loss = cross_entropy_loss(predictions, batch_y)
        
        # Compute accuracy
        let accuracy = self._compute_accuracy(predictions, batch_y)
        
        # Backward pass
        let loss_grad = cross_entropy_loss_backward(predictions, batch_y)
        _ = self.model.backward(loss_grad)
        
        # Optimizer step
        let params = self.model.parameters()
        let grads = self.model.gradients()
        self.optimizer.step(params, grads)
        
        return (Float32(loss), accuracy)
    
    fn validate_step(
        inout self,
        batch_x: Tensor3D,
        batch_y: Tensor3D
    ) -> Tuple[Float32, Float32]:
        """
        Single validation step on a batch.
        
        Args:
            batch_x: Input batch
            batch_y: Labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Forward pass only (no gradients)
        let predictions = self.model.forward(batch_x)
        
        # Compute loss and accuracy
        let loss = cross_entropy_loss(predictions, batch_y)
        let accuracy = self._compute_accuracy(predictions, batch_y)
        
        return (Float32(loss), accuracy)
    
    fn train_epoch(
        inout self,
        train_batches: List[Tuple[Tensor3D, Tensor3D]]
    ) -> Tuple[Float32, Float32]:
        """
        Train for one epoch.
        
        Args:
            train_batches: List of (input, label) tuples
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        var total_loss: Float32 = 0.0
        var total_accuracy: Float32 = 0.0
        let num_batches = len(train_batches)
        
        for i in range(num_batches):
            let batch = train_batches[i]
            let batch_x = batch[0]
            let batch_y = batch[1]
            
            let result = self.train_step(batch_x, batch_y)
            total_loss += result[0]
            total_accuracy += result[1]
        
        return (total_loss / Float32(num_batches), total_accuracy / Float32(num_batches))
    
    fn validate_epoch(
        inout self,
        val_batches: List[Tuple[Tensor3D, Tensor3D]]
    ) -> Tuple[Float32, Float32]:
        """
        Validate for one epoch.
        
        Args:
            val_batches: List of (input, label) tuples
            
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        var total_loss: Float32 = 0.0
        var total_accuracy: Float32 = 0.0
        let num_batches = len(val_batches)
        
        for i in range(num_batches):
            let batch = val_batches[i]
            let batch_x = batch[0]
            let batch_y = batch[1]
            
            let result = self.validate_step(batch_x, batch_y)
            total_loss += result[0]
            total_accuracy += result[1]
        
        return (total_loss / Float32(num_batches), total_accuracy / Float32(num_batches))
    
    fn fit(
        inout self,
        train_batches: List[Tuple[Tensor3D, Tensor3D]],
        val_batches: List[Tuple[Tensor3D, Tensor3D]]
    ) -> TrainingMetrics:
        """
        Full training loop.
        
        Args:
            train_batches: Training data batches
            val_batches: Validation data batches
            
        Returns:
            Final training metrics
        """
        print("Starting training...")
        print("Model parameters:", self.model.num_parameters())
        
        for epoch in range(self.config.epochs):
            # Update learning rate
            let current_lr = self.scheduler.step(epoch)
            self.optimizer.lr = current_lr
            
            # Train epoch
            let train_result = self.train_epoch(train_batches)
            self.metrics.train_loss = train_result[0]
            self.metrics.train_accuracy = train_result[1]
            
            # Validate epoch
            let val_result = self.validate_epoch(val_batches)
            self.metrics.val_loss = val_result[0]
            self.metrics.val_accuracy = val_result[1]
            self.metrics.epoch = epoch
            
            # Check for improvement
            if self.metrics.val_loss < self.metrics.best_val_loss:
                self.metrics.best_val_loss = self.metrics.val_loss
                self.metrics.epochs_without_improvement = 0
                # TODO: Save best checkpoint
            else:
                self.metrics.epochs_without_improvement += 1
            
            # Log progress
            if epoch % self.config.log_frequency == 0 or epoch == self.config.epochs - 1:
                print("Epoch", epoch + 1, "/", self.config.epochs)
                print("  Train Loss:", self.metrics.train_loss, "Acc:", self.metrics.train_accuracy)
                print("  Val Loss:", self.metrics.val_loss, "Acc:", self.metrics.val_accuracy)
                print("  LR:", current_lr)
            
            # Early stopping check
            if self.early_stopping.step(self.metrics.val_loss):
                print("Early stopping triggered at epoch", epoch + 1)
                break
        
        print("Training complete!")
        return self.metrics
    
    fn _compute_accuracy(
        self,
        predictions: Tensor3D,
        targets: Tensor3D
    ) -> Float32:
        """Compute classification accuracy."""
        let batch = predictions.shape.batch
        let num_classes = predictions.shape.channels
        var correct = 0
        
        for b in range(batch):
            # Find predicted class (argmax)
            var max_pred: Float32 = predictions[b, 0, 0, 0, 0]
            var pred_class = 0
            for c in range(1, num_classes):
                let p = predictions[b, c, 0, 0, 0]
                if Float32(p) > max_pred:
                    max_pred = Float32(p)
                    pred_class = c
            
            # Find true class
            var true_class = 0
            for c in range(num_classes):
                if Float32(targets[b, c, 0, 0, 0]) > 0.5:
                    true_class = c
                    break
            
            if pred_class == true_class:
                correct += 1
        
        return Float32(correct) / Float32(batch)
