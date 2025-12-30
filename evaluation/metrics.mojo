"""
Evaluation Metrics for Classification
======================================
Implements accuracy, precision, recall, F1, and confusion matrix.
"""

from models.tensor3d import Tensor3D, Shape5D, FloatDType
from math import sqrt


@value
struct ClassificationMetrics:
    """Container for classification metrics."""
    var accuracy: Float32
    var precision: Float32
    var recall: Float32
    var f1_score: Float32
    var num_samples: Int


struct ConfusionMatrix:
    """
    Confusion matrix for multi-class classification.
    """
    var matrix: Tensor3D  # (num_classes, num_classes, 1, 1, 1)
    var num_classes: Int
    
    fn __init__(inout self, num_classes: Int):
        self.num_classes = num_classes
        self.matrix = Tensor3D.zeros(Shape5D(num_classes, num_classes, 1, 1, 1))
    
    fn update(inout self, pred_class: Int, true_class: Int):
        """Add one prediction to the confusion matrix."""
        if pred_class >= 0 and pred_class < self.num_classes and true_class >= 0 and true_class < self.num_classes:
            let current = self.matrix[true_class, pred_class, 0, 0, 0]
            self.matrix[true_class, pred_class, 0, 0, 0] = current + 1.0
    
    fn get_accuracy(self) -> Float32:
        """Compute overall accuracy."""
        var correct: Float32 = 0.0
        var total: Float32 = 0.0
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                let count = self.matrix[i, j, 0, 0, 0]
                total += Float32(count)
                if i == j:
                    correct += Float32(count)
        
        if total > 0:
            return correct / total
        return 0.0
    
    fn get_precision(self, class_idx: Int) -> Float32:
        """Compute precision for a specific class."""
        var true_positives: Float32 = Float32(self.matrix[class_idx, class_idx, 0, 0, 0])
        var predicted_positives: Float32 = 0.0
        
        for i in range(self.num_classes):
            predicted_positives += Float32(self.matrix[i, class_idx, 0, 0, 0])
        
        if predicted_positives > 0:
            return true_positives / predicted_positives
        return 0.0
    
    fn get_recall(self, class_idx: Int) -> Float32:
        """Compute recall for a specific class."""
        var true_positives: Float32 = Float32(self.matrix[class_idx, class_idx, 0, 0, 0])
        var actual_positives: Float32 = 0.0
        
        for j in range(self.num_classes):
            actual_positives += Float32(self.matrix[class_idx, j, 0, 0, 0])
        
        if actual_positives > 0:
            return true_positives / actual_positives
        return 0.0
    
    fn get_f1(self, class_idx: Int) -> Float32:
        """Compute F1 score for a specific class."""
        let precision = self.get_precision(class_idx)
        let recall = self.get_recall(class_idx)
        
        if precision + recall > 0:
            return 2.0 * precision * recall / (precision + recall)
        return 0.0
    
    fn get_macro_precision(self) -> Float32:
        """Macro-averaged precision across all classes."""
        var total: Float32 = 0.0
        for i in range(self.num_classes):
            total += self.get_precision(i)
        return total / Float32(self.num_classes)
    
    fn get_macro_recall(self) -> Float32:
        """Macro-averaged recall across all classes."""
        var total: Float32 = 0.0
        for i in range(self.num_classes):
            total += self.get_recall(i)
        return total / Float32(self.num_classes)
    
    fn get_macro_f1(self) -> Float32:
        """Macro-averaged F1 score across all classes."""
        var total: Float32 = 0.0
        for i in range(self.num_classes):
            total += self.get_f1(i)
        return total / Float32(self.num_classes)
    
    fn reset(inout self):
        """Reset confusion matrix to zeros."""
        self.matrix = Tensor3D.zeros(Shape5D(self.num_classes, self.num_classes, 1, 1, 1))
    
    fn print_matrix(self):
        """Print the confusion matrix."""
        print("Confusion Matrix:")
        print("================")
        for i in range(self.num_classes):
            var row = "  "
            for j in range(self.num_classes):
                row += String(Int(self.matrix[i, j, 0, 0, 0])) + " "
            print(row)


struct MetricsTracker:
    """
    Track and compute evaluation metrics over batches.
    """
    var confusion: ConfusionMatrix
    var num_classes: Int
    var total_loss: Float32
    var num_batches: Int
    
    fn __init__(inout self, num_classes: Int):
        self.num_classes = num_classes
        self.confusion = ConfusionMatrix(num_classes)
        self.total_loss = 0.0
        self.num_batches = 0
    
    fn update(
        inout self,
        predictions: Tensor3D,
        targets: Tensor3D,
        loss: Float32
    ):
        """
        Update metrics with batch results.
        
        Args:
            predictions: Model predictions (batch, num_classes, 1, 1, 1)
            targets: One-hot encoded targets
            loss: Batch loss value
        """
        let batch = predictions.shape.batch
        
        self.total_loss += loss
        self.num_batches += 1
        
        for b in range(batch):
            # Find predicted class (argmax)
            var max_val = predictions[b, 0, 0, 0, 0]
            var pred_class = 0
            for c in range(1, self.num_classes):
                let val = predictions[b, c, 0, 0, 0]
                if Float32(val) > Float32(max_val):
                    max_val = val
                    pred_class = c
            
            # Find true class
            var true_class = 0
            for c in range(self.num_classes):
                if Float32(targets[b, c, 0, 0, 0]) > 0.5:
                    true_class = c
                    break
            
            self.confusion.update(pred_class, true_class)
    
    fn compute_metrics(self) -> ClassificationMetrics:
        """Compute all metrics from accumulated results."""
        return ClassificationMetrics(
            accuracy=self.confusion.get_accuracy(),
            precision=self.confusion.get_macro_precision(),
            recall=self.confusion.get_macro_recall(),
            f1_score=self.confusion.get_macro_f1(),
            num_samples=self._total_samples()
        )
    
    fn _total_samples(self) -> Int:
        """Count total samples processed."""
        var total: Int = 0
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                total += Int(self.confusion.matrix[i, j, 0, 0, 0])
        return total
    
    fn get_average_loss(self) -> Float32:
        """Get average loss across all batches."""
        if self.num_batches > 0:
            return self.total_loss / Float32(self.num_batches)
        return 0.0
    
    fn reset(inout self):
        """Reset all tracked metrics."""
        self.confusion.reset()
        self.total_loss = 0.0
        self.num_batches = 0
    
    fn print_report(self):
        """Print a summary report of all metrics."""
        let metrics = self.compute_metrics()
        
        print("\n" + "=" * 50)
        print("Classification Report")
        print("=" * 50)
        print("Total Samples:", metrics.num_samples)
        print("Average Loss:", self.get_average_loss())
        print("\nOverall Metrics:")
        print("  Accuracy:  ", metrics.accuracy)
        print("  Precision: ", metrics.precision)
        print("  Recall:    ", metrics.recall)
        print("  F1 Score:  ", metrics.f1_score)
        
        print("\nPer-Class Metrics:")
        for i in range(self.num_classes):
            print("  Class", i, ":")
            print("    Precision:", self.confusion.get_precision(i))
            print("    Recall:   ", self.confusion.get_recall(i))
            print("    F1:       ", self.confusion.get_f1(i))
        
        print("\n")
        self.confusion.print_matrix()
        print("=" * 50)


fn compute_accuracy(predictions: Tensor3D, targets: Tensor3D) -> Float32:
    """
    Compute classification accuracy.
    
    Args:
        predictions: Softmax probabilities (batch, num_classes, 1, 1, 1)
        targets: One-hot encoded targets
        
    Returns:
        Accuracy as fraction [0, 1]
    """
    let batch = predictions.shape.batch
    let num_classes = predictions.shape.channels
    var correct = 0
    
    for b in range(batch):
        # Find predicted class
        var max_val = predictions[b, 0, 0, 0, 0]
        var pred_class = 0
        for c in range(1, num_classes):
            let val = predictions[b, c, 0, 0, 0]
            if Float32(val) > Float32(max_val):
                max_val = val
                pred_class = c
        
        # Find true class
        for c in range(num_classes):
            if Float32(targets[b, c, 0, 0, 0]) > 0.5:
                if pred_class == c:
                    correct += 1
                break
    
    return Float32(correct) / Float32(batch)
