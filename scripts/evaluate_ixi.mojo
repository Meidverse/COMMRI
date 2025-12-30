"""
IXI 3D CNN Model Evaluation Script
===================================
Evaluates the trained IXI 3D CNN model with comprehensive metrics.

Usage:
    mojo run scripts/evaluate_ixi.mojo
"""

from python import Python


fn main() raises:
    print("=" * 70)
    print("IXI 3D CNN Model Evaluation")
    print("=" * 70)
    
    var torch = Python.import_module("torch")
    var np = Python.import_module("numpy")
    var os = Python.import_module("os")
    var glob = Python.import_module("glob")
    var tqdm_module = Python.import_module("tqdm")
    var tqdm = tqdm_module.tqdm
    var builtins = Python.import_module("builtins")
    
    # Check GPU
    var device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
        print("\n🚀 GPU:", torch.cuda.get_device_name(0))
    else:
        print("\n⚠️  Using CPU")
    var device = torch.device(device_str)
    
    # Configuration
    var model_path = "ixi_3dcnn_best.pth"
    var data_dir = "./data/raw"
    var input_size = 64
    var num_classes = 2
    var batch_size = 4
    
    print("\nConfiguration:")
    print("  Model:", model_path)
    print("  Data dir:", data_dir)
    print("  Input size:", input_size, "^3")
    print("  Num classes:", num_classes)
    print("  Device:", device_str)
    
    # Complete evaluation code
    var eval_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
from tqdm import tqdm
import glob
import os
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============== 3D CNN Model ==============
class CNN3D(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(256)
        
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(0.3)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ============== Load and Preprocess ==============
def load_and_preprocess(file_path, target_size=64):
    try:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)
        if len(data.shape) == 4:
            data = data[:,:,:,0]
        zoom_factors = [target_size / s for s in data.shape]
        data = zoom(data, zoom_factors, order=1)
        data = (data - data.mean()) / (data.std() + 1e-8)
        return data
    except Exception as e:
        return None

# ============== Load Model ==============
print("\\nLoading model...")
model = CNN3D(num_classes=num_classes)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"✓ Loaded: {model_path}")
else:
    print(f"✗ Model not found: {model_path}")
    print("Please train the model first with: mojo run scripts/train_advanced.mojo")
    exit(1)

model = model.to(device)
model.eval()

# ============== Load Test Data ==============
print("\\nLoading test data...")
files = glob.glob(os.path.join(data_dir, '**/*.nii.gz'), recursive=True)
files += glob.glob(os.path.join(data_dir, '**/*.nii'), recursive=True)
print(f"Found {len(files)} NIfTI files")

X_list = []
y_list = []
file_names = []

for f in tqdm(files[:100], desc="Loading"):  # Limit for evaluation
    data = load_and_preprocess(f, input_size)
    if data is not None:
        X_list.append(data)
        label = 0 if 'healthy' in f.lower() or 'ixi' in f.lower() else 1
        y_list.append(label)
        file_names.append(os.path.basename(f))

if len(X_list) == 0:
    print("No valid data found!")
    exit(1)

X = torch.tensor(np.stack(X_list)).unsqueeze(1).float()
y_true = np.array(y_list)
print(f"Loaded {len(X)} samples")

# ============== Run Inference ==============
print("\\nRunning inference...")
all_preds = []
all_probs = []

with torch.no_grad():
    for i in tqdm(range(0, len(X), batch_size), desc="Evaluating"):
        batch = X[i:i+batch_size].to(device)
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

y_pred = np.array(all_preds)
y_probs = np.array(all_probs)

# ============== Calculate Metrics ==============
print("\\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"\\nOverall Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")

# Class names
class_names = ['Healthy/IXI', 'Diseased']

# Classification Report
print(f"\\nClassification Report:")
print("-" * 50)
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Confusion Matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# ============== Save Visualizations ==============
os.makedirs('outputs/evaluation', exist_ok=True)

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('IXI 3D CNN - Confusion Matrix')
plt.tight_layout()
plt.savefig('outputs/evaluation/ixi_3dcnn_confusion_matrix.png', dpi=150)
plt.close()
print("\\n✓ Saved: outputs/evaluation/ixi_3dcnn_confusion_matrix.png")

# ROC Curve (if binary)
if num_classes == 2 and len(np.unique(y_true)) > 1:
    try:
        auc = roc_auc_score(y_true, y_probs[:, 1])
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('IXI 3D CNN - ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/evaluation/ixi_3dcnn_roc_curve.png', dpi=150)
        plt.close()
        print(f"✓ Saved: outputs/evaluation/ixi_3dcnn_roc_curve.png")
        print(f"  AUC Score: {auc:.4f}")
    except Exception as e:
        print(f"Could not compute ROC: {e}")

# Per-sample predictions
print("\\nSample Predictions (first 10):")
print("-" * 60)
print(f"{'File':<40} {'True':>8} {'Pred':>8} {'Conf':>8}")
print("-" * 60)
for i in range(min(10, len(file_names))):
    conf = y_probs[i].max() * 100
    true_label = class_names[y_true[i]]
    pred_label = class_names[y_pred[i]]
    correct = "✓" if y_true[i] == y_pred[i] else "✗"
    print(f"{file_names[i][:38]:<40} {true_label:>8} {pred_label:>8} {conf:>6.1f}% {correct}")

print("\\n" + "=" * 70)
print(f"Evaluation complete. Results saved to outputs/evaluation/")
print("=" * 70)
"""
    
    # Execute evaluation
    var globals_dict = Python.dict()
    globals_dict["device"] = device
    globals_dict["model_path"] = model_path
    globals_dict["data_dir"] = data_dir
    globals_dict["input_size"] = input_size
    globals_dict["num_classes"] = num_classes
    globals_dict["batch_size"] = batch_size
    
    _ = builtins.exec(eval_code, globals_dict)
    
    print("\nEvaluation script completed!")
