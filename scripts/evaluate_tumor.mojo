"""
Kaggle Brain Tumor 2D CNN Evaluation Script
============================================
Evaluates the trained tumor classification model with comprehensive metrics.

Usage:
    mojo run scripts/evaluate_tumor.mojo
"""

from python import Python


fn main() raises:
    print("=" * 70)
    print("Kaggle Brain Tumor 2D CNN - Model Evaluation")
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
    var model_path = "kaggle_tumor_2dcnn_best.pth"
    var data_dir = "./data/raw"
    var image_size = 224
    var num_classes = 4
    var batch_size = 32
    
    print("\nConfiguration:")
    print("  Model:", model_path)
    print("  Data dir:", data_dir)
    print("  Image size:", image_size, "x", image_size)
    print("  Classes: glioma, meningioma, notumor, pituitary")
    print("  Device:", device_str)
    
    # Complete evaluation code
    var eval_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============== 2D CNN Model ==============
class TumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(TumorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============== Dataset ==============
class TumorDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except:
            return torch.zeros(3, image_size, image_size), self.labels[idx]

# Transform
eval_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============== Load Model ==============
print("\\nLoading model...")
model = TumorCNN(num_classes=num_classes)

# Try different model paths
model_paths = [model_path, 'best_tumor_model.pth', 'kaggle_tumor_2dcnn_final.pth', 'tumor_model_final.pth']
loaded = False
for mp in model_paths:
    if os.path.exists(mp):
        model.load_state_dict(torch.load(mp, map_location=device))
        print(f"✓ Loaded: {mp}")
        loaded = True
        break

if not loaded:
    print(f"✗ No model found! Looked for: {model_paths}")
    print("Please train first: mojo run scripts/train_tumor.mojo")
    exit(1)

model = model.to(device)
model.eval()

# ============== Load Data ==============
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

print("\\nSearching for images...")
image_paths = []
labels = []

for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    for img_path in glob.glob(os.path.join(data_dir, '**', ext), recursive=True):
        folder = os.path.basename(os.path.dirname(img_path)).lower()
        for class_name in class_names:
            if class_name in folder:
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])
                break

print(f"Found {len(image_paths)} images")

if len(image_paths) == 0:
    print("No images found!")
    exit(1)

# Use validation split for evaluation
_, X_val, _, y_val = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)
print(f"Evaluating on {len(X_val)} validation images")

val_dataset = TumorDataset(X_val, y_val, eval_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# ============== Run Inference ==============
print("\\nRunning inference...")
all_preds = []
all_probs = []
all_labels = []

with torch.no_grad():
    for images, labels_batch in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels_batch.numpy())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_probs = np.array(all_probs)

# ============== Calculate Metrics ==============
print("\\n" + "=" * 70)
print("EVALUATION RESULTS - Kaggle Tumor 2D CNN")
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

# Per-class metrics
print(f"\\nPer-Class Accuracy:")
for i, cls in enumerate(class_names):
    mask = y_true == i
    if mask.sum() > 0:
        cls_acc = (y_pred[mask] == y_true[mask]).mean()
        print(f"  {cls:12}: {cls_acc:.4f} ({cls_acc*100:.1f}%)")

# Classification Report
print(f"\\nClassification Report:")
print("-" * 60)
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# ============== Save Visualizations ==============
os.makedirs('outputs/evaluation', exist_ok=True)

# Confusion Matrix Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title(f'Kaggle Tumor 2D CNN - Confusion Matrix\\nAccuracy: {accuracy*100:.2f}%', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/evaluation/kaggle_tumor_2dcnn_confusion_matrix.png', dpi=150)
plt.close()
print("\\n✓ Saved: outputs/evaluation/kaggle_tumor_2dcnn_confusion_matrix.png")

# Per-class accuracy bar chart
plt.figure(figsize=(10, 6))
class_accs = []
for i in range(num_classes):
    mask = y_true == i
    if mask.sum() > 0:
        class_accs.append((y_pred[mask] == y_true[mask]).mean() * 100)
    else:
        class_accs.append(0)

colors = ['#2ecc71' if acc > 90 else '#f39c12' if acc > 80 else '#e74c3c' for acc in class_accs]
bars = plt.bar(class_names, class_accs, color=colors, edgecolor='black')
plt.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
plt.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='80% threshold')
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Tumor Type', fontsize=12)
plt.title('Per-Class Classification Accuracy', fontsize=14)
plt.ylim(0, 105)
for bar, acc in zip(bars, class_accs):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/evaluation/kaggle_tumor_2dcnn_class_accuracy.png', dpi=150)
plt.close()
print("✓ Saved: outputs/evaluation/kaggle_tumor_2dcnn_class_accuracy.png")

# Confidence distribution
plt.figure(figsize=(10, 6))
correct_mask = y_pred == y_true
correct_conf = y_probs.max(axis=1)[correct_mask] * 100
wrong_conf = y_probs.max(axis=1)[~correct_mask] * 100

plt.hist(correct_conf, bins=20, alpha=0.7, label=f'Correct ({len(correct_conf)})', color='green')
plt.hist(wrong_conf, bins=20, alpha=0.7, label=f'Wrong ({len(wrong_conf)})', color='red')
plt.xlabel('Confidence (%)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Prediction Confidence Distribution', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('outputs/evaluation/kaggle_tumor_2dcnn_confidence.png', dpi=150)
plt.close()
print("✓ Saved: outputs/evaluation/kaggle_tumor_2dcnn_confidence.png")

print("\\n" + "=" * 70)
print(f"Evaluation complete!")
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"Results saved to: outputs/evaluation/")
print("=" * 70)
"""
    
    # Execute evaluation
    var globals_dict = Python.dict()
    globals_dict["device"] = device
    globals_dict["model_path"] = model_path
    globals_dict["data_dir"] = data_dir
    globals_dict["image_size"] = image_size
    globals_dict["num_classes"] = num_classes
    globals_dict["batch_size"] = batch_size
    
    _ = builtins.exec(eval_code, globals_dict)
    
    print("\nEvaluation script completed!")
