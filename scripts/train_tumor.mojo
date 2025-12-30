"""
Brain Tumor Classification - Kaggle Dataset Training
=====================================================
2D CNN for Kaggle Brain Tumor MRI Dataset with advanced preprocessing.

Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Classes: glioma, meningioma, notumor, pituitary

Usage:
    mojo run scripts/train_tumor.mojo
"""

from python import Python


fn main() raises:
    print("=" * 70)
    print("Brain Tumor Classification - 2D CNN with Advanced Preprocessing")
    print("=" * 70)
    
    var torch = Python.import_module("torch")
    var np = Python.import_module("numpy")
    var os = Python.import_module("os")
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
    var data_dir = "./data/raw"
    var num_epochs = 50
    var batch_size = 32
    var learning_rate = 0.0001
    var image_size = 224
    var num_classes = 4  # glioma, meningioma, notumor, pituitary
    
    print("\nConfiguration:")
    print("  Data dir:", data_dir)
    print("  Epochs:", num_epochs)
    print("  Batch size:", batch_size)
    print("  Learning rate:", learning_rate)
    print("  Image size:", image_size, "x", image_size)
    print("  Classes: glioma, meningioma, notumor, pituitary")
    print("  Device:", device_str)
    
    # Complete training code
    var training_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============== Advanced Preprocessing ==============
train_transform = transforms.Compose([
    transforms.Resize((image_size + 32, image_size + 32)),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============== Dataset Class ==============
class TumorDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels[idx]
            return image, label
        except Exception as e:
            # Return a blank image on error
            image = torch.zeros(3, image_size, image_size)
            return image, self.labels[idx]

# ============== 2D CNN Model ==============
class TumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(TumorCNN, self).__init__()
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============== Load Data ==============
def find_images(data_dir):
    '''Find all tumor images and extract labels from folder names.'''
    class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    image_paths = []
    labels = []
    
    # Search for images
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for img_path in glob.glob(os.path.join(data_dir, '**', ext), recursive=True):
            # Get class from folder name
            folder = os.path.basename(os.path.dirname(img_path)).lower()
            for class_name in class_names:
                if class_name in folder:
                    image_paths.append(img_path)
                    labels.append(class_to_idx[class_name])
                    break
    
    return image_paths, labels, class_names

print("\\nSearching for images...")
image_paths, labels, class_names = find_images(data_dir)
print(f"Found {len(image_paths)} images")

if len(image_paths) == 0:
    print("No images found! Using synthetic data...")
    # Generate synthetic for testing
    X_train = torch.randn(100, 3, image_size, image_size)
    y_train = torch.randint(0, num_classes, (100,))
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = train_dataset
else:
    # Print class distribution
    from collections import Counter
    dist = Counter(labels)
    print("Class distribution:")
    for cls_name in class_names:
        idx = class_names.index(cls_name)
        print(f"  {cls_name}: {dist.get(idx, 0)}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"\\nTrain: {len(X_train)}, Validation: {len(X_val)}")
    
    train_dataset = TumorDataset(X_train, y_train, train_transform)
    val_dataset = TumorDataset(X_val, y_val, val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ============== Initialize Model ==============
model = TumorCNN(num_classes=num_classes).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=learning_rate * 10, 
    epochs=num_epochs, steps_per_epoch=len(train_loader)
)

# ============== Training Functions ==============
def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), correct / total
"""
    
    # Execute setup
    var globals_dict = Python.dict()
    globals_dict["device"] = device
    globals_dict["data_dir"] = data_dir
    globals_dict["image_size"] = image_size
    globals_dict["batch_size"] = batch_size
    globals_dict["num_classes"] = num_classes
    globals_dict["learning_rate"] = learning_rate
    globals_dict["num_epochs"] = num_epochs
    
    _ = builtins.exec(training_code, globals_dict)
    
    var model = globals_dict["model"]
    var criterion = globals_dict["criterion"]
    var optimizer = globals_dict["optimizer"]
    var scheduler = globals_dict["scheduler"]
    var train_loader = globals_dict["train_loader"]
    var val_loader = globals_dict["val_loader"]
    var train_epoch_fn = globals_dict["train_epoch"]
    var validate_fn = globals_dict["validate"]
    
    print("\n" + "=" * 70)
    print("Starting Training with Advanced Preprocessing...")
    print("=" * 70 + "\n")
    
    var best_val_acc = 0.0
    var epoch_pbar = tqdm(builtins.range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Train
        var train_result = train_epoch_fn(model, train_loader, criterion, optimizer, scheduler, device)
        var train_loss = Float64(train_result[0])
        var train_acc = Float64(train_result[1])
        
        # Validate
        var val_result = validate_fn(model, val_loader, criterion, device)
        var val_loss = Float64(val_result[0])
        var val_acc = Float64(val_result[1])
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _ = globals_dict["torch"].save(model.state_dict(), "kaggle_tumor_2dcnn_best.pth")
        
        # Update progress
        _ = epoch_pbar.set_postfix(
            train_loss=train_loss, 
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            best=best_val_acc
        )
    
    # Save final model
    _ = globals_dict["torch"].save(model.state_dict(), "kaggle_tumor_2dcnn_final.pth")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("Best Validation Accuracy:", best_val_acc)
    print("Models saved:")
    print("  - kaggle_tumor_2dcnn_best.pth (best validation)")
    print("  - kaggle_tumor_2dcnn_final.pth (final epoch)")
    print("=" * 70)
