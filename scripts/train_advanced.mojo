"""
MRI Classification - Advanced Training Script
==============================================
Uses PyTorch for 3D CNN with GPU acceleration.
Loads real NIfTI files from IXI dataset.

Usage:
    mojo run scripts/train_advanced.mojo
"""

from python import Python


fn main() raises:
    print("=" * 70)
    print("MRI Classification System - Advanced 3D CNN Training")
    print("=" * 70)
    
    # Import Python modules
    var torch = Python.import_module("torch")
    var torch_nn = Python.import_module("torch.nn")
    var torch_optim = Python.import_module("torch.optim")
    var np = Python.import_module("numpy")
    var nibabel = Python.import_module("nibabel")
    var os = Python.import_module("os")
    var glob = Python.import_module("glob")
    var tqdm_module = Python.import_module("tqdm")
    var tqdm = tqdm_module.tqdm
    var builtins = Python.import_module("builtins")
    
    # Check GPU availability
    var device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
        print("\n🚀 GPU Available:", torch.cuda.get_device_name(0))
    else:
        print("\n⚠️  GPU not available, using CPU")
    var device = torch.device(device_str)
    
    # Configuration
    var data_dir = "./data/raw"
    var num_epochs = 30
    var batch_size = 2
    var learning_rate = 0.0001
    var input_size = 64  # Resize volumes to 64x64x64
    var num_classes = 2
    
    print("\nConfiguration:")
    print("  Data dir:", data_dir)
    print("  Epochs:", num_epochs)
    print("  Batch size:", batch_size)
    print("  Learning rate:", learning_rate)
    print("  Input size:", input_size, "^3")
    print("  Device:", device_str)
    
    # Define and create model using exec
    var setup_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
from tqdm import tqdm
import glob
import os

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
        print(f"Error loading {file_path}: {e}")
        return None

def create_data_loader(data_dir, target_size, batch_size, num_classes=2):
    files = glob.glob(os.path.join(data_dir, '**/*.nii.gz'), recursive=True)
    files += glob.glob(os.path.join(data_dir, '**/*.nii'), recursive=True)
    
    X_list = []
    y_list = []
    
    print(f"Found {len(files)} NIfTI files")
    for f in tqdm(files[:50], desc="Loading data"):  # Limit to 50 for memory
        data = load_and_preprocess(f, target_size)
        if data is not None:
            X_list.append(data)
            label = 0 if 'healthy' in f.lower() or 'ixi' in f.lower() else 1
            y_list.append(label)
    
    if len(X_list) == 0:
        print("No valid files, generating synthetic data...")
        X = torch.randn(20, 1, target_size, target_size, target_size)
        y = torch.randint(0, num_classes, (20,))
    else:
        X = torch.tensor(np.stack(X_list)).unsqueeze(1)
        y = torch.tensor(y_list)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    return total_loss / len(loader), correct / total

# Initialize
model = CNN3D(num_classes)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

loader = create_data_loader(data_dir, target_size, batch_size, num_classes)
print(f"Data loader ready: {len(loader)} batches")
"""
    
    # Execute setup
    var globals_dict = Python.dict()
    globals_dict["device"] = device
    globals_dict["data_dir"] = data_dir
    globals_dict["target_size"] = input_size
    globals_dict["batch_size"] = batch_size
    globals_dict["num_classes"] = num_classes
    globals_dict["learning_rate"] = learning_rate
    globals_dict["num_epochs"] = num_epochs
    
    _ = builtins.exec(setup_code, globals_dict)
    
    var model = globals_dict["model"]
    var criterion = globals_dict["criterion"]
    var optimizer = globals_dict["optimizer"]
    var scheduler = globals_dict["scheduler"]
    var loader = globals_dict["loader"]
    var train_epoch_fn = globals_dict["train_epoch"]
    
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")
    
    var best_acc = 0.0
    var epoch_pbar = tqdm(builtins.range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        # Train one epoch
        var result = train_epoch_fn(model, loader, criterion, optimizer, device)
        var loss = Float64(result[0])
        var acc = Float64(result[1])
        
        # Update scheduler
        _ = scheduler.step()
        var current_lr = Float64(optimizer.param_groups[0]["lr"])
        
        # Track best
        if acc > best_acc:
            best_acc = acc
            _ = globals_dict["torch"].save(model.state_dict(), "best_model.pth")
        
        # Update progress bar
        _ = epoch_pbar.set_postfix(loss=loss, acc=acc, best=best_acc, lr=current_lr)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("Best Accuracy:", best_acc)
    print("Model saved to: best_model.pth")
    print("=" * 70)
