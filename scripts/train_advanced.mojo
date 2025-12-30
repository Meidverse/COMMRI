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
    var scipy_ndimage = Python.import_module("scipy.ndimage")
    
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
    
    # Define 3D CNN model in Python
    var model_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN3D(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN3D, self).__init__()
        # Encoder
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
        
        # Global Average Pooling + Classifier
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # Block 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout(x)
        
        # Classifier
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN3D(num_classes=2)
"""
    
    # Execute model definition
    var exec_globals = Python.dict()
    _ = builtins.exec(model_code, exec_globals)
    var model = exec_globals["model"]
    _ = model.to(device)
    
    # Count parameters
    var total_params = builtins.sum(p.numel() for p in model.parameters())
    print("  Model parameters:", total_params)
    
    # Loss and optimizer
    var criterion = torch_nn.CrossEntropyLoss()
    var optimizer = torch_optim.Adam(model.parameters(), lr=learning_rate)
    var scheduler = torch_optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Find NIfTI files
    print("\nSearching for NIfTI files...")
    var nii_files = glob.glob(data_dir + "/**/*.nii.gz", recursive=True)
    var nii_list = builtins.list(nii_files)
    var num_files = builtins.len(nii_list)
    print("  Found", num_files, "NIfTI files")
    
    if num_files == 0:
        print("\n⚠️  No NIfTI files found. Generating synthetic data...")
        # Generate synthetic data if no real data
        var num_synthetic = 20
        var X_data = torch.randn(num_synthetic, 1, input_size, input_size, input_size)
        var y_data = torch.randint(0, num_classes, builtins.tuple([num_synthetic]))
    else:
        print("  Loading and preprocessing NIfTI files...")
    
    # Training function
    var train_code = """
import torch
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
from tqdm import tqdm
import glob
import os

def load_and_preprocess(file_path, target_size=64):
    '''Load NIfTI and resize to target size.'''
    try:
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.float32)
        
        # Take middle slice if 4D
        if len(data.shape) == 4:
            data = data[:,:,:,0]
        
        # Resize to target
        zoom_factors = [target_size / s for s in data.shape]
        data = zoom(data, zoom_factors, order=1)
        
        # Normalize
        data = (data - data.mean()) / (data.std() + 1e-8)
        
        return data
    except:
        return None

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in data_loader:
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
    
    return total_loss / len(data_loader), correct / total

def create_data_loader(data_dir, target_size, batch_size, num_classes=2):
    '''Create data loader from NIfTI files.'''
    files = glob.glob(os.path.join(data_dir, '**/*.nii.gz'), recursive=True)
    files += glob.glob(os.path.join(data_dir, '**/*.nii'), recursive=True)
    
    X_list = []
    y_list = []
    
    print(f"Loading {len(files)} files...")
    for i, f in enumerate(tqdm(files[:100], desc="Loading data")):  # Limit to 100 for memory
        data = load_and_preprocess(f, target_size)
        if data is not None:
            X_list.append(data)
            # Assign label based on directory name (healthy=0, other=1)
            label = 0 if 'healthy' in f.lower() or 'ixi' in f.lower() else 1
            y_list.append(label)
    
    if len(X_list) == 0:
        # Generate synthetic if no valid files
        print("Generating synthetic data...")
        X = torch.randn(20, 1, target_size, target_size, target_size)
        y = torch.randint(0, num_classes, (20,))
    else:
        X = torch.tensor(np.stack(X_list)).unsqueeze(1)  # Add channel dim
        y = torch.tensor(y_list)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Create loader
loader = create_data_loader(data_dir, target_size, batch_size, num_classes)
print(f"Data loader ready: {len(loader)} batches")
"""
    
    # Execute training setup
    var train_globals = Python.dict()
    train_globals["model"] = model
    train_globals["criterion"] = criterion
    train_globals["optimizer"] = optimizer
    train_globals["device"] = device
    train_globals["data_dir"] = data_dir
    train_globals["target_size"] = input_size
    train_globals["batch_size"] = batch_size
    train_globals["num_classes"] = num_classes
    
    _ = builtins.exec(train_code, train_globals)
    var loader = train_globals["loader"]
    var train_epoch_fn = train_globals["train_epoch"]
    
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")
    
    var best_acc = 0.0
    var epoch_pbar = tqdm(builtins.range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        var epoch_int = Int(epoch)
        
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
            # Save best model
            _ = torch.save(model.state_dict(), "best_model.pth")
        
        # Update progress bar
        _ = epoch_pbar.set_postfix(loss=loss, acc=acc, best=best_acc, lr=current_lr)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("Best Accuracy:", best_acc)
    print("Model saved to: best_model.pth")
    print("=" * 70)
