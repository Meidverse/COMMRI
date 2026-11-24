# MRI Brain Tumor Classification Models

This repository contains trained deep learning models for MRI brain scan classification, developed using Mojo and PyTorch.

## Models

### 1. Brain Tumor 2D CNN (`kaggle_tumor_2dcnn_best.pth`)

**Task:** Multi-class brain tumor classification from MRI slices

| Metric | Value |
|--------|-------|
| **Accuracy** | 93.95% |
| **Precision** | 0.94 |
| **Recall** | 0.94 |
| **F1 Score** | 0.94 |

**Classes:**
- `glioma` - 98.1% accuracy
- `meningioma` - 83.9% accuracy
- `notumor` - 98.5% accuracy
- `pituitary` - 94.3% accuracy

**Architecture:**
- 2D CNN with 4 convolutional blocks
- BatchNorm + Dropout regularization
- AdaptiveAvgPool + 3-layer classifier
- 4.85M parameters

**Training:**
- Dataset: Kaggle Brain Tumor MRI (7,023 images)
- Input: 224x224 RGB images
- Epochs: 50
- Optimizer: AdamW with OneCycleLR
- Augmentation: RandomCrop, Flip, Rotation, ColorJitter, GaussianBlur

### 2. IXI 3D Brain CNN (`ixi_3dcnn_best.pth`)

**Task:** 3D brain MRI volume classification

**Architecture:**
- 3D CNN with 4 Conv3D blocks
- BatchNorm3D + Dropout3D
- Global Average Pooling
- 1.2M parameters

**Training:**
- Dataset: IXI Brain MRI (681 NIfTI volumes)
- Input: 64x64x64 3D volumes
- Epochs: 30
- GPU: NVIDIA RTX 4090

## Usage

### Load Tumor Model (PyTorch)

```python
import torch
import torch.nn as nn

class TumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(TumorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))

# Load model
model = TumorCNN(num_classes=4)
model.load_state_dict(torch.load("kaggle_tumor_2dcnn_best.pth"))
model.eval()

# Inference
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open("brain_mri.jpg").convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    pred = output.argmax(1).item()

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
print(f"Prediction: {classes[pred]}")
```

## Training

```bash
# Clone repository
git clone https://huggingface.co/YOUR_USERNAME/mri-brain-classification

# Train tumor model
mojo run scripts/train_tumor.mojo

# Train 3D model
mojo run scripts/train_advanced.mojo
```

## Evaluation

```bash
# Evaluate tumor model
mojo run scripts/evaluate_tumor.mojo

# Evaluate IXI model
mojo run scripts/evaluate_ixi.mojo
```

## Citation

```bibtex
@misc{mri-brain-classification,
  author = {Meidverse},
  title = {MRI Brain Tumor Classification Models},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/Meidverse/mri-brain-classification}
}
```

## License

MIT License

## Acknowledgments

- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [IXI Dataset](https://brain-development.org/ixi-dataset/)
- Built with Mojo 🔥 and PyTorch
