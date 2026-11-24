"""AI models, preprocessing pipeline and Grad-CAM for NeuroScan AI."""
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models as tvm

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
INPUT_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))


class TumorCNN(nn.Module):
    """VGG-style 2D CNN (4.85M params) trained on the Kaggle Brain Tumor MRI dataset."""

    def __init__(self, num_classes=4):
        super().__init__()

        def block(i, o):
            return [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        self.features = nn.Sequential(
            *block(3, 64), *block(64, 64), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            *block(64, 128), *block(128, 128), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            *block(128, 256), *block(256, 256), nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
            *block(256, 512), *block(512, 512), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _no_inplace(m):
    for mod in m.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False
    return m


MODALITIES = {
    "mri": {
        "name": "Brain MRI",
        "task": "Brain tumour classification",
        "weights": os.path.join(ROOT, "kaggle_tumor_2dcnn_best.pth"),
        "arch": "TumorCNN — 8-layer VGG-style CNN",
        "params": "4.85 M",
        "classes": ["glioma", "meningioma", "notumor", "pituitary"],
        "labels": {
            "glioma": ("Glioma Tumour", True, "Tumour arising from glial cells of the brain or spine, most often in the cerebral hemispheres."),
            "meningioma": ("Meningioma Tumour", True, "Tumour of the meninges, the membranes covering the brain. Usually slow-growing and frequently benign."),
            "notumor": ("No Tumour Detected", False, "No tumour pattern was identified in this slice. Brain tissue appears within normal limits to the model."),
            "pituitary": ("Pituitary Tumour", True, "Tumour of the pituitary gland at the base of the brain; may affect hormone regulation."),
        },
    },
    "xray": {
        "name": "Bone X-ray",
        "task": "Bone fracture detection",
        "weights": os.path.join(ROOT, "xray_fracture_mnv3.pth"),
        "arch": "MobileNetV3-Large (ImageNet pre-trained, fine-tuned)",
        "params": "4.2 M",
        "classes": ["fractured", "not fractured"],
        "labels": {
            "fractured": ("Fracture Detected", True, "Image pattern consistent with a bone fracture. Correlate clinically and review the highlighted region."),
            "not fractured": ("No Fracture Detected", False, "No fracture pattern was identified by the model in this radiograph."),
        },
    },
}

_models = {}


def available(modality):
    return os.path.exists(MODALITIES[modality]["weights"])


def get_model(modality):
    if modality in _models:
        return _models[modality]
    cfg = MODALITIES[modality]
    if modality == "mri":
        m = TumorCNN(len(cfg["classes"]))
        target = lambda mm: mm.features[29]
    else:
        m = tvm.mobilenet_v3_large(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(cfg["classes"]))
        target = lambda mm: mm.features[-1]
    m.load_state_dict(torch.load(cfg["weights"], map_location="cpu"))
    m.eval()
    _no_inplace(m)
    _models[modality] = (m, target(m))
    return _models[modality]


def metrics(modality):
    f = os.path.join(APP_DIR, "model_meta", f"{modality}_metrics.json")
    if os.path.exists(f):
        with open(f) as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------- validation
def validate(img: Image.Image):
    """Return (ok, checks). Checks are shown on the processing page."""
    w, h = img.size
    rgb = np.asarray(img.convert("RGB")).astype(np.float32)
    sat = float(np.mean(rgb.max(2) - rgb.min(2)) / 255.0)
    checks = [
        ("Readable image file", True, f"{img.format or 'image'} · mode {img.mode}"),
        ("Minimum resolution 64 × 64", min(w, h) >= 64, f"{w} × {h} px"),
        ("Greyscale-dominant (medical scan)", sat < 0.12, f"mean colour saturation {sat:.3f}"),
    ]
    return all(c[1] for c in checks), checks


# ---------------------------------------------------------------- preprocessing
def preprocess(img: Image.Image):
    """Exactly the model's input pipeline. Returns tensor + intermediate arrays for display."""
    rgb = img.convert("RGB")
    resized = rgb.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0             # [0,1]
    norm = (arr - MEAN) / STD                                          # ImageNet normalisation
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, np.asarray(resized), norm


def gradcam(modality, tensor):
    model, layer = get_model(modality)
    store = {}

    def hook(_m, _i, out):
        store["a"] = out
        out.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(hook)
    try:
        logits = model(tensor)
        probs = torch.softmax(logits, 1)[0]
        idx = int(probs.argmax())
        model.zero_grad()
        logits[0, idx].backward()
    finally:
        h.remove()
    a, g = store["a"][0], store["g"][0]
    cam = torch.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0)).detach().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return probs.detach().numpy(), idx, cam


def _heat(cam, size):
    c = np.clip(cv2.resize(cam, size, interpolation=cv2.INTER_CUBIC), 0, 1)
    heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * c), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    return c, heat


def run_analysis(modality, img: Image.Image, out_dir):
    """Run full pipeline and write all intermediate images into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = MODALITIES[modality]
    t0 = time.perf_counter()

    display = img.convert("RGB")
    display.thumbnail((768, 768))
    display.save(os.path.join(out_dir, "original.png"))

    tensor, resized, norm = preprocess(img)
    Image.fromarray(resized).save(os.path.join(out_dir, "resized.png"))
    nv = norm.mean(2)
    nv = ((nv - nv.min()) / (nv.max() - nv.min() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(cv2.applyColorMap(nv, cv2.COLORMAP_BONE)[..., ::-1]).save(os.path.join(out_dir, "normalized.png"))
    gray = cv2.cvtColor(np.asarray(display), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    Image.fromarray(clahe).save(os.path.join(out_dir, "enhanced.png"))
    hist = np.histogram(gray, bins=32, range=(0, 255))[0]

    probs, idx, cam = gradcam(modality, tensor)
    c, heat = _heat(cam, display.size)
    Image.fromarray(heat).save(os.path.join(out_dir, "heatmap.png"))
    base = np.asarray(display).astype(np.float32)
    alpha = (0.6 * c)[..., None]
    Image.fromarray((base * (1 - alpha) + heat * alpha).astype(np.uint8)).save(os.path.join(out_dir, "overlay.png"))
    ms = (time.perf_counter() - t0) * 1000

    key = cfg["classes"][idx]
    label, abnormal, desc = cfg["labels"][key]
    return {
        "prediction": key,
        "label": label,
        "abnormal": abnormal,
        "description": desc,
        "confidence": float(probs[idx]),
        "probabilities": {cfg["labels"][k][0]: float(p) for k, p in zip(cfg["classes"], probs)},
        "inference_ms": round(ms, 1),
        "image_w": img.size[0], "image_h": img.size[1],
        "histogram": hist.tolist(),
        "stats": {"mean": float(gray.mean()), "std": float(gray.std()),
                  "norm_min": float(norm.min()), "norm_max": float(norm.max())},
    }
