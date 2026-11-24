"""
NeuroScan AI - Brain MRI tumor detection demo.

Runs the trained TumorCNN (kaggle_tumor_2dcnn_best.pth) on an uploaded brain MRI slice,
returns the 4-class prediction and a Grad-CAM heatmap showing which region drove the decision.

Run:  python app\app.py   then open http://127.0.0.1:5000
"""
import base64
import io
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from torchvision import transforms

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
MODEL_PATH = os.path.join(ROOT, "kaggle_tumor_2dcnn_best.pth")
SAMPLES_DIR = os.path.join(APP_DIR, "samples")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
INFO = {
    "glioma": ("Glioma Tumor", True,
               "Tumor arising from glial cells of the brain or spine. Often found in the cerebral hemispheres."),
    "meningioma": ("Meningioma Tumor", True,
                   "Tumor of the meninges (membranes covering the brain). Usually slow-growing and often benign."),
    "notumor": ("No Tumor Detected", False,
                "No tumor pattern found in this slice. Brain tissue appears normal to the model."),
    "pituitary": ("Pituitary Tumor", True,
                  "Tumor in the pituitary gland at the base of the brain. Can affect hormone levels."),
}


class TumorCNN(nn.Module):
    """VGG-style 2D CNN, 4.85M parameters. Must match the training architecture exactly."""

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


def load_model():
    m = TumorCNN(len(CLASSES))
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    m.eval()
    # In-place ReLUs break gradient hooks used by Grad-CAM.
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
    return m


model = load_model()
TARGET_LAYER = model.features[29]  # ReLU after the last conv block (28x28 feature map)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def analyze(img: Image.Image):
    """Return class probabilities and a Grad-CAM map for the predicted class."""
    x = preprocess(img).unsqueeze(0)
    store = {}

    def fwd_hook(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = TARGET_LAYER.register_forward_hook(fwd_hook)
    try:
        logits = model(x)
        probs = torch.softmax(logits, 1)[0]
        idx = int(probs.argmax())
        model.zero_grad()
        logits[0, idx].backward()
    finally:
        h.remove()

    act, grad = store["act"][0], store["grad"][0]           # (512, 28, 28)
    weights = grad.mean(dim=(1, 2))                           # channel importance
    cam = torch.relu((weights[:, None, None] * act).sum(0)).detach().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    return probs.detach().numpy(), idx, cam


def overlay(img: Image.Image, cam: np.ndarray):
    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]
    cam_r = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_r = np.clip(cam_r, 0, 1)
    heat = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    alpha = (0.55 * cam_r)[..., None]                       # stronger colour where attention is high
    out = (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)
    return out


def to_data_url(arr_or_img):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024


@app.get("/")
def index():
    samples = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    return render_template("index.html", samples=samples)


@app.get("/samples/<path:name>")
def sample(name):
    return send_from_directory(SAMPLES_DIR, name)


@app.post("/predict")
def predict():
    f = request.files.get("image")
    if f is None or f.filename == "":
        return jsonify(error="No image uploaded"), 400
    try:
        img = Image.open(f.stream).convert("RGB")
    except Exception:
        return jsonify(error="Could not read that file as an image (use JPG or PNG)"), 400

    t0 = time.perf_counter()
    probs, idx, cam = analyze(img)
    ms = (time.perf_counter() - t0) * 1000

    key = CLASSES[idx]
    label, is_tumor, desc = INFO[key]
    display = img.copy()
    display.thumbnail((512, 512))
    return jsonify(
        prediction=key,
        label=label,
        tumor=is_tumor,
        description=desc,
        confidence=float(probs[idx]),
        probabilities={INFO[c][0]: float(p) for c, p in zip(CLASSES, probs)},
        original=to_data_url(display),
        heatmap=to_data_url(overlay(display, cam)),
        inference_ms=round(ms, 1),
    )


if __name__ == "__main__":
    print("NeuroScan AI running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
