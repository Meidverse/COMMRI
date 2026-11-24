"""Evaluate the MRI tumour model on a folder (glioma/meningioma/notumor/pituitary subfolders) and write
app/model_meta/mri_metrics.json. Headline numbers are the ones recorded at training time (model card);
the 'verification' block is this re-run on the given folder."""
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
import ml  # noqa: E402

classes = ml.MODALITIES["mri"]["classes"]
model, _ = ml.get_model("mri")
ys, ps = [], []
for ci, c in enumerate(classes):
    files = sorted(glob.glob(os.path.join(sys.argv[1], c, "*")))
    for i in range(0, len(files), 32):
        x = torch.cat([ml.preprocess(Image.open(f))[0] for f in files[i:i + 32]])
        with torch.no_grad():
            ps += model(x).argmax(1).tolist()
        ys += [ci] * len(files[i:i + 32])
p, r, f, _ = precision_recall_fscore_support(ys, ps, labels=range(4))
out = {
    "model": "TumorCNN (VGG-style 2D CNN)",
    "classes": classes,
    "dataset": {"name": "Kaggle Brain Tumor MRI", "total": 7023, "train_note": "trained 50 epochs, AdamW, OneCycleLR, batch 32"},
    "test": {  # recorded on the held-out test split at training time
        "accuracy": 0.9395, "precision_macro": 0.94, "recall_macro": 0.94, "f1_macro": 0.94,
        "per_class": {"glioma": {"recall": 0.981}, "meningioma": {"recall": 0.839}, "notumor": {"recall": 0.985}, "pituitary": {"recall": 0.943}},
    },
    "verification": {
        "images": len(ys), "accuracy": float(accuracy_score(ys, ps)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(ys, ps, labels=range(4)).tolist(),
        "note": "Re-evaluation on a separate copy of the Kaggle test folder; some images may overlap the original training pool.",
    },
}
os.makedirs(os.path.join(ml.APP_DIR, "model_meta"), exist_ok=True)
with open(os.path.join(ml.APP_DIR, "model_meta", "mri_metrics.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out["verification"], indent=1))
