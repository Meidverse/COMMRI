"""Build the 'is this a valid scan for this model?' check.

For each model we take deep features (the layer before the classifier) of real training images,
store one centroid per class, and set a similarity threshold from the in-distribution scores.
At upload time an image whose best cosine similarity is below the threshold is rejected.
Usage: python scripts/build_ood.py
"""
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import ml  # noqa: E402

MRI_TRAIN = r"C:\all projects\zz_projects\medAI\mri\dataset\classification\training"
XRAY_CACHE = os.path.join(ROOT, ".cache_xray", "train.npz")
rng = np.random.RandomState(0)


def feats(mod, imgs):
    out = []
    for i in range(0, len(imgs), 32):
        x = torch.cat([ml.preprocess(im)[0] for im in imgs[i:i + 32]])
        out.append(ml.embed(mod, x))
    return np.concatenate(out)


def build(mod, imgs, labels):
    f = feats(mod, imgs)
    f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    cents = np.stack([f[labels == c].mean(0) for c in sorted(set(labels))])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    s = (f @ cents.T).max(1)
    th = float(np.percentile(s, 0.5)) - 0.05  # keep 99.5 % of real scans, plus a margin
    np.savez(os.path.join(ml.APP_DIR, "model_meta", f"{mod}_ood.npz"), centroids=cents, threshold=th, median=float(np.median(s)))
    print(f"{mod}: {len(f)} imgs, sim min {s.min():.3f} p0.5 {np.percentile(s, .5):.3f} median {np.median(s):.3f} -> threshold {th:.3f}")


# MRI: 250 per class from the training folder
imgs, labels = [], []
for ci, c in enumerate(ml.MODALITIES["mri"]["classes"]):
    files = sorted(glob.glob(os.path.join(MRI_TRAIN, c, "*")))
    for f in rng.choice(files, 250, replace=False):
        imgs.append(Image.open(f).convert("RGB")); labels.append(ci)
build("mri", imgs, np.array(labels))

# X-ray: 1000 from the cached training images
z = np.load(XRAY_CACHE)
idx = rng.choice(len(z["x"]), 1000, replace=False)
ximgs = [Image.fromarray(a).convert("RGB") for a in z["x"][idx]]
build("xray", ximgs, z["y"][idx])

# Modality classifier (MRI vs X-ray) on X-ray-model features
from sklearn.linear_model import LogisticRegression  # noqa: E402
F = feats("xray", imgs + ximgs)
F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
y = np.array([1] * len(imgs) + [0] * len(ximgs))  # 1 = MRI
clf = LogisticRegression(C=1.0, max_iter=2000).fit(F, y)
print("modality classifier train acc", clf.score(F, y))
np.savez(os.path.join(ml.APP_DIR, "model_meta", "modality_clf.npz"), coef=clf.coef_[0], intercept=clf.intercept_[0])
