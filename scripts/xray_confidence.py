"""Pick the 'Inconclusive' cutoff for the X-ray model on its held-out test split (never seen in training).
Prints, for each cutoff, how many test images become inconclusive and how many errors that catches."""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import ml  # noqa: E402
from train_xray import CACHE  # noqa: E402

raw = {s: np.load(os.path.join(CACHE, f"{s}.npz"), allow_pickle=True) for s in ("train", "val", "test")}
pool_x = np.concatenate([raw[s]["x"] for s in raw]); pool_y = np.concatenate([raw[s]["y"] for s in raw])
pool_h = sum((list(raw[s]["h"]) for s in raw), [])
first, lab = {}, {}
for i, h in enumerate(pool_h):
    lab.setdefault(h, set()).add(int(pool_y[i])); first.setdefault(h, i)
keep = [i for h, i in first.items() if len(lab[h]) == 1]
rng = np.random.RandomState(42); test = []
for c in (0, 1):  # identical to train_xray.py split
    idx = np.array([i for i in keep if pool_y[i] == c]); rng.shuffle(idx)
    test.extend(idx[:int(round(0.1 * len(idx)))].tolist())
ids = np.array(sorted(test)); x, y = pool_x[ids], pool_y[ids]

model, _ = ml.get_model("xray")
mean = torch.tensor(ml.MEAN).view(1, 3, 1, 1); std = torch.tensor(ml.STD).view(1, 3, 1, 1)
with torch.no_grad():
    t = torch.from_numpy(x).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1)
    p = torch.cat([torch.softmax(model((t[i:i + 64] - mean) / std), 1) for i in range(0, len(t), 64)]).numpy()
pred, conf = p.argmax(1), p.max(1)
print(f"test images {len(y)}, accuracy {np.mean(pred == y):.4f}, errors {int(np.sum(pred != y))}")
rows = []
for cut in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    inc = conf < cut
    rows.append({"cutoff": cut, "inconclusive": int(inc.sum()), "errors_caught": int(np.sum(inc & (pred != y))),
                 "errors_left": int(np.sum(~inc & (pred != y))), "correct_lost": int(np.sum(inc & (pred == y))),
                 "accuracy_on_decided": float(np.mean(pred[~inc] == y[~inc]))})
    print(rows[-1])
json.dump(rows, open(os.path.join(ROOT, "app", "model_meta", "xray_confidence.json"), "w"), indent=1)
