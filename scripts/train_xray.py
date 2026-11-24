"""
Train a bone-fracture X-ray classifier (fractured / not fractured) by fine-tuning
MobileNetV3-Large (ImageNet weights) on CPU.

Usage:  python scripts/train_xray.py <dataset_dir> [epochs]
dataset_dir must contain train/, val/, test/ each with "fractured" and "not fractured" folders.
Outputs: xray_fracture_mnv3.pth (repo root) and app/model_meta/xray_metrics.json
"""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, ".cache_xray")
CLASSES = ["fractured", "not fractured"]
SIZE = 224


def load_one(path):
    from PIL import Image
    try:
        raw = open(path, "rb").read()
        img = Image.open(path).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8), hashlib.md5(raw).hexdigest()
    except Exception:
        return None, None


def build_cache(data_dir):
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for split in ["train", "val", "test"]:
        f = os.path.join(CACHE, f"{split}.npz")
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            out[split] = (z["x"], z["y"], list(z["h"]))
            continue
        paths, labels = [], []
        for ci, c in enumerate(CLASSES):
            d = os.path.join(data_dir, split, c)
            for n in sorted(os.listdir(d)):
                paths.append(os.path.join(d, n)); labels.append(ci)
        t = time.time()
        with ProcessPoolExecutor() as ex:
            res = list(ex.map(load_one, paths, chunksize=32))
        keep = [i for i, (a, _) in enumerate(res) if a is not None]
        x = np.stack([res[i][0] for i in keep]); y = np.array([labels[i] for i in keep]); h = [res[i][1] for i in keep]
        np.savez(f, x=x, y=y, h=np.array(h))
        print(f"cached {split}: {len(x)} images ({len(paths)-len(keep)} unreadable) in {time.time()-t:.0f}s", flush=True)
        out[split] = (x, y, h)
    return out


def main():
    import torch
    import torch.nn as nn
    from torchvision import models
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    data_dir = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    torch.set_num_threads(os.cpu_count())
    torch.manual_seed(0); np.random.seed(0)

    raw = build_cache(data_dir)
    htrain = set(raw["train"][2])
    dup_test = sum(h in htrain for h in raw["test"][2])
    dup_val = sum(h in htrain for h in raw["val"][2])
    print(f"exact duplicate files in original splits: test-in-train={dup_test}, val-in-train={dup_val}", flush=True)

    # The original Kaggle splits leak (most test files are byte-identical to training files).
    # Pool everything, keep one copy per unique file hash, drop hashes with conflicting labels,
    # then make a fresh stratified 80/10/10 split so test images were never seen in training.
    pool_x = np.concatenate([raw[s][0] for s in ("train", "val", "test")])
    pool_y = np.concatenate([raw[s][1] for s in ("train", "val", "test")])
    pool_h = sum((list(raw[s][2]) for s in ("train", "val", "test")), [])
    first, labels_by_h = {}, {}
    for i, h in enumerate(pool_h):
        labels_by_h.setdefault(h, set()).add(int(pool_y[i]))
        first.setdefault(h, i)
    keep = [i for h, i in first.items() if len(labels_by_h[h]) == 1]
    conflicts = sum(len(v) > 1 for v in labels_by_h.values())
    total_files = len(pool_h)
    rng = np.random.RandomState(42)
    data = {"train": [[], []], "val": [[], []], "test": [[], []]}
    for c in (0, 1):
        idx = np.array([i for i in keep if pool_y[i] == c]); rng.shuffle(idx)
        n_te = n_va = int(round(0.1 * len(idx)))
        for split, part in (("test", idx[:n_te]), ("val", idx[n_te:n_te + n_va]), ("train", idx[n_te + n_va:])):
            data[split][0].extend(part.tolist())
    for split in data:
        ids = np.array(sorted(data[split][0]))
        data[split] = (pool_x[ids], pool_y[ids], [pool_h[i] for i in ids])
    print(f"dedup: {total_files} files -> {len(keep)} unique ({conflicts} label conflicts dropped); "
          f"train {len(data['train'][0])} val {len(data['val'][0])} test {len(data['test'][0])}", flush=True)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def to_tensor(xb, augment=False):
        t = torch.from_numpy(xb).float().div(255).unsqueeze(1)
        if augment:
            flip = torch.rand(len(t)) < 0.5
            t[flip] = t[flip].flip(-1)
            # random brightness/contrast jitter
            a = (1 + (torch.rand(len(t), 1, 1, 1) - 0.5) * 0.3); b = (torch.rand(len(t), 1, 1, 1) - 0.5) * 0.2
            t = (t * a + b).clamp(0, 1)
        return (t.repeat(1, 3, 1, 1) - mean) / std

    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    xtr, ytr, _ = data["train"]
    bs = 48
    steps = epochs * ((len(xtr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.15)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    def predict(x):
        model.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(x), 128):
                outs.append(torch.softmax(model(to_tensor(x[i:i+128])), 1))
        return torch.cat(outs).numpy()

    best, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(xtr)); t = time.time(); tl = 0
        for k, i in enumerate(range(0, len(perm), bs)):
            idx = perm[i:i+bs]
            xb = to_tensor(xtr[idx], augment=True); yb = torch.from_numpy(ytr[idx]).long()
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step(); sched.step()
            tl += loss.item()
            if k % 20 == 0:
                print(f"  epoch {ep+1} step {k}/{len(perm)//bs} loss {loss.item():.3f} ({time.time()-t:.0f}s)", flush=True)
        pv = predict(data["val"][0]).argmax(1); va = accuracy_score(data["val"][1], pv)
        hist.append({"epoch": ep + 1, "train_loss": tl / (k + 1), "val_accuracy": va})
        print(f"epoch {ep+1}: train_loss {tl/(k+1):.4f} val_acc {va:.4f} ({time.time()-t:.0f}s)", flush=True)
        if va > best:
            best = va; best_state = {k2: v.clone() for k2, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(ROOT, "xray_fracture_mnv3.pth"))

    model.load_state_dict(best_state)
    xte, yte, _ = data["test"]
    pt = predict(xte).argmax(1)
    p, r, f, _ = precision_recall_fscore_support(yte, pt, average=None, labels=[0, 1])
    pm, rm, fm, _ = precision_recall_fscore_support(yte, pt, average="macro")
    metrics = {
        "model": "MobileNetV3-Large (ImageNet pretrained, fine-tuned)",
        "classes": CLASSES,
        "dataset": {"train": int(len(xtr)), "val": int(len(data["val"][0])), "test": int(len(xte))},
        "dedup": {"original_files": int(total_files), "unique_files": int(len(keep)), "label_conflicts_dropped": int(conflicts),
                  "original_test_in_train": int(dup_test), "original_val_in_train": int(dup_val),
                  "note": "Original Kaggle splits leaked; a fresh stratified 80/10/10 split over unique files was used."},
        "epochs": epochs, "history": hist,
        "test": {
            "accuracy": float(accuracy_score(yte, pt)),
            "precision_macro": float(pm), "recall_macro": float(rm), "f1_macro": float(fm),
            "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i, c in enumerate(CLASSES)},
            "confusion_matrix": confusion_matrix(yte, pt, labels=[0, 1]).tolist(),
        },
    }
    os.makedirs(os.path.join(ROOT, "app", "model_meta"), exist_ok=True)
    with open(os.path.join(ROOT, "app", "model_meta", "xray_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("TEST", json.dumps(metrics["test"]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
