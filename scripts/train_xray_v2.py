"""X-ray fracture model v2: Kaggle (deduplicated) + FracAtlas, with a body-part head.

- FracAtlas images with metal hardware are dropped (97 of 99 are 'fractured' -> shortcut).
- Two heads on one MobileNetV3-Large: fracture (2 classes) and body part
  (hand / leg / hip / shoulder / mixed). The body-part loss is only applied to FracAtlas images
  (the Kaggle set has no body-part labels).
- Model selection on the mean validation AUC of the two sources; final numbers are reported per
  source and per body part on held-out test images, next to the old model on the same images.

usage: python scripts/train_xray_v2.py <fracatlas_dir> [epochs]
Outputs: xray_fracture_v2.pth (repo root), app/model_meta/xray_v2_metrics.json
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from train_xray import CACHE  # noqa: E402

REGIONS = ["hand", "leg", "hip", "shoulder", "mixed"]
SIZE, CROP = 256, 224


def kaggle_splits():
    """Same deduplicated 80/10/10 split as train_xray.py (test images identical to v1's test set)."""
    raw = {s: np.load(os.path.join(CACHE, f"{s}.npz"), allow_pickle=True) for s in ("train", "val", "test")}
    px = np.concatenate([raw[s]["x"] for s in raw]); py = np.concatenate([raw[s]["y"] for s in raw])
    ph = sum((list(raw[s]["h"]) for s in raw), [])
    first, lab = {}, {}
    for i, h in enumerate(ph):
        lab.setdefault(h, set()).add(int(py[i])); first.setdefault(h, i)
    keep = [i for h, i in first.items() if len(lab[h]) == 1]
    rng = np.random.RandomState(42); out = {"train": [], "val": [], "test": []}
    for c in (0, 1):
        idx = np.array([i for i in keep if py[i] == c]); rng.shuffle(idx)
        n = int(round(0.1 * len(idx)))
        out["test"] += idx[:n].tolist(); out["val"] += idx[n:2 * n].tolist(); out["train"] += idx[2 * n:].tolist()
    from PIL import Image
    res = {}
    for s, ids in out.items():
        ids = sorted(ids)
        x = np.stack([np.asarray(Image.fromarray(px[i]).resize((SIZE, SIZE), Image.BILINEAR)) for i in ids])
        res[s] = (x, 1 - py[ids].astype(np.int64), np.full(len(ids), -1))  # v1 label 0=fractured -> v2 1=fractured
    return res


def fracatlas_splits(root):
    import pandas as pd
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cache = os.path.join(ROOT, ".cache_xray", "fracatlas_256.npz")
    df = pd.read_csv(os.path.join(root, "dataset.csv"))
    df = df[df.hardware == 0].reset_index(drop=True)
    df["region"] = df[REGIONS].values.argmax(1)
    if os.path.exists(cache):
        x = np.load(cache)["x"]
    else:
        x = np.stack([np.asarray(Image.open(os.path.join(root, "images", "Fractured" if r.fractured else "Non_fractured", r.image_id))
                                 .convert("L").resize((SIZE, SIZE), Image.BILINEAR)) for r in df.itertuples()])
        np.savez(cache, x=x)
    rng = np.random.RandomState(0); split = np.empty(len(df), dtype=object)
    for _, g in df.groupby(["fractured", "region"]):  # stratified 70/15/15
        idx = g.index.values.copy(); rng.shuffle(idx)
        n_te = n_va = int(round(0.15 * len(idx)))
        split[idx[:n_te]] = "test"; split[idx[n_te:n_te + n_va]] = "val"; split[idx[n_te + n_va:]] = "train"
    df["split"] = split
    df.to_csv(os.path.join(ROOT, "app", "model_meta", "fracatlas_split.csv"), columns=["image_id", "split"], index=False)
    return {s: (x[df.split == s], df.fractured.values[df.split == s].astype(np.int64), df.region.values[df.split == s],
                df[df.split == s].reset_index(drop=True)) for s in ("train", "val", "test")}


def build_model():
    import torch.nn as nn
    from torchvision import models

    class XrayNet(nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()
            w = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            m = models.mobilenet_v3_large(weights=w)
            self.features, self.avgpool = m.features, m.avgpool
            self.neck = nn.Sequential(m.classifier[0], m.classifier[1], m.classifier[2])
            self.fracture = nn.Linear(1280, 2)
            self.region = nn.Linear(1280, len(REGIONS))

        def forward(self, x):
            h = self.neck(self.avgpool(self.features(x)).flatten(1))
            return self.fracture(h), self.region(h)

    return XrayNet()


def main():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    fa_dir = sys.argv[1]; epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    torch.set_num_threads(os.cpu_count()); torch.manual_seed(0); np.random.seed(0)
    t0 = time.time()
    kg, fa = kaggle_splits(), fracatlas_splits(fa_dir)
    print(f"data ready in {time.time() - t0:.0f}s: kaggle " + " ".join(f"{s}={len(kg[s][0])}" for s in kg)
          + " | fracatlas " + " ".join(f"{s}={len(fa[s][0])}" for s in fa), flush=True)

    xtr = np.concatenate([kg["train"][0], fa["train"][0]]); ytr = np.concatenate([kg["train"][1], fa["train"][1]])
    rtr = np.concatenate([kg["train"][2], fa["train"][2]])
    src = np.concatenate([np.zeros(len(kg["train"][0])), np.ones(len(fa["train"][0]))])
    # sample so each source and each fracture label get a fair share every epoch
    w = np.zeros(len(ytr))
    for s in (0, 1):
        for c in (0, 1):
            m = (src == s) & (ytr == c); w[m] = 1.0 / m.sum()
    w /= w.sum()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def to_tensor(xb, augment=False):
        t = torch.from_numpy(xb).float().div(255).unsqueeze(1)
        if augment:
            n = len(t)
            ang = (torch.rand(n) - 0.5) * 30 * np.pi / 180; sc = 1 + (torch.rand(n) - 0.5) * 0.3
            cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
            tx, ty = (torch.rand(n) - 0.5) * 0.2, (torch.rand(n) - 0.5) * 0.2
            theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
            t = F.grid_sample(t, F.affine_grid(theta, t.shape, align_corners=False), align_corners=False, padding_mode="zeros")
            flip = torch.rand(n) < 0.5; t[flip] = t[flip].flip(-1)
            a = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.4; b = (torch.rand(n, 1, 1, 1) - 0.5) * 0.2
            t = (t * a + b).clamp(0, 1)
            inv = torch.rand(n) < 0.1; t[inv] = 1 - t[inv]  # some films are stored inverted
            i, j = np.random.randint(0, SIZE - CROP + 1, 2)
        else:
            i = j = (SIZE - CROP) // 2
        t = t[:, :, i:i + CROP, j:j + CROP]
        return (t.repeat(1, 3, 1, 1) - mean) / std

    model = build_model()
    bs = 32; steps_per_epoch = len(ytr) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=8e-4, total_steps=epochs * steps_per_epoch, pct_start=0.15)

    def predict(x):
        model.eval(); pf, pr = [], []
        with torch.no_grad():
            for k in range(0, len(x), 96):
                a, b = model(to_tensor(x[k:k + 96])); pf.append(torch.softmax(a, 1)[:, 1]); pr.append(b.argmax(1))
        return torch.cat(pf).numpy(), torch.cat(pr).numpy()

    out_w = os.path.join(ROOT, "xray_fracture_v2.pth"); best, hist = -1, []
    for ep in range(epochs):
        model.train(); t = time.time(); tl = 0
        for k in range(steps_per_epoch):
            idx = np.random.choice(len(ytr), bs, p=w)
            xb = to_tensor(xtr[idx], augment=True); yb = torch.from_numpy(ytr[idx]); rb = torch.from_numpy(rtr[idx]).long()
            lf, lr_ = model(xb)
            loss = F.cross_entropy(lf, yb, label_smoothing=0.05)
            if (rb >= 0).any():
                loss = loss + 0.5 * F.cross_entropy(lr_[rb >= 0], rb[rb >= 0])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl += loss.item()
            if k % 50 == 0:
                print(f"  epoch {ep + 1} step {k}/{steps_per_epoch} loss {loss.item():.3f} ({time.time() - t:.0f}s)", flush=True)
        auc_k = roc_auc_score(kg["val"][1], predict(kg["val"][0])[0])
        pf, pr = predict(fa["val"][0]); auc_f = roc_auc_score(fa["val"][1], pf)
        score = (auc_k + auc_f) / 2
        hist.append({"epoch": ep + 1, "loss": tl / steps_per_epoch, "val_auc_kaggle": auc_k, "val_auc_fracatlas": auc_f,
                     "val_region_acc": float(np.mean(pr == fa["val"][2]))})
        print(f"epoch {ep + 1}: loss {tl / steps_per_epoch:.3f} val AUC kaggle {auc_k:.3f} fracatlas {auc_f:.3f} "
              f"region acc {hist[-1]['val_region_acc']:.3f} ({time.time() - t:.0f}s)", flush=True)
        if score > best:
            best = score; torch.save(model.state_dict(), out_w)

    model.load_state_dict(torch.load(out_w))
    # threshold: best Youden J on the pooled validation sets
    pv = np.concatenate([predict(kg["val"][0])[0], predict(fa["val"][0])[0]]); yv = np.concatenate([kg["val"][1], fa["val"][1]])
    ths = np.linspace(0.05, 0.95, 91)
    th = float(ths[np.argmax([np.mean(pv[yv == 1] >= x) + np.mean(pv[yv == 0] < x) - 1 for x in ths])])

    def report(y, p, extra=None):
        r = {"n": int(len(y)), "fractured": int(y.sum()), "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
             "sensitivity": float(np.mean(p[y == 1] >= th)) if y.sum() else None,
             "specificity": float(np.mean(p[y == 0] < th)) if (y == 0).sum() else None}
        if extra is not None:
            r.update(extra)
        return r

    pk = predict(kg["test"][0])[0]; pf, pr = predict(fa["test"][0])
    import pandas as pd
    base = pd.read_csv(os.path.join(os.path.dirname(fa_dir.rstrip("\\/")), "..", "baseline_preds.csv"))
    tdf = fa["test"][3].merge(base[["image_id", "p_frac"]], on="image_id", how="left")
    metrics = {"model": "MobileNetV3-Large, two heads (fracture + body part)", "threshold": th, "epochs": epochs, "history": hist,
               "train_size": int(len(ytr)), "hardware_images_dropped": True,
               "test": {"kaggle": report(kg["test"][1], pk), "fracatlas_all": report(fa["test"][1], pf)},
               "test_by_region": {}, "region_head_accuracy": float(np.mean(pr == fa["test"][2])),
               "old_model_same_fracatlas_test": {}}
    for ri, rn in enumerate(REGIONS):
        m = fa["test"][2] == ri
        metrics["test_by_region"][rn] = report(fa["test"][1][m], pf[m])
        yb_, pb_ = fa["test"][1][m], tdf.p_frac.values[m]
        metrics["old_model_same_fracatlas_test"][rn] = {"sensitivity": float(np.mean(pb_[yb_ == 1] >= 0.5)),
                                                        "specificity": float(np.mean(pb_[yb_ == 0] < 0.5))}
    with open(os.path.join(ROOT, "app", "model_meta", "xray_v2_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("RESULT", json.dumps({k: metrics[k] for k in ("threshold", "test", "test_by_region", "region_head_accuracy",
                                                        "old_model_same_fracatlas_test")}), flush=True)
    print(f"DONE in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
