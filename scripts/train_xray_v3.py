"""X-ray fracture model v3 — built to run on one rented GPU (vast.ai RTX 3090/4090).

What is new compared with v2
- Adds GRAZPEDWRI-DX (20,327 paediatric wrist X-rays, 6,091 patients, CC BY 4.0) through the
  "ClearWrist" Hugging Face mirror, which carries the original labels and patient-level splits.
- Higher input resolution (default 448 px cache, 416 px crop; v2 used 256/224).
- 16-bit PNGs are windowed to the 0.5-99.5 percentile before converting to 8 bit.
- Everything is resumable: downloads, per-file preprocessing and training (checkpoint every epoch).
  Run the same command again after a dropped instance and it carries on.
- Evaluation is on the SAME FracAtlas test images as v2 (app/model_meta/fracatlas_split.csv),
  and v2 itself is re-scored in the same run, so the comparison is like for like.

Rule for "v3 beats v2" (fixed before training; all three must hold on the FracAtlas test set):
  1. ROC AUC of v3 > ROC AUC of v2;
  2. at the specificity v2 reaches with its live cut-off (0.6), v3's sensitivity >= v2's;
  3. hand-region AUC of v3 >= v2's.
v2 stays live unless the verdict printed at the end is "v3 beats v2".

usage (on the GPU box, from the repo root):
  python scripts/train_xray_v3.py                 # full run: data -> train -> evaluate
  python scripts/train_xray_v3.py --prep-only     # just download + preprocess + label checks
  python scripts/train_xray_v3.py --smoke         # tiny CPU test with fake data (no downloads)
Outputs (in --out, default ./v3_out): xray_fracture_v3.pth, xray_v3_metrics.json
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGIONS = ["hand", "leg", "hip", "shoulder", "mixed"]
CLEARWRIST = "Layered-Labs/clearwrist-pediatric-wrist-xray"
FRACATLAS_REPO, FRACATLAS_FILE = "runananya/fracatlas", "archive (9).zip"
GRAZ_COLS = ["patient_id", "fracture_visible", "has_cast", "has_metal", "diagnosis_uncertain", "has_osteopenia",
             "projection", "age_years"]
V2_LIVE_HI = 0.6  # v2 live cut-off: score >= 0.6 is reported as a fracture


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def done(work, name):
    return os.path.exists(os.path.join(work, name + ".done"))


def mark(work, name, info=""):
    with open(os.path.join(work, name + ".done"), "w") as fh:
        fh.write(info or time.strftime("%Y-%m-%d %H:%M:%S"))


# ---------------------------------------------------------------- image decoding (worker processes)
def window_to_u8(a, size):
    """Any-depth greyscale array -> percentile-windowed uint8 square of `size`. None if the image is blank/broken."""
    from PIL import Image
    if a.ndim == 3:
        a = a[..., :3].mean(-1)
    a = a.astype(np.float32)
    sub = a[::4, ::4]
    if sub.size < 64 or len(np.unique(sub[::4, ::4])) < 16:  # blank or posterised (the old ClearWrist bug)
        return None
    lo, hi = np.percentile(sub, (0.5, 99.5))
    if hi - lo < 1e-6:
        return None
    a = np.clip((a - lo) / (hi - lo), 0, 1)
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR))


def decode_bytes(args):
    b, size = args
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return window_to_u8(np.asarray(Image.open(io.BytesIO(b))), size)
    except Exception:
        return None


def decode_path(args):
    p, size = args
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return window_to_u8(np.asarray(Image.open(p)), size)
    except Exception:
        return None


# ---------------------------------------------------------------- data: GRAZPEDWRI-DX (ClearWrist mirror)
def graz_files(args):
    if args.graz_dir:  # local parquet files (smoke test / manual copy)
        return sorted(f for f in os.listdir(args.graz_dir) if f.endswith(".parquet"))
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files(CLEARWRIST, repo_type="dataset") if f.endswith(".parquet")]
    return sorted(files)


def graz_revision():
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(CLEARWRIST).sha
    except Exception:
        return None


def prep_graz(args, pool):
    work = args.work
    files = graz_files(args)
    if not files:
        sys.exit("No GRAZPEDWRI-DX parquet files found.")
    log(f"GRAZPEDWRI-DX: {len(files)} parquet files")
    os.makedirs(os.path.join(work, "graz"), exist_ok=True)
    for k, f in enumerate(files):
        tag = "graz_" + os.path.basename(f).replace(".parquet", "")
        if done(work, tag):
            continue
        t = time.time()
        if args.graz_dir:
            path = os.path.join(args.graz_dir, f)
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(CLEARWRIST, f, repo_type="dataset", local_dir=os.path.join(work, "dl"))
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        have = set(pf.schema_arrow.names)
        missing = [c for c in ["image"] + GRAZ_COLS if c not in have]
        if missing:
            sys.exit(f"STOP: {f} is missing columns {missing}. Columns present: {sorted(have)}")
        xs, meta = [], {c: [] for c in GRAZ_COLS}
        bad = 0
        for batch in pf.iter_batches(batch_size=128, columns=["image"] + GRAZ_COLS):
            d = batch.to_pydict()
            imgs = pool.map(decode_bytes, [((im or {}).get("bytes") or b"", args.size) for im in d["image"]])
            for i, x in enumerate(imgs):
                if x is None:
                    bad += 1
                    continue
                xs.append(x)
                for c in GRAZ_COLS:
                    meta[c].append(d[c][i])
        split = os.path.basename(f).split("-")[0]  # train / validation / test
        out = os.path.join(work, "graz", tag + ".npz")
        np.savez(out, x=np.stack(xs) if xs else np.zeros((0, args.size, args.size), np.uint8), split=split,
                 **{c: np.array([np.nan if v is None else v for v in meta[c]], dtype=np.float32) for c in GRAZ_COLS})
        if not args.graz_dir and not args.keep_downloads:
            os.remove(path)  # keep disk use low; the npz is all we need
        mark(work, tag, json.dumps({"images": len(xs), "skipped_broken": bad}))
        log(f"  [{k + 1}/{len(files)}] {f}: {len(xs)} images, {bad} broken skipped ({time.time() - t:.0f}s)")
        if k == 0:
            label_report(meta)


def label_report(meta):
    """Print label balance and shortcut crosstabs from the first file, before paying for the rest."""
    import pandas as pd
    df = pd.DataFrame(meta)
    log("label check on first file:")
    print(df["fracture_visible"].value_counts(dropna=False).to_string(), flush=True)
    for c in ("has_cast", "has_metal", "diagnosis_uncertain"):
        print(f"--- {c} vs fracture_visible\n" + pd.crosstab(df[c], df["fracture_visible"]).to_string(), flush=True)
    if df["fracture_visible"].isna().all():
        sys.exit("STOP: fracture_visible is empty — labels missing, not training on this.")


def xtab(a, b):
    """Crosstab as plain JSON-safe dict, e.g. {"has_cast=1": {"fracture_visible=1": 812, ...}}."""
    import pandas as pd
    t = pd.crosstab(a, b)
    return {f"{a.name}={int(i)}": {f"{b.name}={int(j)}": int(t.loc[i, j]) for j in t.columns} for i in t.index}


def load_graz(args):
    """All cached GRAZ arrays -> dict split -> (x, y, meta DataFrame). Applies the exclusion rules."""
    import pandas as pd
    parts = {"train": [], "validation": [], "test": []}
    d = os.path.join(args.work, "graz")
    for f in sorted(os.listdir(d)):
        z = np.load(os.path.join(d, f))
        parts[str(z["split"])].append(z)
    out, excl = {}, {}
    for s, zs in parts.items():
        if not zs:
            continue
        x = np.concatenate([z["x"] for z in zs])
        df = pd.DataFrame({c: np.concatenate([z[c] for z in zs]) for c in GRAZ_COLS})
        keep = df.fracture_visible.notna().values
        confound = ((df.has_cast == 1) | (df.has_metal == 1)).values
        if s == "train":  # casts/metal sit mostly on known fractures (shortcut); uncertain labels are noise
            keep &= ~confound & ~(df.diagnosis_uncertain == 1).values
        elif s == "validation":
            keep &= ~confound
        excl[s] = {"total": int(len(df)), "kept": int(keep.sum()), "cast_or_metal": int(confound.sum()),
                   "diagnosis_uncertain": int((df.diagnosis_uncertain == 1).sum()),
                   "crosstab_cast_vs_fracture": xtab(df.has_cast, df.fracture_visible),
                   "crosstab_metal_vs_fracture": xtab(df.has_metal, df.fracture_visible)}
        df = df[keep].reset_index(drop=True)
        out[s] = (x[keep], df.fracture_visible.values.astype(np.int64), df)
    return out, excl


# ---------------------------------------------------------------- data: FracAtlas (same split as v2)
def fracatlas_root(args):
    if args.fracatlas_dir:
        return args.fracatlas_dir
    root = os.path.join(args.work, "fracatlas")
    if not done(args.work, "fracatlas_unzip"):
        from huggingface_hub import hf_hub_download
        z = hf_hub_download(FRACATLAS_REPO, FRACATLAS_FILE, repo_type="dataset", local_dir=os.path.join(args.work, "dl"))
        with zipfile.ZipFile(z) as zf:
            zf.extractall(root)
        mark(args.work, "fracatlas_unzip")
    for dp, _, fs in os.walk(root):
        if "dataset.csv" in fs and os.path.isdir(os.path.join(dp, "images")):
            return dp
    sys.exit("FracAtlas dataset.csv not found after unzip")


def fa_image_path(root, r):
    return os.path.join(root, "images", "Fractured" if r.fractured else "Non_fractured", r.image_id)


def prep_fracatlas(args, pool):
    import pandas as pd
    root = fracatlas_root(args)
    cache = os.path.join(args.work, f"fracatlas_{args.size}.npz")
    split = pd.read_csv(os.path.join(ROOT, "app", "model_meta", "fracatlas_split.csv"))  # v2's split, read-only
    df = pd.read_csv(os.path.join(root, "dataset.csv")).merge(split, on="image_id")  # hardware images not in split
    if args.limit_fa:
        df = df.groupby(["split", "fractured"], group_keys=False).head(args.limit_fa // 2).reset_index(drop=True)
    df["region"] = df[REGIONS].values.argmax(1)
    if not os.path.exists(cache):
        xs = pool.map(decode_path, [(fa_image_path(root, r), args.size) for r in df.itertuples()])
        ok = np.array([x is not None for x in xs])
        np.savez(cache, x=np.stack([x for x in xs if x is not None]), ok=ok)
        log(f"FracAtlas: {ok.sum()} images cached, {(~ok).sum()} unreadable")
    z = np.load(cache)
    df = df[z["ok"]].reset_index(drop=True)
    return root, {s: (z["x"][(df.split == s).values], df.fractured.values[(df.split == s).values].astype(np.int64),
                      df.region.values[(df.split == s).values], df[df.split == s].reset_index(drop=True))
                  for s in ("train", "val", "test")}


# ---------------------------------------------------------------- model
def build_model(pretrained=True):
    import torch.nn as nn
    from torchvision import models

    class XrayNet(nn.Module):  # same layout as the live v2 model (app/ml.py), so the app can load it
        def __init__(self):
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


# ---------------------------------------------------------------- metrics helpers
def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def sens_at_spec(y, p, spec):
    """Sensitivity at the threshold where specificity first reaches `spec`."""
    neg = np.sort(p[y == 0])
    if not len(neg) or not y.sum():
        return None, None
    th = neg[min(len(neg) - 1, int(np.ceil(spec * len(neg))) - 1)] + 1e-9
    return float(np.mean(p[y == 1] >= th)), float(th)


def rates(y, p, lo, hi):
    """Three-way read-out like the app: < lo normal, >= hi fracture, in between inconclusive."""
    pos, neg = y == 1, y == 0
    return {"n": int(len(y)), "fractured": int(y.sum()), "auc": auc(y, p),
            "sensitivity_at_hi": float(np.mean(p[pos] >= hi)) if pos.any() else None,
            "specificity_at_hi": float(np.mean(p[neg] < hi)) if neg.any() else None,
            "missed_fractures_below_lo": float(np.mean(p[pos] < lo)) if pos.any() else None,
            "normals_cleared_below_lo": float(np.mean(p[neg] < lo)) if neg.any() else None,
            "inconclusive_rate": float(np.mean((p >= lo) & (p < hi)))}


def boot_auc_diff(y, a, b, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            ds.append(auc(y[i], a[i]) - auc(y[i], b[i]))
    return [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("V3_WORK", os.path.join(ROOT, "v3_work")))
    ap.add_argument("--out", default=os.path.join(ROOT, "v3_out"))
    ap.add_argument("--size", type=int, default=448, help="cached image size")
    ap.add_argument("--crop", type=int, default=416, help="training/eval crop")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-hours", type=float, default=3.0, help="training wall-clock cap; evaluation still runs")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--keep-downloads", action="store_true")
    ap.add_argument("--graz-dir", help="local folder of ClearWrist parquet files instead of downloading")
    ap.add_argument("--fracatlas-dir", help="local FracAtlas folder (with dataset.csv) instead of downloading")
    ap.add_argument("--limit-fa", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--stop-after-epoch", type=int, default=0, help=argparse.SUPPRESS)  # resume test
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run on generated data")
    args = ap.parse_args()
    if args.smoke:
        smoke_setup(args)
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu" and not args.smoke and not args.prep_only:
        sys.exit("No CUDA GPU visible — refusing to train on CPU. (Use --prep-only or --smoke.)")
    log(f"device {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "") + f", workers {args.workers}")

    with Pool(args.workers) as pool:
        if not done(args.work, "graz_all"):
            prep_graz(args, pool)
            mark(args.work, "graz_all")
        fa_root, fa = prep_fracatlas(args, pool)
    gz, excl = load_graz(args)
    for s in gz:
        log(f"GRAZ {s}: {len(gz[s][1])} images kept, {gz[s][1].mean():.0%} fractured")
    for s in fa:
        log(f"FracAtlas {s}: {len(fa[s][1])} images, {fa[s][1].mean():.0%} fractured")
    log(f"data ready in {(time.time() - t0) / 60:.1f} min")
    if args.prep_only:
        return

    import torch.nn.functional as F
    torch.manual_seed(0)
    np.random.seed(0)
    use_amp = dev == "cuda"
    if dev == "cuda":
        torch.backends.cudnn.benchmark = True

    # training pool: GRAZ train + FracAtlas train; region label: GRAZ = hand/wrist
    xtr = np.concatenate([gz["train"][0], fa["train"][0]])
    ytr = np.concatenate([gz["train"][1], fa["train"][1]])
    rtr = np.concatenate([np.zeros(len(gz["train"][1]), np.int64), fa["train"][2].astype(np.int64)])
    src = np.concatenate([np.zeros(len(gz["train"][1])), np.ones(len(fa["train"][1]))])
    share = {0: 0.65, 1: 0.35}  # sampling share per source; labels balanced within each source
    w = np.zeros(len(ytr))
    for s in (0, 1):
        for c in (0, 1):
            m = (src == s) & (ytr == c)
            if m.any():
                w[m] = share[s] / 2 / m.sum()
    w /= w.sum()

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    S, C = args.size, args.crop

    def to_tensor(xb, augment=False):
        t = torch.from_numpy(xb).to(dev).float().div(255).unsqueeze(1)
        if augment:
            n = len(t)
            ang = (torch.rand(n, device=dev) - 0.5) * 30 * np.pi / 180
            sc = 1 + (torch.rand(n, device=dev) - 0.5) * 0.3
            cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
            tx, ty = (torch.rand(n, device=dev) - 0.5) * 0.2, (torch.rand(n, device=dev) - 0.5) * 0.2
            theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
            t = F.grid_sample(t, F.affine_grid(theta, t.shape, align_corners=False), align_corners=False)
            flip = torch.rand(n, device=dev) < 0.5
            t[flip] = t[flip].flip(-1)
            a = 1 + (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.4
            b = (torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.2
            g = torch.exp((torch.rand(n, 1, 1, 1, device=dev) - 0.5) * 0.6)  # gamma
            t = (t.clamp(1e-4, 1) ** g * a + b).clamp(0, 1)
            inv = torch.rand(n, device=dev) < 0.1
            t[inv] = 1 - t[inv]  # some films are stored inverted
            i, j = np.random.randint(0, S - C + 1, 2)
        else:
            i = j = (S - C) // 2
        t = t[:, :, i:i + C, j:j + C]
        return ((t.repeat(1, 3, 1, 1) - mean) / std).contiguous(memory_format=torch.channels_last)

    model = build_model(pretrained=not args.smoke).to(dev).to(memory_format=torch.channels_last)
    steps = max(1, len(ytr) // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=8e-4, total_steps=args.epochs * steps, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def predict(m, x, tf=None):
        m.eval()
        pf, pr = [], []
        with torch.no_grad(), torch.autocast(dev, enabled=use_amp):
            for k in range(0, len(x), 128):
                a, b = m(tf(x[k:k + 128]) if tf else to_tensor(x[k:k + 128]))
                pf.append(torch.softmax(a.float(), 1)[:, 1].cpu())
                pr.append(b.argmax(1).cpu())
        return torch.cat(pf).numpy(), torch.cat(pr).numpy()

    ckpt, best_path = os.path.join(args.work, "ckpt.pt"), os.path.join(args.work, "best.pt")
    start, best, hist = 0, -1.0, []
    if os.path.exists(ckpt):
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"])
        start, best, hist = st["epoch"], st["best"], st["hist"]
        np.random.set_state(st["np_rng"])
        torch.set_rng_state(st["torch_rng"])
        if dev == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(st["cuda_rng"])
        log(f"resumed from checkpoint after epoch {start} (best val score {best:.4f})")

    train_t0, capped = time.time(), False
    for ep in range(start, args.epochs):
        model.train()
        t, tl = time.time(), 0.0
        for k in range(steps):
            idx = np.sort(np.random.choice(len(ytr), args.batch, p=w))
            xb = to_tensor(xtr[idx], augment=True)
            yb = torch.from_numpy(ytr[idx]).to(dev)
            rb = torch.from_numpy(rtr[idx]).to(dev)
            with torch.autocast(dev, enabled=use_amp):
                lf, lr_ = model(xb)
                loss = F.cross_entropy(lf.float(), yb, label_smoothing=0.05) + 0.5 * F.cross_entropy(lr_.float(), rb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
            if k % 100 == 0:
                log(f"  epoch {ep + 1} step {k}/{steps} loss {loss.item():.3f} ({time.time() - t:.0f}s)")
        auc_g = auc(gz["validation"][1], predict(model, gz["validation"][0])[0])
        pf, pr = predict(model, fa["val"][0])
        auc_f = auc(fa["val"][1], pf)
        score = np.mean([v for v in (auc_g, auc_f) if v is not None])
        hist.append({"epoch": ep + 1, "loss": tl / steps, "val_auc_graz": auc_g, "val_auc_fracatlas": auc_f,
                     "val_region_acc_fracatlas": float(np.mean(pr == fa["val"][2])), "seconds": time.time() - t})
        log(f"epoch {ep + 1}/{args.epochs}: loss {tl / steps:.3f} val AUC graz {auc_g} fracatlas {auc_f} "
            f"({time.time() - t:.0f}s)")
        if not np.isfinite(score):
            score = 0.0
        if score > best or not os.path.exists(best_path):
            best = float(score)
            torch.save(model.state_dict(), best_path)
        st = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": ep + 1, "best": best, "hist": hist,
              "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state() if dev == "cuda" else None}
        torch.save(st, ckpt + ".tmp")
        os.replace(ckpt + ".tmp", ckpt)  # atomic: a crash never leaves a half-written checkpoint
        if args.stop_after_epoch and ep + 1 == args.stop_after_epoch:
            log("stopping here on purpose (--stop-after-epoch); rerun to resume")
            return
        if (time.time() - train_t0) / 3600 > args.max_hours:
            capped = True
            log(f"training time cap of {args.max_hours} h reached — evaluating the best checkpoint so far")
            break

    # ------------------------------------------------------------ evaluation
    model.load_state_dict(torch.load(best_path, map_location=dev))
    # cut-offs chosen on FracAtlas validation only (adult, multi-region — closest to real use)
    pv, _ = predict(model, fa["val"][0])
    yv = fa["val"][1]
    ths = np.linspace(0.01, 0.99, 197)
    lo = float(max([x for x in ths if np.mean(pv[yv == 1] >= x) >= 0.90] or [0.01]))  # keep >= 90% of fractures out of "normal"
    hi = float(min([x for x in ths if np.mean(pv[yv == 0] < x) >= 0.95] or [0.99]))  # >= 95% specificity for "fracture"
    if hi < lo:
        lo = hi

    # v2 re-scored on the same images with its own preprocessing (plain greyscale, 256 resize, 224 centre crop)
    v2 = build_model(pretrained=False).to(dev)
    v2.load_state_dict(torch.load(os.path.join(ROOT, "xray_fracture_v2.pth"), map_location=dev))
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    tdf = fa["test"][3]
    v2x = np.stack([np.asarray(Image.open(fa_image_path(fa_root, r)).convert("L").resize((256, 256), Image.BILINEAR))
                    for r in tdf.itertuples()])
    m2, s2 = mean.cpu(), std.cpu()

    def v2_tf(xb):
        t = torch.from_numpy(xb).float().div(255).unsqueeze(1)[:, :, 16:240, 16:240]
        return ((t.repeat(1, 3, 1, 1) - m2) / s2).to(dev)

    p2, _ = predict(v2, v2x, v2_tf)
    p3, r3 = predict(model, fa["test"][0])
    y = fa["test"][1]
    reg = fa["test"][2]

    v2_spec = float(np.mean(p2[y == 0] < V2_LIVE_HI))
    v2_sens = float(np.mean(p2[y == 1] >= V2_LIVE_HI))
    v3_sens_matched, v3_th_matched = sens_at_spec(y, p3, v2_spec)
    hand = reg == 0
    a3, a2 = auc(y, p3), auc(y, p2)
    h3, h2 = auc(y[hand], p3[hand]), auc(y[hand], p2[hand])
    checks = {"auc_higher": (a3 or 0) > (a2 or 0), "sensitivity_at_v2_specificity_not_lower": (v3_sens_matched or 0) >= v2_sens,
              "hand_auc_not_lower": (h3 or 0) >= (h2 or 0)}
    verdict = "v3 beats v2" if all(checks.values()) else "keep v2"

    gt = gz["test"]
    pg3 = predict(model, gt[0])[0]
    clean = ((gt[2].has_cast != 1) & (gt[2].has_metal != 1)).values
    metrics = {
        "model": "MobileNetV3-Large, two heads (fracture + body part) — same layout as v2",
        "input": {"preprocess": "greyscale; window to 0.5-99.5 percentile; resize to size x size (bilinear); "
                                "centre crop; ImageNet mean/std; grey repeated to 3 channels",
                  "size": S, "crop": C},
        "cutoffs_from_fracatlas_val": {"lo": lo, "hi": hi,
                                       "rule": "lo: highest cut keeping >=90% val fractures out of 'normal'; "
                                               "hi: lowest cut with >=95% val specificity"},
        "beats_v2_rule": "FracAtlas test: AUC_v3 > AUC_v2 AND sensitivity at v2's live specificity >= v2 "
                         "AND hand AUC_v3 >= hand AUC_v2",
        "beats_v2_checks": checks, "verdict": verdict,
        "fracatlas_test": {
            "v3": rates(y, p3, lo, hi), "v2_rescored": {"auc": a2, "sensitivity_at_0.6": v2_sens, "specificity_at_0.6": v2_spec},
            "v3_sensitivity_at_v2_specificity": v3_sens_matched, "v3_threshold_for_that": v3_th_matched,
            "auc_difference_v3_minus_v2_95ci": boot_auc_diff(y, p3, p2),
            "by_region": {rn: {"n": int((reg == ri).sum()), "fractured": int(y[reg == ri].sum()),
                               "auc_v3": auc(y[reg == ri], p3[reg == ri]), "auc_v2": auc(y[reg == ri], p2[reg == ri])}
                          for ri, rn in enumerate(REGIONS)},
            "region_head_accuracy": float(np.mean(r3 == reg))},
        "graz_test": {"all": rates(gt[1], pg3, lo, hi), "without_cast_or_metal": rates(gt[1][clean], pg3[clean], lo, hi)},
        "data": {"graz_source": f"hf://datasets/{CLEARWRIST}", "graz_revision": None if args.smoke else graz_revision(),
                 "graz_exclusions": excl, "fracatlas_split": "app/model_meta/fracatlas_split.csv (v2's split)",
                 "train_images": int(len(ytr)), "sampling_share": {"graz": share[0], "fracatlas": share[1]}},
        "training": {"epochs_planned": args.epochs, "epochs_run": len(hist), "time_capped": capped, "history": hist,
                     "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
                     "total_minutes": (time.time() - t0) / 60},
    }
    torch.save(model.state_dict(), os.path.join(args.out, "xray_fracture_v3.pth"))
    with open(os.path.join(args.out, "xray_v3_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"FracAtlas test AUC: v3 {a3} vs v2 {a2} | hand v3 {h3} vs v2 {h2} | "
        f"sensitivity at v2's specificity ({v2_spec:.2f}): v3 {v3_sens_matched} vs v2 {v2_sens:.2f}")
    log(f"VERDICT: {verdict}  {checks}")
    log(f"wrote {args.out}/xray_fracture_v3.pth and xray_v3_metrics.json — total {(time.time() - t0) / 60:.0f} min")


# ---------------------------------------------------------------- smoke test
def smoke_setup(args):
    """Fake ClearWrist parquet files (16-bit PNGs) + a small slice of the local FracAtlas copy."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    base = os.path.join(ROOT, "v3_smoke")
    args.work, args.out = os.path.join(base, "work"), os.path.join(base, "out")
    gdir = os.path.join(base, "fake_graz")
    os.makedirs(gdir, exist_ok=True)
    rng = np.random.RandomState(1)
    for split, n in (("train", 40), ("validation", 16), ("test", 16)):
        p = os.path.join(gdir, f"{split}-00000.parquet")
        if os.path.exists(p):
            continue
        rows = {c: [] for c in ["image"] + GRAZ_COLS}
        for i in range(n):
            frac = i % 2
            a = (rng.rand(300, 240) * 20000 + 5000).astype(np.uint16)
            if frac:
                a[140:160, :] = 60000  # a bright "fracture line" so the toy task is learnable
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, format="PNG")
            rows["image"].append({"bytes": buf.getvalue(), "path": f"{split}_{i}.png"})
            rows["patient_id"].append(i)
            rows["fracture_visible"].append(frac)
            rows["has_cast"].append(int(i % 7 == 0))
            rows["has_metal"].append(0)
            rows["diagnosis_uncertain"].append(int(i % 11 == 0))
            rows["has_osteopenia"].append(0)
            rows["projection"].append(1 + i % 2)
            rows["age_years"].append(10.0)
        # one broken (posterised) image, like the old ClearWrist train bug — must be skipped
        buf = io.BytesIO()
        Image.fromarray(np.zeros((300, 240), np.uint16)).save(buf, format="PNG")
        rows["image"].append({"bytes": buf.getvalue(), "path": "broken.png"})
        for c, v in zip(GRAZ_COLS, (999, 1, 0, 0, 0, 0, 1, 10.0)):
            rows[c].append(v)
        pq.write_table(pa.table(rows), p)
    args.graz_dir = gdir
    args.fracatlas_dir = args.fracatlas_dir or os.path.join(os.path.dirname(ROOT), "datasets", "fracatlas", "data", "FracAtlas")
    args.limit_fa = args.limit_fa or 40
    args.size, args.crop = 96, 88
    args.epochs = 2 if args.epochs == 24 else args.epochs
    args.batch = 16
    args.workers = min(args.workers, 4)


if __name__ == "__main__":
    main()
