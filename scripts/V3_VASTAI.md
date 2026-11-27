# X-ray model v3 — training run on a rented vast.ai GPU

One script, `scripts/train_xray_v3.py`, does the whole run on the GPU box:
download data → preprocess → train → evaluate against v2 → save weights and metrics.
**v2 stays live.** Nothing in the app changes until v3 has been checked and wired in.

## What v3 changes

| | v2 (live) | v3 |
|---|---|---|
| Training data | Kaggle + FracAtlas (~5,850 images) | **GRAZPEDWRI-DX** (20,327 paediatric wrist X-rays, 6,091 patients) + FracAtlas |
| Input size | 256 → 224 crop | 448 → 416 crop |
| 16-bit images | n/a | windowed to the 0.5–99.5 percentile before 8-bit |
| Shortcut control | FracAtlas metal hardware dropped | also drops GRAZ images with a **cast or metal**, and "diagnosis uncertain" labels, from training |
| Test set | FracAtlas test (598 images) | **the same 598 images**; v2 is re-scored in the same run |

Data source for GRAZPEDWRI-DX: the "ClearWrist" Hugging Face mirror
(`Layered-Labs/clearwrist-pediatric-wrist-xray`, CC BY 4.0), which keeps the original labels and splits
by patient, so no child appears in both training and test. Figshare often blocks automated downloads,
which is why the script uses this mirror. Broken images are detected and skipped; the count is printed.

**"v3 beats v2" rule, fixed before training.** All three must hold on the FracAtlas test set:
1. v3 AUC > v2 AUC (v2 scored 0.84);
2. at the specificity v2 reaches with its live 0.6 cut-off, v3 catches at least as many fractures;
3. v3 hand-region AUC ≥ v2's.

The script prints `VERDICT: v3 beats v2` or `VERDICT: keep v2` at the end.

## Cost and time (budget ≈ $1.50)

| Step | Expected time |
|---|---|
| Set-up (clone, pip) | ~5 min |
| Data: 23 GB download + decode 20k images | 15–30 min (depends on the host's internet and CPU) |
| Training: 24 epochs | 15–45 min on a 3090 (capped at 3 h by `--max-hours`) |
| Evaluation | ~2 min |
| **Total** | **≈ 1–1.5 h ⇒ about $0.20–0.45 at $0.20–0.30/hr** |

Worst case, if the 3-hour training cap is hit: about 3.5 h, roughly $0.70–1.05. Bandwidth for 23 GB costs
cents if you pick a host with cheap download (see the filters below).

## 1. Rent the instance (vast.ai console)

1. **Account → Keys:** add your SSH public key, if it isn't there yet.
2. **Templates:** choose **PyTorch (Vast)**, the official one, with CUDA 12.x and PyTorch ≥ 2.3.
   Launch mode: **SSH**.
3. **Search filters:**
   - GPU: **1× RTX 3090** (or 1× RTX 4090), **On-demand** (not interruptible)
   - **Disk: 60 GB**
   - CPU: **≥ 8 cores**; RAM: **≥ 32 GB**
   - Internet download: **≥ 500 Mbps**; download cost: **≤ $5/TB**
   - Reliability: **≥ 98 %**
   - Sort by price. A 3090 is usually $0.15–0.30/hr.
4. **Rent.** When it shows *Running*, click **Connect** and copy the SSH command. It looks like
   `ssh -p 41234 root@ssh5.vast.ai`.

## 2. Start the run (inside the instance)

```bash
ssh -p <PORT> root@<HOST>            # from the Connect button
tmux new -s v3                        # keeps running if your SSH connection drops
cd /workspace
git clone https://github.com/Meidverse/COMMRI.git && cd COMMRI
pip install -q pandas pyarrow scikit-learn "huggingface_hub[hf_transfer]" pillow
export HF_HUB_ENABLE_HF_TRANSFER=1
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"   # sanity check
python scripts/train_xray_v3.py 2>&1 | tee -a v3.log
```

- Detach with `Ctrl-b` then `d`; come back with `tmux attach -t v3`. Check progress with `tail -f v3.log`.
- About 1–2 minutes in, the log prints **"label check on first file"**: the fracture/normal counts, and
  cast/metal against fracture. If it says `STOP: …`, the labels are missing, and the run halts before
  spending more. Send me the log in that case.

**If the instance drops or restarts:** reattach (or open a new `tmux`), `cd /workspace/COMMRI`, and run the
same `python scripts/train_xray_v3.py …` line again. Finished downloads, preprocessed files and the last
training epoch are all kept, and it picks up where it stopped. Only a *destroyed* instance loses this.

## 3. Copy the results back (on your PC, PowerShell)

When the log shows `VERDICT: …` and `wrote …/v3_out/…`, run these from `C:\all projects\mri`:

```powershell
scp -P <PORT> root@<HOST>:/workspace/COMMRI/v3_out/xray_fracture_v3.pth .
```
```powershell
scp -P <PORT> root@<HOST>:/workspace/COMMRI/v3_out/xray_v3_metrics.json app/model_meta/
```
```powershell
scp -P <PORT> root@<HOST>:/workspace/COMMRI/v3.log app/model_meta/xray_v3_train.log
```

Check that the files arrived (`xray_fracture_v3.pth` is about 17 MB), then **destroy the instance** in the
vast.ai console. *Stopping* it is not enough, because a stopped instance keeps charging for its disk.

## 4. After the run

- Send me `app/model_meta/xray_v3_metrics.json` or the last lines of the log.
- If the verdict is **"v3 beats v2"**, the next step is wiring v3 into `app/ml.py`: new preprocessing
  (percentile window, 448/416), plus the cut-offs `lo`/`hi` from the metrics file, which were chosen on
  validation data only. Then test it on your father's films and deploy.
- If it says **"keep v2"**, v2 stays live and nothing changes.

## Options

| Flag | Use |
|---|---|
| `--prep-only` | download + preprocess + label check only (CPU is fine), no training |
| `--max-hours 3` | training time cap; evaluation still runs on the best epoch so far |
| `--epochs 24 --batch 64` | training length / batch size |
| `--size 448 --crop 416` | cached image size / crop |
| `--smoke` | tiny local test on generated data (no downloads, CPU, ~30 s) |
