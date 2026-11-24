"""Check the input validation: real scans should pass, non-scans / wrong modality should be rejected."""
import glob
import os
import sys

from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))
import ml  # noqa: E402

MRI = r"C:\all projects\zz_projects\medAI\mri\dataset\classification\test"
XR = r"C:\all projects\zz_projects\medAI\xray\dataset\test"
mri_files = sorted(glob.glob(os.path.join(MRI, "*", "*")))[::2]
xr_files = sorted(glob.glob(os.path.join(XR, "*", "*")))[::2]


def rate(files, mod, transform=None):
    ok = 0
    for f in files:
        im = Image.open(f)
        if transform:
            im = transform(im)
        ok += ml.validate(im, mod)[0]
    return ok, len(files)


print("MRI test scans accepted as MRI:   %d/%d" % rate(mri_files, "mri"))
print("X-ray test images accepted as X-ray: %d/%d" % rate(xr_files, "xray"))
print("X-ray images wrongly accepted as MRI: %d/%d" % rate(xr_files[::4], "mri"))
print("MRI scans wrongly accepted as X-ray: %d/%d" % rate(mri_files[::4], "xray"))
others = sys.argv[1:] + sorted(glob.glob(os.path.join(ROOT, "report_figures", "*.png")))
for f in others:
    im = Image.open(f).convert("RGB")
    for name, x in (("colour", im), ("greyscale", ImageOps.grayscale(im).convert("RGB"))):
        for mod in ("mri", "xray"):
            ok, checks = ml.validate(x, mod)
            print(f"{'ACCEPTED' if ok else 'rejected'}  {mod:4} {name:9} {os.path.basename(f)}  {checks[-1][0]}: {checks[-1][2]}")
