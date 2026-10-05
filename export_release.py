"""Package the image version of the full dataset for public release (no raw binaries, no file paths).

Run it after the images have been built (the folder that holds clean.npy, Encryption.npy, ... meta.csv):

    python export_release.py --work /content/work_full --out /content/drive/MyDrive/trapmine_images_release

Output (everything is derived, nothing executable):
  metadata.csv        idx, label (0 benign, 1 malware), sha256, bytes, entropy   (file paths removed)
  images_<name>.npz   uint8 array (N,128,128,3) for 'clean' and for each obfuscation type, rows in metadata order
  valid_<type>.npy    which rows the obfuscation was valid for
  store.json          image size, resampling, list of techniques
  README.txt          how to read the files
"""
import argparse, csv, shutil
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
work, out = Path(a.work), Path(a.out)
out.mkdir(parents=True, exist_ok=True)

with open(work / "meta.csv", newline="") as f, open(out / "metadata.csv", "w", newline="") as g:
    r, w = csv.DictReader(f), csv.writer(g)
    w.writerow(["idx", "label", "sha256", "bytes", "entropy"])
    n = 0
    for row in r:
        w.writerow([row["idx"], row["label"], row["sha256"], row["bytes"], row["entropy"]])
        n += 1
print("metadata rows:", n)

for p in sorted(work.glob("*.npy")):
    if p.name.startswith("valid_") or p.name == "labels.npy":
        shutil.copy(p, out / p.name)
    else:
        arr = np.load(p, mmap_mode="r")
        np.savez_compressed(out / f"images_{p.stem}.npz", images=arr)
        print("wrote", f"images_{p.stem}.npz", arr.shape)
shutil.copy(work / "store.json", out / "store.json")
(out / "README.txt").write_text(
    "Image version of the TRAPMINE files used in the study.\n"
    "images_<name>.npz -> key 'images', uint8 (N,128,128,3); row i matches row i of metadata.csv.\n"
    "'clean' = unmodified file; other names = the six byte-level obfuscations applied before imaging.\n"
    "label: 0 benign, 1 malware. Raw binaries and file paths are NOT included.\n"
    "Train/validation/test splits are generated from the labels with seeds 0,1,2 by the code (data.py).\n")
print("done ->", out)
