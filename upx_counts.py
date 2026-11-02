"""
How many files did UPX accept, per class?  (Reviewer request: report the
number and class distribution of packed files, the acceptance rates, and check
whether the refused files differ from the accepted ones.)

  python upx_counts.py --meta meta.csv --valid valid_UPX.npy [--upx-version "upx 4.2.2"]

meta.csv is written by the build step; valid_UPX.npy by the UPX run (a
boolean array, True where UPX produced a packed file).  Both are in the
results folder of the UPX run (dae_study_results_upx on Drive).
"""
import argparse

import numpy as np
import pandas as pd

from data import make_split

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--valid", required=True)
ap.add_argument("--upx-version", default="")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()

m = pd.read_csv(a.meta)
v = np.load(a.valid).astype(bool)
assert len(m) == len(v), (len(m), len(v))
m["accepted"] = v
m["class"] = m["label"].map({0: "benign", 1: "malware"})

print(f"UPX version: {a.upx_version or '(not given)'}\n")
print("| Class | Files | Packed by UPX | Acceptance rate |")
print("|---|---|---|---|")
for c, g in m.groupby("class"):
    print(f"| {c} | {len(g)} | {int(g.accepted.sum())} | {100 * g.accepted.mean():.1f}% |")
print(f"| all | {len(m)} | {int(v.sum())} | {100 * v.mean():.1f}% |")

acc = m[m.accepted]
print(f"\nShare of malware among packed files: {100 * acc.label.mean():.1f}% "
      f"(whole set: {100 * m.label.mean():.1f}%)")

print("\nAccepted against refused files (median):")
print("| Class | Group | n | Size (bytes) | Entropy |")
print("|---|---|---|---|---|")
for c in ("benign", "malware"):
    for name, g in (("accepted", m[(m["class"] == c) & m.accepted]),
                    ("refused", m[(m["class"] == c) & ~m.accepted])):
        if len(g):
            print(f"| {c} | {name} | {len(g)} | {int(g.bytes.median()):,} | {g.entropy.median():.2f} |")

print("\nPacked files in the test split of each seed:")
print("| Seed | Test files | Packed benign | Packed malware |")
print("|---|---|---|---|")
labels = m.label.to_numpy()
for sd in a.seeds:
    te = np.where(make_split(labels, sd) == 2)[0]
    k = te[v[te]]
    print(f"| {sd} | {len(te)} | {int((labels[k] == 0).sum())} | {int((labels[k] == 1).sum())} |")