"""Metadata-only baseline: gradient boosting on file size and byte entropy, same splits as the main experiment.

    python metadata_baseline.py --meta meta.csv

meta.csv is written by `run_experiment.py build` (columns idx,label,sha256,path,bytes,entropy; path is not used).
Model: sklearn HistGradientBoostingClassifier(random_state=0), default settings.
Splits: data.make_split(labels, seed) for seeds 0,1,2, trained on the train part, AUROC on the test part.
"""
import argparse, csv
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from data import make_split

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()
rows = list(csv.DictReader(open(a.meta)))
y = np.array([int(r["label"]) for r in rows])
size = np.array([float(r["bytes"]) for r in rows])
ent = np.array([float(r["entropy"]) for r in rows])
for name, X in (("file size", size[:, None]), ("byte entropy", ent[:, None]), ("size + entropy", np.c_[size, ent])):
    s = []
    for seed in a.seeds:
        sp = make_split(y, seed)
        tr, te = sp == 0, sp == 2
        m = HistGradientBoostingClassifier(random_state=0).fit(X[tr], y[tr])
        s.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    print(f"{name:15s} AUROC {np.mean(s):.3f} +- {np.std(s):.3f}   per seed {[round(x, 3) for x in s]}")
