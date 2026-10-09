"""
Aggregate results into tables.   python report.py --work OUT

Writes OUT/report/*.csv and OUT/report/report.md.  Mean +/- std over seeds,
paired deltas (same seed, same classifier, same technique), win counts.

Baselines (identity, augmentation, blur, blur_matched, oracle) are stored once
and joined into every DAE table (one table per DAE architecture / training set).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["auroc", "auprc", "tpr@fpr1e-2", "acc", "sens", "spec"]
COND_ORDER = ["clean_ref", "clean_subset", "oracle", "identity", "augmentation", "blur", "blur_matched",
              "dae_t", "dae_u", "dae_u_matched"]
# (defence, baseline) pairs worth reporting
PAIRS = [("blur", "identity"), ("blur_matched", "augmentation"),
         ("dae_t", "identity"), ("dae_t", "augmentation"), ("dae_t", "blur"),
         ("dae_u", "identity"), ("dae_u", "augmentation"), ("dae_u", "blur"),
         ("dae_u_matched", "augmentation"), ("dae_u_matched", "blur_matched")]
RCOLS = ["mse", "mse_identity", "psnr", "psnr_identity", "ssim", "ssim_identity"]


def fmt(m, s):
    return f"{m:.3f}±{0 if np.isnan(s) else s:.3f}"


def md_table(df):
    cols = [str(c) for c in df.columns]
    head = "| " + " | ".join([df.index.name or ""] + cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |"
            for i, r in zip(df.index, df.values)]
    return "\n".join([head, sep] + rows)


def read_all(work, pattern):
    files = sorted((work / "results").glob(pattern))
    return pd.concat([pd.read_json(p, lines=True) for p in files], ignore_index=True) \
        if files else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    out = work / "report"
    out.mkdir(exist_ok=True)

    df = read_all(work, "raw_*.jsonl")
    if df.empty:
        raise SystemExit("no results yet")
    base = df[df.dae_train == "none"]
    dae = df[df.dae_train != "none"]
    groups = [(dt, ar, pd.concat([base, g], ignore_index=True))
              for (dt, ar), g in dae.groupby(["dae_train", "dae_arch"])] \
        or [("none", "none", base)]

    md = ["# Results\n"]
    deltas, aggs = [], []
    for dtrain, arch, full in groups:
        md.append(f"\n# DAE architecture `{arch}`, DAE trained on `{dtrain}`\n")
        aggs.append(full.assign(table=f"{arch}/{dtrain}"))
        for clf, g in full.groupby("clf"):
            md.append(f"\n## AUROC, classifier `{clf}`\n")
            s = g.groupby(["tech", "condition"])["auroc"].agg(["mean", "std", "count"])
            techs = list(dict.fromkeys(s.index.get_level_values(0)))
            conds = [c for c in COND_ORDER if c in s.index.get_level_values(1)]
            tab = pd.DataFrame({c: {t: fmt(s.loc[(t, c), "mean"], s.loc[(t, c), "std"])
                                    for t in techs if (t, c) in s.index}
                                for c in conds}).reindex(techs).fillna("—")
            tab.index.name = "technique"
            md.append(md_table(tab))
            md.append(f"\n(seeds per cell: {int(s['count'].min())}–{int(s['count'].max())})\n")

        piv = full.pivot_table(index=["clf", "tech", "seed"], columns="condition",
                               values="auroc").reset_index()
        rows = []
        for (clf, tech), g in piv.groupby(["clf", "tech"]):
            if tech == "clean":
                continue
            for cond, bs in PAIRS:
                if cond in g and bs in g:
                    d = (g[cond] - g[bs]).dropna()
                    rows.append({"dae_arch": arch, "dae_train": dtrain, "clf": clf,
                                 "tech": tech, "pair": f"{cond} vs {bs}",
                                 "n_seeds": len(d), "mean": d.mean(),
                                 "std": d.std(ddof=0), "wins": int((d > 0).sum())})
        if rows:
            dl = pd.DataFrame(rows)
            deltas.append(dl)
            md.append("\n## Paired ΔAUROC (positive = first method better; `wins` = seeds won)\n")
            for clf, g in dl.groupby("clf"):
                md.append(f"\n### `{clf}`\n")
                t = g.assign(v=[f"{m:+.3f}±{s:.3f} ({w}/{n})" for m, s, w, n in
                                zip(g["mean"], g["std"], g.wins, g.n_seeds)])
                t = t.pivot_table(index="tech", columns="pair", values="v", aggfunc="first")
                t = t[[f"{c} vs {b}" for c, b in PAIRS if f"{c} vs {b}" in t.columns]]
                t.index.name = "technique"
                md.append(md_table(t.fillna("—")))

    pd.concat(aggs).groupby(["table", "clf", "tech", "condition"])[METRICS].agg(
        ["mean", "std"]).to_csv(out / "classification_mean_std.csv")
    if deltas:
        pd.concat(deltas).to_csv(out / "paired_deltas.csv", index=False)

    rc = read_all(work, "recon_*.jsonl")
    if not rc.empty:
        rc.groupby(["dae_train", "dae_arch", "dae", "tech", "cls"])[RCOLS].mean().to_csv(
            out / "reconstruction.csv")
        md.append("\n# Reconstruction vs the clean image (all test files, mean over seeds)\n")
        md.append("`*_identity` = distance of the obfuscated image itself from clean. "
                  "A DAE should beat both that and the BLUR row; if it does not, it has "
                  "not learned to undo the obfuscation.\n")
        b = rc[rc.dae == "BLUR"]
        if not b.empty:
            t = b[b.cls == "all"].groupby("tech")[RCOLS].mean().round(4)
            t.index.name = "technique"
            md.append("\n### BLUR baseline (non-learned)\n")
            md.append(md_table(t))
        for (dt, ar, dn), g in rc[rc.dae != "BLUR"].groupby(["dae_train", "dae_arch", "dae"]):
            t = g[g.cls == "all"].groupby("tech")[RCOLS].mean().round(4)
            t.index.name = "technique"
            md.append(f"\n### DAE `{dn}`, architecture `{ar}`, trained on `{dt}`\n")
            md.append(md_table(t))

    (out / "report.md").write_text("\n".join(md))
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
