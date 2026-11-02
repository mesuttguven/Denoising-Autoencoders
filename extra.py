"""
Additional analyses added for the revision (reviewer requests).

Every sub-command takes --work, the folder of a finished run (the folder that
holds store.json, the image arrays, meta.csv and models/).  Nothing here
changes the main experiment; results go to <work>/results/.

  scores    save the per-file test score of every condition (uses the cached
            models; --allow-train retrains a missing one)
  stats     paired bootstrap over test files: confidence interval of the AUROC
            difference between two defences, and an equivalence check
  matched   size-controlled evaluation: AUROC inside strata of equal file size
            (and size x entropy), for the CNNs and for a metadata-only model
  neardup   how similar are the test images to the training images, and how do
            the results change when near-duplicates are removed
  loo       leave-one-obfuscation-out: train on all types except one, test on
            the held-out type (unseen transformation)
  sens      sensitivity of the DAE result to learning rate, width, loss,
            training length and training mixture (validation split only)

Typical order:  scores -> stats -> matched -> neardup ;  loo and sens train
new models (the slow part).
"""
import argparse
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from data import Store, make_split
from metrics import clf_metrics
from models import Blur, batch_iter, build_cnn, build_dae, fit, predict_scores

MAIN_TECHS = ["Code_Packing", "Control_Flow_Obfuscation", "Data_Obfuscation",
              "Encryption", "String_Obfuscation"]          # the five invertible types
COMPS = [("dae_u_matched", "augmentation"), ("dae_t", "augmentation"),
         ("blur_matched", "augmentation"), ("dae_u_matched", "blur_matched")]
MARGINS = (0.01, 0.02, 0.03)


# ------------------------------------------------------------------ helpers
def ns(a, **kw):
    """Namespace with the training settings of the main run."""
    d = dict(work=a.work, dae_train="benign", dae_epochs=a.dae_epochs,
             clf_epochs=a.clf_epochs, batch=a.batch, patience=4)
    d.update(kw)
    return SimpleNamespace(**d)


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def splits(labels, seed):
    sp = make_split(labels, seed)
    return tuple(np.where(sp == k)[0] for k in (0, 1, 2))


class WAuc:
    """AUROC with case weights and exact tie handling (fast: one sort per set)."""
    def __init__(self, s, y):
        self.y = np.asarray(y).astype(bool)
        u, self.inv = np.unique(np.asarray(s, np.float64), return_inverse=True)
        self.k = len(u)

    def __call__(self, w=None):
        w = np.ones(len(self.y)) if w is None else w
        wp = np.bincount(self.inv, weights=w * self.y, minlength=self.k)
        wn = np.bincount(self.inv, weights=w * ~self.y, minlength=self.k)
        den = wp.sum() * wn.sum()
        if den <= 0:
            return float("nan")
        cn = np.cumsum(wn) - wn
        return float((wp * (cn + 0.5 * wn)).sum() / den)


def clf_path(a, kind, seed, cache):
    return Path(a.work) / "models" / f"clf_{kind}_{cache}_seed{seed}.keras"


def dae_path(a, arch, seed, name):
    return Path(a.work) / "models" / f"dae_{arch}_{name}_benign_seed{seed}.keras"


def check_cached(a, path):
    if not path.exists() and not getattr(a, "allow_train", False):
        raise SystemExit(f"missing cached model {path.name}; copy the models/ folder of the "
                         f"main run into --work, or pass --allow-train to retrain it")


def load_scores(res, tag):
    f = Path(res) / tag
    return dict(np.load(f)) if f.exists() else None


# ------------------------------------------------------------------- scores
def cmd_scores(a):
    from run_experiment import get_clf, get_dae
    store = Store(a.work)
    labels, techs = store.labels, store.techs
    invertible = set(store.info["invertible"])
    res = Path(a.work) / "results"
    res.mkdir(exist_ok=True)
    blur = Blur(4)
    cfg = ns(a)
    for seed in a.seeds:
        tr, va, te = splits(labels, seed)
        dtr, dva = tr[labels[tr] == 0], va[labels[va] == 0]
        for kind in a.clfs:
            for arch in a.dae_archs:
                out = res / f"scores_{arch}_seed{seed}_{kind}.npz"
                if out.exists():
                    continue
                paths = [clf_path(a, kind, seed, c) for c in
                         ("clean", "aug", "blurmatched", f"daematched-{arch}-benign")]
                paths += [dae_path(a, arch, seed, n) for n in techs + ["UNION"]]
                for p in paths:
                    check_cached(a, p)
                clf_clean = get_clf(cfg, store, kind, seed, "clean", tr, va, ["clean"], None)
                clf_aug = get_clf(cfg, store, kind, seed, "aug", tr, va, ["clean"] + techs, None)
                clf_bm = get_clf(cfg, store, kind, seed, "blurmatched", tr, va, ["clean"] + techs, blur)
                d = {t: get_dae(cfg, store, seed, arch, t, [t], dtr, dva) for t in techs}
                du = get_dae(cfg, store, seed, arch, "UNION", ["clean"] + techs, dtr, dva)
                clf_dm = get_clf(cfg, store, kind, seed, f"daematched-{arch}-benign", tr, va,
                                 ["clean"] + techs, du)
                S = {}

                def put(cond, tech, idx, s):
                    S[f"{cond}|{tech}"] = np.asarray(s, np.float32)
                    S[f"idx|{tech}"] = np.asarray(idx)

                s_clean = predict_scores(clf_clean, store, te, "clean")
                put("clean_ref", "clean", te, s_clean)
                put("augmentation", "clean", te, predict_scores(clf_aug, store, te, "clean"))
                put("blur_matched", "clean", te, predict_scores(clf_bm, store, te, "clean", dae=blur))
                put("dae_u", "clean", te, predict_scores(clf_clean, store, te, "clean", dae=du))
                put("dae_u_matched", "clean", te, predict_scores(clf_dm, store, te, "clean", dae=du))
                for t in techs:
                    mask = store.valid[t][te]
                    idx = te[mask]
                    if len(idx) == 0:
                        continue
                    put("identity", t, idx, predict_scores(clf_clean, store, idx, t))
                    put("augmentation", t, idx, predict_scores(clf_aug, store, idx, t))
                    put("blur_matched", t, idx, predict_scores(clf_bm, store, idx, t, dae=blur))
                    put("dae_t", t, idx, predict_scores(clf_clean, store, idx, t, dae=d[t]))
                    put("dae_u", t, idx, predict_scores(clf_clean, store, idx, t, dae=du))
                    put("dae_u_matched", t, idx, predict_scores(clf_dm, store, idx, t, dae=du))
                    put("oracle" if t in invertible else "clean_subset", t, idx, s_clean[mask])
                np.savez_compressed(out, **S)
                print("saved", out.name, flush=True)


# -------------------------------------------------------------------- stats
def bootstrap_compare(sets, labels, splits_te, techs, comps, B, rng):
    """
    sets      : {seed: {key: array}}   (scores files)
    returns rows (list of dict).  The bootstrap resamples test files
    (stratified by class) with the SAME draw for every condition, so the
    paired structure is kept; the statistic is the mean over seeds of the
    AUROC difference.
    """
    seeds = sorted(sets)
    conds = sorted({c for ab in comps for c in ab})
    objs = {}
    for sd in seeds:
        te = splits_te[sd]
        for t in techs:
            for c in conds:
                k = f"{c}|{t}"
                if k in sets[sd]:
                    idx = sets[sd][f"idx|{t}"]
                    objs[(sd, t, c)] = (WAuc(sets[sd][k], labels[idx]), np.searchsorted(te, idx))
    point = {}
    boot = {}
    for t, (ca, cb) in itertools.product(techs, comps):
        if not all((sd, t, c) in objs for sd in seeds for c in (ca, cb)):
            continue
        d = np.array([objs[(sd, t, ca)][0]() - objs[(sd, t, cb)][0]() for sd in seeds])
        point[(t, ca, cb)] = d
        boot[(t, ca, cb)] = np.zeros((B, len(seeds)))
    for b in range(B):
        for si, sd in enumerate(seeds):
            te = splits_te[sd]
            w = np.zeros(len(te))
            for cls in (0, 1):
                pos = np.where(labels[te] == cls)[0]
                if len(pos):
                    w[pos] = rng.multinomial(len(pos), np.full(len(pos), 1.0 / len(pos)))
            for (t, ca, cb), arr in boot.items():
                (oa, pa), (ob, pb) = objs[(sd, t, ca)], objs[(sd, t, cb)]
                arr[b, si] = oa(w[pa]) - ob(w[pb])
    rows = []

    def summarise(name, d, bs, ca, cb):
        m = bs.mean(1)
        lo, hi = np.percentile(m, [2.5, 97.5])
        lo90, hi90 = np.percentile(m, [5, 95])
        r = {"tech": name, "comparison": f"{ca} - {cb}", "n_seeds": len(seeds),
             "mean_delta": float(d.mean()), "seed_sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
             "seeds_won": int((d > 0).sum()), "ci95_lo": float(lo), "ci95_hi": float(hi),
             "ci90_lo": float(lo90), "ci90_hi": float(hi90)}
        for mg in MARGINS:
            r[f"equiv_{mg}"] = bool(lo90 > -mg and hi90 < mg)
        rows.append(r)

    for (t, ca, cb), d in point.items():
        summarise(t, d, boot[(t, ca, cb)], ca, cb)
    for ca, cb in comps:
        ts = [t for t in MAIN_TECHS if (t, ca, cb) in point]
        if len(ts) >= 2:
            d = np.mean([point[(t, ca, cb)] for t in ts], 0)
            bs = np.mean([boot[(t, ca, cb)] for t in ts], 0)
            summarise(f"MEAN_{len(ts)}_types", d, bs, ca, cb)
    return rows


def write_csv(rows, path):
    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def cmd_stats(a):
    store = Store(a.work)
    labels = store.labels
    res = Path(a.work) / "results"
    rng = np.random.default_rng(2026)
    allrows = []
    tes = {sd: splits(labels, sd)[2] for sd in a.seeds}
    for arch in a.dae_archs:
        for kind in a.clfs:
            if a.loo:
                sets = {}
                for sd in a.seeds:
                    S = {}
                    for f in sorted(res.glob(f"loo_*_seed{sd}_{kind}.npz")):
                        S.update(dict(np.load(f)))
                    if S:
                        sets[sd] = S
                comps = [("dae_u_matched_loo", "augmentation_loo"), ("dae_u_loo", "identity")]
                techs = sorted({k.split("|")[1] for s in sets.values() for k in s if not k.startswith("idx")})
            else:
                sets = {sd: load_scores(res, f"scores_{arch}_seed{sd}_{kind}.npz") for sd in a.seeds}
                sets = {k: v for k, v in sets.items() if v}
                comps, techs = COMPS, store.techs
            if len(sets) < 1:
                print("no scores for", arch, kind)
                continue
            rows = bootstrap_compare(sets, labels, tes, techs, comps, a.boot, rng)
            for r in rows:
                r.update(arch=arch, clf=kind)
            allrows += rows
            if a.loo:
                break
    name = "paired_bootstrap_loo.csv" if a.loo else "paired_bootstrap.csv"
    write_csv(allrows, res / name)
    print(f"wrote {res / name}  ({len(allrows)} rows; B={a.boot})")
    print("\nDAE (union, matched) minus augmentation; 95% CI over test files, mean of seeds")
    print("| arch | clf | transformation | mean | 95% CI | seeds won | equiv. +-0.02 |")
    print("|---|---|---|---|---|---|---|")
    for r in allrows:
        if r["comparison"].startswith(("dae_u_matched - augmentation", "dae_u_matched_loo")):
            print(f"| {r['arch']} | {r['clf']} | {r['tech']} | {r['mean_delta']:+.3f} | "
                  f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}] | {r['seeds_won']}/{r['n_seeds']} | "
                  f"{'yes' if r['equiv_0.02'] else 'no'} |")


# ------------------------------------------------------------------ matched
def strata_ids(meta, variant, ref):
    """Integer stratum id per file from quantile bins of size (and entropy)."""
    ls = np.log10(meta["bytes"].to_numpy(float))
    en = meta["entropy"].to_numpy(float)
    ref_ls, ref_en = np.log10(ref["bytes"].to_numpy(float)), ref["entropy"].to_numpy(float)
    q = lambda v, r, k: np.clip(np.searchsorted(np.quantile(r, np.linspace(0, 1, k + 1)[1:-1]), v), 0, k - 1)
    if variant == "size10":
        return q(ls, ref_ls, 10)
    if variant == "size_entropy":
        return q(ls, ref_ls, 6) * 6 + q(en, ref_en, 6)
    raise ValueError(variant)


def stratified_auc(y, s, strata):
    """Pairs-weighted mean of within-stratum AUROCs (Mantel-Haenszel style)."""
    num = den = 0.0
    used = 0
    for g in np.unique(strata):
        m = strata == g
        npos, nneg = int(y[m].sum()), int((~y[m].astype(bool)).sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * WAuc(s[m], y[m])()
        den += w
        used += int(m.sum())
    return (num / den if den else float("nan")), used


def cmd_matched(a):
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    store = Store(a.work)
    labels = store.labels
    res = Path(a.work) / "results"
    meta = pd.read_csv(Path(a.work) / "meta.csv")
    # --min-bytes N: keep only files with at least N bytes.  With the fixed
    # byte budget (N = size*size*3) these files fill the whole image, so no
    # padding can reveal their length.
    big = meta["bytes"].to_numpy() >= a.min_bytes
    rows = []
    conds = ["clean_ref", "identity", "augmentation", "dae_t", "dae_u_matched", "oracle", "clean_subset"]
    for seed in a.seeds:
        tr, va, te = splits(labels, seed)
        te = te[big[te]]
        meta_models = {}
        for name, cols in (("meta_size", ["bytes"]), ("meta_size_entropy", ["bytes", "entropy"])):
            m = HistGradientBoostingClassifier(random_state=0).fit(meta.loc[tr, cols], labels[tr])
            meta_models[name] = (cols, m)
        for variant in ("none", "size10", "size_entropy"):
            st = None if variant == "none" else strata_ids(meta, variant, meta.loc[tr])
            for name, (cols, m) in meta_models.items():
                s = m.predict_proba(meta.loc[te, cols])[:, 1]
                y = labels[te]
                auc, used = (WAuc(s, y)(), len(te)) if st is None else stratified_auc(y, s, st[te])
                rows.append(dict(seed=seed, arch="-", clf="-", tech="clean", cond=name,
                                 variant=variant, auroc=auc, n_used=used))
            for arch in a.dae_archs:
                for kind in a.clfs:
                    S = load_scores(res, f"scores_{arch}_seed{seed}_{kind}.npz")
                    if S is None:
                        continue
                    for key in S:
                        if key.startswith("idx|"):
                            continue
                        cond, tech = key.split("|")
                        if cond not in conds:
                            continue
                        idx = S[f"idx|{tech}"]
                        keep = big[idx]
                        idx, sc = idx[keep], S[key][keep]
                        y = labels[idx]
                        auc, used = (WAuc(sc, y)(), len(idx)) if st is None else \
                            stratified_auc(y, sc, st[idx])
                        rows.append(dict(seed=seed, arch=arch, clf=kind, tech=tech, cond=cond,
                                         variant=variant, auroc=auc, n_used=used))
    write_csv(rows, res / "matched_eval.csv")
    df = pd.DataFrame(rows)
    print("mean over seeds; 'none' = ordinary AUROC, size10 / size_entropy = inside strata of equal size (and entropy)")
    t = df[(df.tech == "clean") & df.cond.isin(["clean_ref", "meta_size", "meta_size_entropy"])]
    print(t.pivot_table(index=["arch", "clf", "cond"], columns="variant", values="auroc").round(3).to_string())
    print("\nobfuscated types, U-Net, strata of size x entropy:")
    t = df[(df.arch == "unet") & (df.variant == "size_entropy") & df.cond.isin(["identity", "augmentation", "dae_u_matched"])]
    print(t.pivot_table(index=["clf", "tech"], columns="cond", values="auroc").round(3).to_string())
    print("wrote", res / "matched_eval.csv")


# ----------------------------------------------------------------- neardup
def pooled_vectors(store, k=16, chunk=500):
    n = store.info["n"]
    out = np.zeros((n, k * k * 3), np.float32)
    f = store.size // k
    for s in range(0, n, chunk):
        x = store.arrays["clean"][s:s + chunk].astype(np.float32) / 255.0
        x = x.reshape(len(x), k, f, k, f, 3).mean((2, 4))
        out[s:s + chunk] = x.reshape(len(x), -1)
    out -= out.mean(0, keepdims=True)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)
    return out


def cmd_neardup(a):
    import pandas as pd
    store = Store(a.work)
    labels = store.labels
    res = Path(a.work) / "results"
    V = pooled_vectors(store)
    rows = []
    for seed in a.seeds:
        tr, va, te = splits(labels, seed)
        sim = V[te] @ V[tr].T
        j = sim.argmax(1)
        smax = sim[np.arange(len(te)), j]
        nn_label = labels[tr][j]
        np.savez_compressed(res / f"neardup_seed{seed}.npz", te=te, smax=smax, nn_label=nn_label)
        for thr in a.thr:
            near = smax >= thr
            rows.append(dict(seed=seed, thr=thr, share_near=float(near.mean()),
                             nn_label_agrees_near=float((nn_label[near] == labels[te][near]).mean()) if near.any() else float("nan"),
                             nn_label_agrees_far=float((nn_label[~near] == labels[te][~near]).mean()) if (~near).any() else float("nan")))
        for arch in a.dae_archs:
            for kind in a.clfs:
                S = load_scores(res, f"scores_{arch}_seed{seed}_{kind}.npz")
                if S is None:
                    continue
                for thr in a.thr:
                    keep = smax < thr                      # drop test files with a near-duplicate in train
                    kept_files = te[keep]
                    for cond in ("clean_ref", "augmentation", "dae_u_matched"):
                        s_all = S[f"{cond}|clean"]
                        y = labels[te]
                        rows.append(dict(seed=seed, arch=arch, clf=kind, thr=thr, cond=cond,
                                         n_kept=int(keep.sum()),
                                         auroc_all=WAuc(s_all, y)(),
                                         auroc_dedup=WAuc(s_all[keep], y[keep])() if keep.sum() > 10 else float("nan")))
    write_csv(rows, res / "neardup.csv")
    df = pd.DataFrame(rows)
    print(df[df.cond.isna()].groupby("thr")[["share_near", "nn_label_agrees_near", "nn_label_agrees_far"]].mean().round(3).to_string())
    if "cond" in df and df.cond.notna().any():
        print(df[df.cond.notna()].groupby(["arch", "clf", "cond", "thr"])[["n_kept", "auroc_all", "auroc_dedup"]].mean().round(3).to_string())
    print("wrote", res / "neardup.csv")


# --------------------------------------------------------------------- loo
def cmd_loo(a):
    from run_experiment import get_clf, get_dae
    store = Store(a.work)
    labels, techs = store.labels, store.techs
    res = Path(a.work) / "results"
    res.mkdir(exist_ok=True)
    cfg = ns(a)
    held = a.held or [t for t in MAIN_TECHS if t in techs]
    raw = res / "raw_loo.jsonl"
    for seed in a.seeds:
        tr, va, te = splits(labels, seed)
        dtr, dva = tr[labels[tr] == 0], va[labels[va] == 0]
        for h in held:
            mix = ["clean"] + [t for t in techs if t != h]
            du = None
            for kind in a.clfs:
                out = res / f"loo_{h}_seed{seed}_{kind}.npz"
                if out.exists():
                    continue
                if du is None:
                    du = get_dae(cfg, store, seed, "unet", f"LOO-{h}", mix, dtr, dva)
                clf_clean = get_clf(cfg, store, kind, seed, "clean", tr, va, ["clean"], None)
                clf_aug = get_clf(cfg, store, kind, seed, f"loo-{h}-aug", tr, va, mix, None)
                clf_dm = get_clf(cfg, store, kind, seed, f"loo-{h}-daematched", tr, va, mix, du)
                idx = te[store.valid[h][te]]
                S = {f"idx|{h}": idx}
                for cond, (clf, dae) in {"identity": (clf_clean, None),
                                         "augmentation_loo": (clf_aug, None),
                                         "dae_u_loo": (clf_clean, du),
                                         "dae_u_matched_loo": (clf_dm, du)}.items():
                    s = predict_scores(clf, store, idx, h, dae=dae)
                    S[f"{cond}|{h}"] = s.astype(np.float32)
                    append(raw, {"seed": seed, "clf": kind, "held_out": h, "condition": cond,
                                 **clf_metrics(labels[idx], s)})
                np.savez_compressed(out, **S)
                print(f"seed {seed} / {kind} / held out {h} done", flush=True)


# -------------------------------------------------------------------- sens
VARIANTS = {
    "default": {},
    "lr_3e-4": {"lr": 3e-4},
    "lr_3e-3": {"lr": 3e-3},
    "width_0.5": {"width": 0.5},
    "width_2": {"width": 2.0},
    "loss_mae": {"loss": "mae"},
    "epochs_5": {"epochs": 5},
    "epochs_40": {"epochs": 40, "patience": 8},
    "mix_clean_0.5": {"p_clean": 0.5},
}


def cmd_sens(a):
    store = Store(a.work)
    labels, techs = store.labels, store.techs
    res = Path(a.work) / "results"
    res.mkdir(exist_ok=True)
    mdir = Path(a.work) / "models_sens"
    mdir.mkdir(exist_ok=True)
    out_csv = res / "sens_raw.jsonl"
    names = list(VARIANTS) if a.variants == ["all"] else a.variants
    kind = a.clfs[0]
    for seed in a.seeds:
        tr, va, te = splits(labels, seed)
        dtr, dva = tr[labels[tr] == 0], va[labels[va] == 0]
        mix = ["clean"] + techs
        for name in names:
            v = VARIANTS[name]
            done = res / f"sens_{name}_seed{seed}.flag"
            if done.exists():
                continue
            pc = v.get("p_clean")
            ep, pat = v.get("epochs", a.dae_epochs), v.get("patience", 4)
            import tensorflow as tf

            def get(path, build, train_it, val_it, epochs, patience, tag):
                if path.exists():
                    return tf.keras.models.load_model(path)
                tf.keras.utils.set_random_seed(seed)
                m = build()
                fit(m, train_it, val_it, epochs, patience, tag=tag)
                m.save(path)
                return m

            rng = np.random.default_rng(seed)
            du = get(mdir / f"dae_{name}_seed{seed}.keras",
                     lambda: build_dae(store.size, "unet", v.get("width", 1.0), v.get("lr"), v.get("loss", "mse")),
                     lambda: batch_iter(store, dtr, mix, a.batch, rng, target="clean", p_clean=pc),
                     lambda: batch_iter(store, dva, mix, a.batch, np.random.default_rng(999),
                                        target="clean", shuffle=False, p_clean=pc),
                     ep, pat, f"sens-DAE[{name}] seed{seed}")
            rng2 = np.random.default_rng(seed)
            clf_dm = get(mdir / f"clf_{kind}_dm_{name}_seed{seed}.keras",
                         lambda: build_cnn(kind, store.size),
                         lambda: batch_iter(store, tr, mix, a.batch, rng2, dae=du, p_clean=pc),
                         lambda: batch_iter(store, va, mix, a.batch, np.random.default_rng(999),
                                            dae=du, shuffle=False, p_clean=pc),
                         a.clf_epochs, 4, f"sens-{kind}-matched[{name}] seed{seed}")
            rng3 = np.random.default_rng(seed)
            clf_aug = get(mdir / f"clf_{kind}_aug_{name if pc is not None else 'default'}_seed{seed}.keras",
                          lambda: build_cnn(kind, store.size),
                          lambda: batch_iter(store, tr, mix, a.batch, rng3, p_clean=pc),
                          lambda: batch_iter(store, va, mix, a.batch, np.random.default_rng(999),
                                             shuffle=False, p_clean=pc),
                          a.clf_epochs, 4, f"sens-{kind}-aug[{name}] seed{seed}")
            for t in techs:           # evaluation on the VALIDATION split only
                idx = va[store.valid[t][va]]
                s_dm = predict_scores(clf_dm, store, idx, t, dae=du)
                s_aug = predict_scores(clf_aug, store, idx, t)
                m_dm, m_aug = clf_metrics(labels[idx], s_dm), clf_metrics(labels[idx], s_aug)
                append(out_csv, {"variant": name, "seed": seed, "clf": kind, "tech": t,
                                 "auroc_dae_matched": m_dm["auroc"], "auroc_augmentation": m_aug["auroc"],
                                 "delta": m_dm["auroc"] - m_aug["auroc"]})
            done.write_text("ok")
            print(f"sens {name} seed {seed} done", flush=True)
    summarize_sens(res)


def summarize_sens(res):
    import pandas as pd
    f = Path(res) / "sens_raw.jsonl"
    if not f.exists():
        return
    df = pd.read_json(f, lines=True)
    main = df[df.tech.isin(MAIN_TECHS)]
    t = main.groupby("variant")[["auroc_dae_matched", "auroc_augmentation", "delta"]].mean().round(3)
    t["min_delta"] = main.groupby(["variant", "tech"])["delta"].mean().groupby("variant").min().round(3)
    t["max_delta"] = main.groupby(["variant", "tech"])["delta"].mean().groupby("variant").max().round(3)
    print("validation split; mean over the five invertible types and seeds")
    print(t.to_string())
    t.to_csv(Path(res) / "sens_summary.csv")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--work", required=True)
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        p.add_argument("--clfs", nargs="+", default=["cnn1", "cnn2"])
        p.add_argument("--dae-archs", nargs="+", default=["unet"], choices=["paper", "unet"])
        p.add_argument("--batch", type=int, default=64)
        p.add_argument("--dae-epochs", type=int, default=20)
        p.add_argument("--clf-epochs", type=int, default=20)
        p.add_argument("--allow-train", action="store_true")
        return p
    common(sub.add_parser("scores"))
    p = common(sub.add_parser("stats"))
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--loo", action="store_true", help="use the leave-one-out score files")
    p = common(sub.add_parser("matched"))
    p.add_argument("--min-bytes", type=int, default=0, help="keep only files of at least this size")
    p = common(sub.add_parser("neardup"))
    p.add_argument("--thr", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    p = common(sub.add_parser("loo"))
    p.add_argument("--held", nargs="*", default=None)
    p = common(sub.add_parser("sens"))
    p.add_argument("--variants", nargs="+", default=["all"], choices=["all"] + list(VARIANTS))
    a = ap.parse_args()
    {"scores": cmd_scores, "stats": cmd_stats, "matched": cmd_matched, "neardup": cmd_neardup,
     "loo": cmd_loo, "sens": cmd_sens}[a.cmd](a)


if __name__ == "__main__":
    main()