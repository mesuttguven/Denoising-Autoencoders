"""
Do Denoising Autoencoders Improve Static Malware Detection Under
Obfuscation?  A controlled evaluation.

  python run_experiment.py build  --benign DIR --malware DIR --work OUT
  python run_experiment.py run    --work OUT --seeds 0 1 2 --clfs cnn1 cnn2
  python report.py                --work OUT

Protocol
--------
* Obfuscation is applied to BOTH classes, so every file has a known clean
  version.  Evaluation is on obfuscated TEST files of both classes.
* Whatever preprocessing a condition uses is applied to EVERY test file.
* Splits (60/20/20, stratified) are drawn per seed over deduplicated files;
  DAEs and classifiers are fit on the train split only.
* Everything trained is cached in OUT/models, so runs are resumable and a DAE
  variant can be added later without retraining the baselines.

Conditions (per obfuscation technique T, per classifier)
--------------------------------------------------------
  clean_ref      trained on clean, tested on CLEAN                (reference)
  identity       trained on clean, tested on obfuscated T         (no defence)
  augmentation   trained on clean+obfuscated mix, NO preprocessing
  blur           trained on clean, tested on blur(obf T)          (non-learned)
  blur_matched   trained on blur(clean+obf mix), tested on blur(obf T)
  dae_t          trained on clean, tested on DAE_T(obf T)   (technique known)
  dae_u          trained on clean, tested on DAE_U(obf T)   (one union DAE)
  dae_u_matched  trained on DAE_U(clean+obf mix), tested on DAE_U(obf T)
  oracle         exact analytic inverse (invertible T only; = clean_ref)

DAE architectures: "paper" (the original) and "unet" (skip connections: the
best case for the DAE).  The DAE is the proposed method; blur and
augmentation are the baselines it has to beat.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from data import Store, build_store, make_split
from metrics import clf_metrics, recon_metrics, summarize_recon
from models import Blur, batch_iter, build_cnn, build_dae, fit, predict_scores


def append(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load(path):
    return tf.keras.models.load_model(path, compile=False)


# ------------------------------------------------------------------- training
def get_dae(a, store, seed, arch, name, sources, dtr, dva):
    path = Path(a.work) / "models" / f"dae_{arch}_{name}_{a.dae_train}_seed{seed}.keras"
    if path.exists():
        return load(path)
    tf.keras.utils.set_random_seed(seed)
    m = build_dae(store.size, arch)
    rng = np.random.default_rng(seed)
    fit(m,
        lambda: batch_iter(store, dtr, sources, a.batch, rng, target="clean"),
        lambda: batch_iter(store, dva, sources, a.batch, np.random.default_rng(999),
                           target="clean", shuffle=False),
        a.dae_epochs, a.patience, tag=f"DAE-{arch}[{name}] seed{seed}")
    m.save(path)
    return m


def get_clf(a, store, kind, seed, cache, tr, va, sources, pre):
    """Train (or load from cache) a classifier; `pre` is applied to its inputs."""
    path = Path(a.work) / "models" / f"clf_{kind}_{cache}_seed{seed}.keras"
    if path.exists():
        return load(path)
    tf.keras.utils.set_random_seed(seed)
    m = build_cnn(kind, store.size)
    rng = np.random.default_rng(seed)
    fit(m,
        lambda: batch_iter(store, tr, sources, a.batch, rng, dae=pre),
        lambda: batch_iter(store, va, sources, a.batch, np.random.default_rng(999),
                           dae=pre, shuffle=False),
        a.clf_epochs, a.patience, tag=f"{kind}[{cache}] seed{seed}")
    m.save(path)
    return m


# ------------------------------------------------------------------- figures
def make_figures(a, store, arch, daes, blur, te):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = store.labels
    for t in store.techs:
        idx = te[store.valid[t][te]]
        ben, mal = idx[labels[idx] == 0][:4], idx[labels[idx] == 1][:4]
        pick = np.concatenate([ben, mal])
        if len(pick) == 0:
            continue
        rows = [("obfuscated", t, None), ("blur baseline", t, blur),
                (f"DAE_T ({arch})", t, daes[t]), (f"DAE_U ({arch})", t, daes["UNION"]),
                ("clean (target)", "clean", None)]
        fig, ax = plt.subplots(len(rows), len(pick),
                               figsize=(1.6 * len(pick), 1.7 * len(rows)), squeeze=False)
        for r, (title, src, model) in enumerate(rows):
            x = store.arrays[src][pick].astype(np.float32) / 255.0
            if model is not None:
                x = np.clip(model.predict_on_batch(x), 0, 1)
            for c in range(len(pick)):
                ax[r, c].imshow(x[c])
                ax[r, c].axis("off")
                if c == 0:
                    ax[r, c].set_title(title, fontsize=8, loc="left")
        for c in range(len(pick)):
            ax[0, c].text(0.5, 1.25, "benign" if labels[pick[c]] == 0 else "malware",
                          transform=ax[0, c].transAxes, ha="center", fontsize=7)
        fig.suptitle(f"{t}", fontsize=9)
        fig.tight_layout()
        fig.savefig(Path(a.work) / "figures" / f"recon_{arch}_{t}.png", dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------------- main
def cmd_run(a):
    work = Path(a.work)
    store = Store(work)
    for d in ("models", "results", "figures"):
        (work / d).mkdir(exist_ok=True)
    res = work / "results"
    raw_base, rec_base = res / "raw_base.jsonl", res / "recon_base.jsonl"
    raw_dae = res / f"raw_{a.dae_train}.jsonl"
    rec_dae = res / f"recon_{a.dae_train}.jsonl"
    labels, techs = store.labels, store.techs
    invertible = set(store.info["invertible"])
    blur = Blur(4)

    for seed in a.seeds:
        split = make_split(labels, seed)
        tr, va, te = (np.where(split == k)[0] for k in (0, 1, 2))
        if a.dae_train == "benign":
            dtr, dva = tr[labels[tr] == 0], va[labels[va] == 0]
        else:
            dtr, dva = tr, va

        # ---- blur reconstruction quality (once per seed)
        flag = res / f"recon_base_seed{seed}.flag"
        if not flag.exists():
            for t in techs:
                te_t = te[store.valid[t][te]]
                r = recon_metrics(store, blur, te_t, t)
                for which in ("benign", "malware", "all"):
                    s = summarize_recon(r, labels, which)
                    if s:
                        append(rec_base, {"seed": seed, "dae_train": "none",
                                          "dae_arch": "none", "dae": "BLUR", "tech": t, **s})
            flag.write_text("ok")

        # ---- DAEs + their reconstruction quality (per architecture)
        daes = {}
        for arch in a.dae_archs:
            d = {t: get_dae(a, store, seed, arch, t, [t], dtr, dva) for t in techs}
            d["UNION"] = get_dae(a, store, seed, arch, "UNION", ["clean"] + techs, dtr, dva)
            daes[arch] = d
            flag = res / f"recon_{a.dae_train}_{arch}_seed{seed}.flag"
            if not flag.exists():
                for t in techs:
                    te_t = te[store.valid[t][te]]
                    for dname, model in (("T", d[t]), ("UNION", d["UNION"])):
                        r = recon_metrics(store, model, te_t, t)
                        for which in ("benign", "malware", "all"):
                            s = summarize_recon(r, labels, which)
                            if s:
                                append(rec_dae, {"seed": seed, "dae_train": a.dae_train,
                                                 "dae_arch": arch, "dae": dname, "tech": t, **s})
                r = recon_metrics(store, d["UNION"], te, "clean")
                for which in ("benign", "malware", "all"):
                    s = summarize_recon(r, labels, which)
                    if s:
                        append(rec_dae, {"seed": seed, "dae_train": a.dae_train,
                                         "dae_arch": arch, "dae": "UNION", "tech": "clean", **s})
                if seed == a.seeds[0]:
                    make_figures(a, store, arch, d, blur, te)
                flag.write_text("ok")

        for kind in a.clfs:
            # ------------------------------------------------ baselines group
            flag = res / f"done_base_seed{seed}_{kind}.flag"
            if not flag.exists():
                clf_clean = get_clf(a, store, kind, seed, "clean", tr, va, ["clean"], None)
                clf_aug = get_clf(a, store, kind, seed, "aug", tr, va, ["clean"] + techs, None)
                clf_bm = get_clf(a, store, kind, seed, "blurmatched", tr, va,
                                 ["clean"] + techs, blur)

                def logb(tech, cond, idx, s):
                    append(raw_base, {"seed": seed, "clf": kind, "tech": tech,
                                      "condition": cond, "dae_train": "none",
                                      "dae_arch": "none", **clf_metrics(labels[idx], s)})

                s_clean = predict_scores(clf_clean, store, te, "clean")
                logb("clean", "clean_ref", te, s_clean)
                logb("clean", "augmentation", te, predict_scores(clf_aug, store, te, "clean"))
                logb("clean", "blur", te, predict_scores(clf_clean, store, te, "clean", dae=blur))
                logb("clean", "blur_matched", te,
                     predict_scores(clf_bm, store, te, "clean", dae=blur))
                for t in techs:
                    mask = store.valid[t][te]
                    idx = te[mask]
                    if len(idx) == 0:
                        continue
                    logb(t, "identity", idx, predict_scores(clf_clean, store, idx, t))
                    logb(t, "augmentation", idx, predict_scores(clf_aug, store, idx, t))
                    logb(t, "blur", idx, predict_scores(clf_clean, store, idx, t, dae=blur))
                    logb(t, "blur_matched", idx, predict_scores(clf_bm, store, idx, t, dae=blur))
                    if t in invertible:
                        logb(t, "oracle", idx, s_clean[mask])
                flag.write_text("ok")
                print(f"seed {seed} / {kind} / baselines complete")

            # ------------------------------------------------- DAE groups
            for arch in a.dae_archs:
                flag = res / f"done_{a.dae_train}_{arch}_seed{seed}_{kind}.flag"
                if flag.exists():
                    continue
                d = daes[arch]
                du = d["UNION"]
                clf_clean = get_clf(a, store, kind, seed, "clean", tr, va, ["clean"], None)
                clf_dm = get_clf(a, store, kind, seed, f"daematched-{arch}-{a.dae_train}",
                                 tr, va, ["clean"] + techs, du)

                def logd(tech, cond, idx, s):
                    append(raw_dae, {"seed": seed, "clf": kind, "tech": tech,
                                     "condition": cond, "dae_train": a.dae_train,
                                     "dae_arch": arch, **clf_metrics(labels[idx], s)})

                logd("clean", "dae_u", te, predict_scores(clf_clean, store, te, "clean", dae=du))
                logd("clean", "dae_u_matched", te,
                     predict_scores(clf_dm, store, te, "clean", dae=du))
                for t in techs:
                    idx = te[store.valid[t][te]]
                    if len(idx) == 0:
                        continue
                    logd(t, "dae_t", idx, predict_scores(clf_clean, store, idx, t, dae=d[t]))
                    logd(t, "dae_u", idx, predict_scores(clf_clean, store, idx, t, dae=du))
                    logd(t, "dae_u_matched", idx, predict_scores(clf_dm, store, idx, t, dae=du))
                flag.write_text("ok")
                print(f"seed {seed} / {kind} / DAE-{arch} complete")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--benign", required=True)
    b.add_argument("--malware", required=True)
    b.add_argument("--work", required=True)
    b.add_argument("--size", type=int, default=128)
    b.add_argument("--resample", default="nearest", choices=["nearest", "bilinear", "area"])
    b.add_argument("--max-per-class", type=int, default=None)
    b.add_argument("--workers", type=int, default=2)
    b.add_argument("--techniques", nargs="*", default=None)
    r = sub.add_parser("run")
    r.add_argument("--work", required=True)
    r.add_argument("--seeds", type=int, nargs="+", default=[0])
    r.add_argument("--clfs", nargs="+", default=["cnn1"])
    r.add_argument("--dae-archs", nargs="+", default=["paper", "unet"],
                   choices=["paper", "unet"])
    r.add_argument("--dae-train", default="benign", choices=["benign", "both"])
    r.add_argument("--dae-epochs", type=int, default=20)
    r.add_argument("--clf-epochs", type=int, default=20)
    r.add_argument("--batch", type=int, default=64)
    r.add_argument("--patience", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "build":
        build_store(a.benign, a.malware, a.work, a.size, a.resample,
                    a.max_per_class, a.workers, a.techniques)
    else:
        cmd_run(a)


if __name__ == "__main__":
    main()
