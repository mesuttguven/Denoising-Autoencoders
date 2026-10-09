"""Classification and reconstruction metrics."""
import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve



def clf_metrics(y, s):
    y = np.asarray(y).astype(int)
    out = {"n_test": int(len(y)), "n_malware": int(y.sum())}
    if len(set(y)) < 2:
        return {**out, **{k: float("nan") for k in
                ("auroc", "auprc", "tpr@fpr1e-2", "tpr@fpr1e-3", "acc", "sens", "spec")}}
    fpr, tpr, _ = roc_curve(y, s)
    pred = (s > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {**out,
            "auroc": float(roc_auc_score(y, s)),
            "auprc": float(average_precision_score(y, s)),
            "tpr@fpr1e-2": float(np.interp(1e-2, fpr, tpr)),
            # only meaningful when n_benign_test >> 1000; see n_test
            "tpr@fpr1e-3": float(np.interp(1e-3, fpr, tpr)),
            "acc": float((tp + tn) / len(y)),
            "sens": float(tp / max(tp + fn, 1)),
            "spec": float(tn / max(tn + fp, 1))}


def _psnr(mse):
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-10))


def recon_metrics(store, dae, idx, source, batch=128):
    """
    Per-sample reconstruction quality of dae(source image) against the clean
    image, and of the do-nothing baseline (source image vs clean).
    Returns dict of arrays aligned with sorted(idx) batches.
    """
    mse, ssim, mse0, ssim0, ids = [], [], [], [], []
    idx = np.asarray(idx)
    for s in range(0, len(idx), batch):
        b = np.sort(idx[s:s + batch])
        x = store.arrays[source][b].astype(np.float32) / 255.0
        c = store.arrays["clean"][b].astype(np.float32) / 255.0
        r = np.clip(dae.predict_on_batch(x), 0.0, 1.0).astype(np.float32)
        mse.append(((r - c) ** 2).mean((1, 2, 3)))
        mse0.append(((x - c) ** 2).mean((1, 2, 3)))
        ssim.append(tf.image.ssim(r, c, max_val=1.0).numpy())
        ssim0.append(tf.image.ssim(x, c, max_val=1.0).numpy())
        ids.append(b)
    cat = lambda v: np.concatenate(v) if v else np.zeros(0)
    return {"idx": cat(ids), "mse": cat(mse), "ssim": cat(ssim),
            "mse_identity": cat(mse0), "ssim_identity": cat(ssim0)}


def summarize_recon(rec, labels, which):
    """which: 'benign' | 'malware' | 'all'."""
    lab = labels[rec["idx"]]
    m = np.ones(len(lab), bool) if which == "all" else \
        (lab == (0 if which == "benign" else 1))
    if not m.any():
        return None
    mse, mse0 = rec["mse"][m].mean(), rec["mse_identity"][m].mean()
    return {"cls": which, "n": int(m.sum()),
            "mse": float(mse), "psnr": float(_psnr(mse)),
            "ssim": float(rec["ssim"][m].mean()),
            "mse_identity": float(mse0), "psnr_identity": float(_psnr(mse0)),
            "ssim_identity": float(rec["ssim_identity"][m].mean())}
