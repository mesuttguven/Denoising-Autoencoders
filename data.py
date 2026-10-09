"""
Dataset construction.

Order of operations (this order is the point):
  1. hash every file, drop exact duplicates, drop hashes present in both
     classes (label noise);
  2. build ONE image store holding, for every file, the clean image and the
     image of every obfuscated version -- obfuscation is applied to BOTH
     classes, so every sample has a known clean counterpart;
  3. splits are drawn later, per seed, over file indices.  Nothing is fit
     before the split exists, so nothing can leak across it.
"""
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from imaging import bytes_to_rgb, byte_entropy
from obfuscate import TECHNIQUES


def sha256_file(p, bs=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(bs):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(benign_dir, malware_dir, max_per_class=None):
    rows = []
    for d, label in ((benign_dir, 0), (malware_dir, 1)):
        for p in sorted(Path(d).rglob("*")):      # nested folders are fine
            if p.is_file() and p.stat().st_size > 0:
                rows.append({"path": str(p), "label": label,
                             "sha": sha256_file(p)})
    by_sha = {}
    for r in rows:
        by_sha.setdefault(r["sha"], []).append(r)
    keep, stats = [], {"files_seen": len(rows), "within_class_duplicates_dropped": 0,
                       "cross_class_files_dropped": 0}
    for rs in by_sha.values():
        if len({r["label"] for r in rs}) > 1:
            stats["cross_class_files_dropped"] += len(rs)
            continue
        keep.append(rs[0])
        stats["within_class_duplicates_dropped"] += len(rs) - 1
    keep.sort(key=lambda r: (r["label"], r["path"]))
    if max_per_class:
        rng = np.random.default_rng(12345)
        sel = []
        for label in (0, 1):
            cls = [r for r in keep if r["label"] == label]
            take = rng.permutation(len(cls))[:max_per_class]
            sel += [cls[i] for i in sorted(take)]
        keep = sel
    stats["benign"] = sum(r["label"] == 0 for r in keep)
    stats["malware"] = sum(r["label"] == 1 for r in keep)
    return keep, stats


def make_split(labels, seed):
    """0=train 60%, 1=val 20%, 2=test 20%, stratified by class."""
    idx = np.arange(len(labels))
    tr, rest = train_test_split(idx, test_size=0.4, random_state=seed,
                                stratify=labels)
    va, te = train_test_split(rest, test_size=0.5, random_state=seed,
                              stratify=labels[rest])
    split = np.zeros(len(labels), dtype=np.int8)
    split[va], split[te] = 1, 2
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) \
        and not (set(va) & set(te)), "split overlap"
    return split


def _process(args):
    i, path, size, resample, techs = args
    b = Path(path).read_bytes()
    out = {"clean": bytes_to_rgb(b, size, resample)}
    valid = {}
    for name in techs:
        t = TECHNIQUES[name]
        try:
            ob = t.apply(b)
        except Exception:
            out[name] = np.zeros((size, size, 3), np.uint8)
            valid[name] = False
            continue
        if t.invertible and t.invert(ob) != b:
            raise RuntimeError(f"{name} does not round-trip on {path}")
        out[name] = bytes_to_rgb(ob, size, resample)
        valid[name] = True
    return i, out, valid, len(b), byte_entropy(b)


def build_store(benign_dir, malware_dir, work, size=128, resample="nearest",
                max_per_class=None, workers=2, techniques=None):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    if (work / "build_done.flag").exists():
        print("store already built; delete build_done.flag to rebuild")
        return
    techs = techniques or list(TECHNIQUES)
    manifest, stats = build_manifest(benign_dir, malware_dir, max_per_class)
    n = len(manifest)
    print(f"{n} files after dedup: {stats}")
    names = ["clean"] + techs
    mm = {k: np.lib.format.open_memmap(work / f"{k}.npy", mode="w+",
                                       dtype=np.uint8, shape=(n, size, size, 3))
          for k in names}
    valid = {t: np.ones(n, bool) for t in techs}
    nbytes, ent = np.zeros(n, np.int64), np.zeros(n)
    jobs = ((i, r["path"], size, resample, techs) for i, r in enumerate(manifest))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (i, out, v, nb, e) in enumerate(ex.map(_process, jobs, chunksize=8)):
            for name in names:
                mm[name][i] = out[name]
            for t in techs:
                valid[t][i] = v[t]
            nbytes[i], ent[i] = nb, e
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{n}")
    for a in mm.values():
        a.flush()
    for t in techs:
        np.save(work / f"valid_{t}.npy", valid[t])
    with open(work / "meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "label", "sha256", "path", "bytes", "entropy"])
        for i, r in enumerate(manifest):
            w.writerow([i, r["label"], r["sha"], r["path"], nbytes[i], f"{ent[i]:.4f}"])
    np.save(work / "labels.npy", np.array([r["label"] for r in manifest], np.int8))
    (work / "store.json").write_text(json.dumps(
        {"size": size, "resample": resample, "n": n, "techniques": techs,
         "invertible": [t for t in techs if TECHNIQUES[t].invertible],
         "manifest_stats": stats,
         "valid_fraction": {t: float(valid[t].mean()) for t in techs}}, indent=2))
    (work / "build_done.flag").write_text("ok")
    print("store built")


class Store:
    def __init__(self, work):
        work = Path(work)
        self.work = work
        self.info = json.loads((work / "store.json").read_text())
        self.size = self.info["size"]
        self.techs = self.info["techniques"]
        self.arrays = {k: np.load(work / f"{k}.npy", mmap_mode="r")
                       for k in ["clean"] + self.techs}
        self.valid = {"clean": np.ones(self.info["n"], bool)}
        for t in self.techs:
            self.valid[t] = np.load(work / f"valid_{t}.npy")
        self.labels = np.load(work / "labels.npy")
