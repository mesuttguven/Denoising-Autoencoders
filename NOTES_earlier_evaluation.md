# Do Denoising Autoencoders Improve Static Malware Detection Under Obfuscation? A Controlled Evaluation

Experiment package for the rebuilt study. It replaces the original notebook's evaluation.

## Question

If an attacker obfuscates a file, does passing its image through a denoising autoencoder (DAE) before
classification restore detection, and does it do so better than the obvious alternatives
(no defence; training the classifier on obfuscated data)?

## What was wrong with the original evaluation

| # | Finding in the original notebook | Effect |
|---|---|---|
| 1 | Benign test images were copied raw; only malware went through the DAE (cells 42, 44) | Specificity frozen across models; the DAE output acted as a class marker |
| 2 | "DAE + X" re-used the classifier trained on raw images (cells 37, 48-50 load the same model folder) | Pure train/test shift, which explains the sensitivity jumps and collapses; Layers/Params columns in Table 1 are arithmetic, not measurements |
| 3 | DAEs trained at 128x128, applied at 224x224 | Untrained scale |
| 4 | `save_img(scale=True)` min-max stretched DAE outputs only | Per-image contrast normalisation that raw images never received |
| 5 | Committed notebook does not reproduce Table 1 (output folder mismatch) | Results not reproducible from the repo |
| 6 | Notebook loads TRAPMINE / ML4CS data; the manuscript names VirusTotal, MalwareBazaar, Any.Run, Softonic, SourceForge, CNET | Provenance must be reconciled before submission |
| 7 | The coded obfuscations differ from the manuscript's descriptions (see `obfuscate.py`) | In particular **File_Header_Modification keeps 128 bytes and overwrites the rest with 0xFF**, which destroys the file and cannot be inverted |

## Protocol

* Obfuscation is applied to **both** classes, so every file has a known clean counterpart.
* Whatever preprocessing a condition uses is applied to **every** test file.
* Files are deduplicated by SHA-256 (hashes present in both classes are dropped) **before** splitting.
  Splits are 60/20/20, stratified, drawn per seed. DAEs and classifiers are fit on the train split only.
* Pipeline holds everything in uint8 memmaps; there is no PNG round-trip.

### Conditions (per technique, per classifier)

| condition | classifier trained on | test input |
|---|---|---|
| `clean_ref` | clean | clean |
| `identity` | clean | obfuscated (no defence) |
| `dae_t` | clean | DAE trained for that technique (technique known) |
| `dae_u` | clean | one union DAE (technique unknown; realistic) |
| `dae_u_matched` | union-DAE outputs of clean+obfuscated mix | union DAE output |
| `augmentation` | clean+obfuscated mix, **no DAE** | obfuscated |
| `oracle` | clean | exact analytic inverse (invertible techniques only; equals `clean_ref`) |

The union DAE is trained with clean images in its input mix, so it must pass clean files through unchanged.
The cost of that is measured (`tech = clean` rows).

### Metrics
AUROC, AUPRC, TPR at FPR 1e-2 and 1e-3, plus accuracy/sensitivity/specificity at 0.5 for continuity with the paper.
Mean ± std over seeds, and **paired** per-seed deltas against `identity` and `augmentation` with win counts.
`TPR@FPR=1e-3` is only meaningful when the benign test set is well above 1,000 files (see `n_test`).
Reconstruction: MSE, PSNR and SSIM of DAE output against the clean image, split by class, next to the
do-nothing baseline. A DAE that does not beat the baseline has learned nothing.

## Run

```bash
pip install tensorflow numpy pandas pillow scikit-learn matplotlib

# 1. build the image store (one-off; ~7 stores x N files x 49 KB at 128x128, about 6 GB for 17k files)
python run_experiment.py build --benign PATH/benign_exe --malware PATH/malicious_exe --work OUT --workers 4

# dress rehearsal first
python run_experiment.py build --benign ... --malware ... --work OUT_small --max-per-class 500
python run_experiment.py run   --work OUT_small --seeds 0 --clfs cnn1 --dae-epochs 5 --clf-epochs 5

# 2. full run (resumable: re-running skips finished seed/classifier blocks)
python run_experiment.py run --work OUT --seeds 0 1 2 3 4 --clfs cnn1 cnn2

# 3. ablation: DAE trained on benign+malware clean/obfuscated pairs
python run_experiment.py run --work OUT --seeds 0 1 2 3 4 --dae-train both

# 4. tables
python report.py --work OUT        # -> OUT/report/report.md and CSVs, OUT/figures/recon_*.png
```

Extra ablation worth one run: `build --resample area` (the original nearest-neighbour downsizing keeps roughly
5% of the bytes of a 1 MB file; area averaging keeps all of them but blurs byte structure).

## Not yet covered (do these before submitting)

* **Near-duplicates.** Only exact hashes are removed. Cluster with TLSH or imphash and split by cluster.
* **Real packers.** `UPX` is wired in and activates automatically if `upx` is on PATH, but it has not been
  exercised against real binaries here. Add MPRESS and a source-level obfuscator for ecological validity.
* **Packed benign files.** `meta.csv` records per-file entropy and size; use it (or Detect-It-Easy) to report how
  many "clean" benign files are already packed.
* **Pretrained classifiers** (VGG16, ResNet50, InceptionV3, MobileNet) are not wired in; `models.build_cnn` is
  where to add them.
* **Non-image baseline** (LightGBM on EMBER features) and a **byte-level learned inverse** are still absent.
* **Data provenance** (item 6 above) and the **antivirus-engine threshold** wording must be fixed in the manuscript.
* No hyperparameter search; defaults are Adam, 20 epochs, early stopping on validation loss.

## Files

`obfuscate.py` exact obfuscations and inverses | `imaging.py` binary to RGB | `data.py` dedup, split, store |
`models.py` DAE/CNN, batching, training | `metrics.py` | `run_experiment.py` | `report.py` |
`tests/make_synthetic.py` toy binaries used for the smoke test.
