# Do Denoising Autoencoders Improve Image-Based Static Malware Detection Under Synthetic Byte-Level Obfuscation and UPX Packing? A Controlled Evaluation

Code and result tables for the paper (MDPI *Electronics*, manuscript electronics-4646019).
The question: if a file is obfuscated, does passing its image through a denoising autoencoder (DAE) before the classifier restore detection, and does it do so better than the simple alternatives (no defence, training the classifier on obfuscated files, a blur filter)?

## Data

- The collection is TRAPMINE: 17,000 files (10,000 benign, 7,000 malware) after removing duplicates by SHA-256.
- The raw executables are confidential and some are live malware. They are not released.
- The clean 128x128 RGB images of the 17,000 files (3 bytes = 1 pixel) are on Zenodo: https://doi.org/10.5281/zenodo.19100300 (`benign.zip`, `malicious.zip`). The same record also holds `Cleaned_Images.zip` from an earlier study; it is not used here.
- The obfuscated images are made from the raw bytes and are **not** released. The code applies every transformation to executable files, so the whole experiment can be repeated on any other folder of benign and malware executables.

## Protocol (short)

- Six synthetic byte-level transformations (`obfuscate.py`) and the real packer UPX are applied to both classes, so every file has a known clean version. Five of the six are exactly invertible (the exact inverse is the oracle).
- Conditions per transformation and classifier: `clean_ref`, `identity` (no defence), `augmentation`, `blur_matched`, `dae_t`, `dae_u`, `dae_u_matched`, `oracle`, `clean_subset`.
- Two CNN classifiers (`cnn1`, `cnn2`), two DAEs (baseline and U-Net), 3 random splits (60/20/20, stratified). DAEs and classifiers are fit on the train split only. All numbers are on the test split.
- Statistics: paired stratified bootstrap over test files (2,000 resamples), equivalence margin of +-0.02 AUC (also +-0.01 and +-0.03).

## Run

```
pip install -r requirements.txt

# 1. image store (about 6 GB for 17,000 files)
python run_experiment.py build --benign DIR_BENIGN --malware DIR_MALWARE --work OUT

# 2. main experiment (resumable)
python run_experiment.py run --work OUT --seeds 0 1 2 --clfs cnn1 cnn2

# 3. analyses added in the revision (results go to OUT/results/)
python extra.py scores  --work OUT --seeds 0 1 2 --clfs cnn1 cnn2
python extra.py stats   --work OUT --seeds 0 1 2 --clfs cnn1 cnn2 --boot 2000
python extra.py matched --work OUT --seeds 0 1 2 --clfs cnn1 cnn2
python extra.py neardup --work OUT --seeds 0 1 2 --clfs cnn1 cnn2

# 4. UPX: how many files UPX accepted, per class
python upx_counts.py --meta meta.csv --valid valid_UPX.npy --upx-version "upx 4.2.2"
```

UPX is used automatically when `upx` is on the PATH (the paper used UPX 4.2.2).
`dae_study_revision.ipynb` is the Colab notebook that runs these steps.

## Result tables (CSV files in the repository root)

| File | Content | Paper |
|---|---|---|
| `paired_bootstrap.csv` | paired AUC differences between defences, 95% interval, equivalence at three margins | Tables 6, 7 |
| `matched_eval.csv` | AUC inside groups of equal size and of equal size and entropy; metadata-only baselines | Table 13, Table 1 |
| `neardup.csv` | near-duplicate check of the test images against the train images (cosine similarity of 16x16x3 thumbnails, thresholds 0.90, 0.95, 0.99) | Table 14 |
| `upx_counts.csv` | files accepted by UPX 4.2.2, by class | Table 10 |

The other tables of the paper are produced by `report.py`.

## Not run

`extra.py` also contains a leave-one-transformation-out test (`loo`), a DAE sensitivity analysis (`sens`) and a fixed-byte-budget image mode (`build --mode fixed`). They were **not** run for the paper, and no result in the paper depends on them. They are listed as limitations.

## Files

`obfuscate.py` transformations and exact inverses | `imaging.py` bytes to RGB | `data.py` dedup, split, image store | `models.py` DAEs, CNNs, training | `metrics.py` | `run_experiment.py` build and run | `extra.py` revision analyses | `upx_counts.py` | `report.py` tables | `metadata_baseline.py`, `export_release.py`, `make_notebook*.py` helpers from the first release.

`NOTES_earlier_evaluation.md` is the previous README, kept for the record.
