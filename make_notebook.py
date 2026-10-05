"""Build dae_study_colab.ipynb: one self-contained notebook (code embedded via %%writefile)."""
import json
from pathlib import Path

SRC = Path("/home/claude/dae_study")
MODULES = ["obfuscate.py", "imaging.py", "data.py", "models.py", "metrics.py",
           "run_experiment.py", "report.py"]

cells = []
def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})
def code(text, hide=False):
    meta = {"cellView": "form"} if hide else {}
    cells.append({"cell_type": "code", "metadata": meta, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(True)})

md("""
# Do Denoising Autoencoders Improve Static Malware Detection Under Obfuscation?

**How to use this notebook (4 steps)**
1. *Runtime → Change runtime type → T4 GPU.*
2. Look at the **Settings** cell. The paths are already filled in; change them only if your files moved.
3. *Runtime → Run all.* Leave `REHEARSAL = True` the first time (a short test). When it finishes without errors, set `REHEARSAL = False` and *Run all* again for the real run. The real run starts with ONE seed and one classifier so you get a first answer sooner. Everything is saved to Google Drive, so after a disconnect just *Run all* again and it continues where it stopped.
4. When done, send me the file **`report.md`** (saved in your Google Drive, folder `dae_study_results_full/report/`, and also printed at the bottom). To strengthen the result later, change `SEEDS` to `"0 1 2"` and `CLFS` to `"cnn1 cnn2"` in the Settings cell and *Run all* again: finished parts are skipped.

Nothing here executes any malware: files are only read as bytes and turned into images.
""")

code('''
# ===================== SETTINGS (the only cell you may edit) =====================
REHEARSAL = True      # True = quick test.  False = the real experiment.

TRAPMINE_DIR = "/content/drive/MyDrive/Drive_Mesut/Drive_IEEE/PAPER/1. ULUSLARARASI_MAKALE/4_ML for MALWARE/DATASETS/TRAPMINE"
BENIGN_ZIP, BENIGN_PASSWORD   = TRAPMINE_DIR + "/ml-data-benign.zip",    "benign"
MALWARE_ZIP, MALWARE_PASSWORD = TRAPMINE_DIR + "/ml-data-malicious.zip", "infected"
# =================================================================================

MODE = "rehearsal" if REHEARSAL else "full"
WORK = f"/content/work_{MODE}"                                   # fast local disk
OUT  = f"/content/drive/MyDrive/dae_study_results_{MODE}"         # survives disconnects
MAX_PER_CLASS = 500 if REHEARSAL else 10000
SEEDS         = "0"                       # later: "0 1 2"  (finished seeds are skipped)
CLFS          = "cnn1"                    # later: "cnn1 cnn2"
DAE_ARCHS     = "paper unet"              # paper = original DAE, unet = stronger DAE
EPOCHS        = 5 if REHEARSAL else 20
print(f"mode={MODE}  max_per_class={MAX_PER_CLASS}  seeds={SEEDS}  classifiers={CLFS}  dae={DAE_ARCHS}  epochs={EPOCHS}")
''')

code('''
import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPU found:", gpus if gpus else "NONE - set Runtime > Change runtime type > T4 GPU, then re-run")
from google.colab import drive
drive.mount("/content/drive")
''')

code('''
# Extract TRAPMINE keeping folder structure (the old notebook flattened it and lost files to name collisions).
import os, shutil, subprocess
from pathlib import Path

if shutil.which("7z") is None:
    subprocess.run(["apt-get", "-qq", "install", "-y", "p7zip-full"], check=True)

def extract(zip_path, out_dir, password):
    if (Path(out_dir) / ".done").exists():
        return
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run(["7z", "x", zip_path, f"-o{out_dir}", f"-p{password}", "-y"],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):          # 1 = warnings only
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(f"7z failed on {zip_path}")
    (Path(out_dir) / ".done").write_text("ok")

extract(BENIGN_ZIP,  "/content/trapmine/benign",  BENIGN_PASSWORD)
extract(MALWARE_ZIP, "/content/trapmine/malware", MALWARE_PASSWORD)

count = lambda d: sum(1 for p in Path(d).rglob("*") if p.is_file() and p.name != ".done")
print("benign files :", count("/content/trapmine/benign"))
print("malware files:", count("/content/trapmine/malware"))
''')

md("## Code (embedded so this notebook is self-contained - no need to open these)")

os_header = "import os\nos.makedirs('/content/dae_study', exist_ok=True)\n"
code(os_header)
for name in MODULES:
    code(f"%%writefile /content/dae_study/{name}\n" + (SRC / name).read_text())

md("## Step 1 - Build the image dataset (hash-dedup, then obfuscate both classes)")
code('''
!python /content/dae_study/run_experiment.py build \\
    --benign /content/trapmine/benign --malware /content/trapmine/malware \\
    --work {WORK} --max-per-class {MAX_PER_CLASS} --workers 2
''')

md("## Step 2 - Run the experiment (results are saved to Google Drive as it goes; safe to re-run after a disconnect)")
code('''
import os
for d in ("models", "results", "figures"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)
    link = f"{WORK}/{d}"
    if not os.path.exists(link):
        os.symlink(f"{OUT}/{d}", link)

!python /content/dae_study/run_experiment.py run --work {WORK} \\
    --seeds {SEEDS} --clfs {CLFS} --dae-archs {DAE_ARCHS} --dae-epochs {EPOCHS} --clf-epochs {EPOCHS} --batch 64
''')

md("## Step 3 - Report")
code('''
import shutil
!python /content/dae_study/report.py --work {WORK}

os.makedirs(f"{OUT}/report", exist_ok=True)
for f in os.listdir(f"{WORK}/report"):
    shutil.copy(f"{WORK}/report/{f}", f"{OUT}/report/{f}")
for f in ("meta.csv", "store.json"):           # dataset manifest: hashes, labels, counts (publishable)
    shutil.copy(f"{WORK}/{f}", f"{OUT}/{f}")

print(open(f"{WORK}/report/report.md").read())

from IPython.display import Image, display
for f in sorted(os.listdir(f"{WORK}/figures")):
    if f.endswith(".png"):
        print(f); display(Image(f"{WORK}/figures/{f}"))
print("\\nSaved to Google Drive:", OUT)
''')

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
Path("/mnt/user-data/outputs/dae_study_colab.ipynb").write_text(json.dumps(nb, indent=1))
print("cells:", len(cells))
