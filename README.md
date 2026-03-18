# Predicting Original Malicious Code from Obfuscated Code Using Denoising AutoEncoders

[![Paper](https://img.shields.io/badge/PeerJ%20Computer%20Science-Under%20Review-blue)](https://peerj.com/computer-science/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

## Description

This repository contains the code and resources accompanying the research paper:

> **Predicting Original Malicious Code from Obfuscated Code Using Denoising AutoEncoders**  
> Mesut Guven, TOBB University of Economics and Technology, Ankara, Turkey  
> *Submitted to PeerJ Computer Science, 2026*

The project investigates the use of **Denoising Autoencoders (DAEs)** as a preprocessing step to reverse obfuscation in malware samples, improving the accuracy and sensitivity of malware classification systems. Software binaries are converted into RGB images, and DAE models are trained to reconstruct the pre-obfuscated visual representations of malicious code before passing them to CNN-based classifiers.

---

## Repository Contents

| File | Description |
|------|-------------|
| `Using_DAE_for_Detecting_Obfuscated_Malware.ipynb` | Main notebook: full pipeline for obfuscation, DAE training, and malware classification |
| `8_Auto_Encoder.ipynb` | Introductory notebook explaining the autoencoder architecture used |
| `DAE.png` | Conceptual diagram of the denoising autoencoder architecture |

---

## Dataset Information

The dataset used in this study consists of:

- **10,000 benign software samples** collected from Softonic, SourceForge, and CNET
- **7,000 malware samples** obtained via public APIs from VirusTotal, Any.Run, and MalwareBazaar

All samples were validated using VirusTotal's multi-engine scanning system. Only files consistently classified as malicious or benign by over 60 antivirus engines were retained.

Each benign binary was synthetically obfuscated using six techniques:
1. Code Packing (zlib compression)
2. Encryption (byte-wise XOR)
3. File Header Modification (first 128 bytes)
4. Control Flow Obfuscation (XOR on every 4th byte)
5. Data Obfuscation (byte value incrementing)
6. String Obfuscation (XOR from fixed offset)

All binaries (original and obfuscated) were converted to 128×128 RGB images by mapping every three consecutive bytes to one RGB pixel.

**Dataset access:** A representative subset is available upon request. Please contact the corresponding author at mesuttguven@gmail.com. A sample subset is also hosted at:  
https://drive.google.com/drive/u/0/folders/1y-GLUtsNYoeFok_mHeyFC9r6B_uwoizn

> **Note:** A permanently archived version with a DOI will be linked here upon paper acceptance.

---

## Code Information

- `Using_DAE_for_Detecting_Obfuscated_Malware.ipynb` — The core research notebook. It covers:
  - Binary-to-RGB image conversion
  - Synthetic obfuscation of benign executables
  - Training six separate DAE models (one per obfuscation technique)
  - Classification using three custom CNNs and four pre-trained architectures (VGG16, ResNet50, InceptionV3, MobileNet)
  - Performance comparison (accuracy, sensitivity, specificity) with and without DAE preprocessing

- `8_Auto_Encoder.ipynb` — A standalone educational notebook that explains the autoencoder architecture, encoder-decoder structure, and MSE-based training objective used in the study.

---

## Usage Instructions

### Running on Google Colab (Recommended)

1. Open the notebook directly in Colab by clicking the badge or navigating to the `.ipynb` file and selecting **Open in Colab**.
2. Mount your Google Drive if you wish to use your own dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Update the dataset paths in the notebook to point to your data directory.
4. Run all cells sequentially from top to bottom.

### Running Locally

```bash
# Clone the repository
git clone https://github.com/mesuttguven/Denoising-Autoencoders.git
cd Denoising-Autoencoders

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook Using_DAE_for_Detecting_Obfuscated_Malware.ipynb
```

---

## Requirements

The code is written in Python 3.8+ and tested on Google Colab with the following libraries:

```
tensorflow>=2.8.0
keras>=2.8.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
Pillow>=8.3.0
```

Install all dependencies with:
```bash
pip install tensorflow numpy opencv-python matplotlib scikit-learn Pillow
```

> The notebooks are designed to run on Google Colab, which provides GPU acceleration. Local execution with a GPU is strongly recommended for DAE training.

---

## Methodology Summary

1. **Data Collection** — Collect and validate 10,000 benign and 7,000 malware samples.
2. **Synthetic Obfuscation** — Apply six obfuscation techniques to benign binaries.
3. **RGB Conversion** — Convert all binaries (original and obfuscated) to 128×128 RGB images.
4. **DAE Training** — Train one DAE per obfuscation type (obfuscated image as input, clean image as target).
5. **Classification** — Classify malware RGB images directly (baseline) and after DAE preprocessing (proposed method).
6. **Evaluation** — Compare accuracy, sensitivity, and specificity across all model/condition combinations.

For full methodology details, please refer to the paper.

---

## Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{guven2026dae_malware,
  title     = {Predicting Original Malicious Code from Obfuscated Code Using Denoising AutoEncoders},
  author    = {Guven, Mesut},
  journal   = {PeerJ Computer Science},
  year      = {2026},
  note      = {Under Review}
}
```

Also consider citing the related prior work this study builds upon:

```bibtex
@article{guven2024malware,
  title   = {Leveraging deep learning and image conversion of executable files for effective malware detection: A static malware analysis approach},
  author  = {Guven, Mesut},
  journal = {AIMS Mathematics},
  volume  = {9},
  pages   = {15223--15245},
  year    = {2024},
  doi     = {10.3934/math.2024873}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contact

**Mesut Guven**  
TOBB University of Economics and Technology, Ankara, Turkey  
📧 mesuttguven@gmail.com | mesutguven@etu.edu.tr
