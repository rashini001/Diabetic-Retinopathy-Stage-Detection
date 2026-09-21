# Diabetic Retinopathy Stage Detection

A CNN transfer-learning pipeline that classifies retinal fundus photographs into five diabetic retinopathy stages — **Healthy, Mild, Moderate, Severe, and Proliferate DR** — built with TensorFlow/Keras (EfficientNetB0) and OpenCV.

Diabetic retinopathy is a leading cause of preventable blindness in adults with diabetes. Automated staging from fundus photographs can help triage patients for specialist review in settings where ophthalmologist access is limited.

> Built as a Computer Vision coursework project. Dataset sourced from Kaggle.

---

## Results

Trained on 2,750 fundus images across 5 classes, evaluated on a held-out stratified test set (413 images, 15%):

| Metric | Score |
|---|---|
| Overall accuracy | **69.5%** |
| Macro-average F1 | **0.559** |
| Weighted-average F1 | 0.690 |

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Healthy | 0.943 | 0.987 | 0.964 | 150 |
| Mild DR | 0.615 | 0.286 | 0.390 | 56 |
| Moderate DR | 0.664 | 0.674 | 0.669 | 135 |
| Proliferate DR | 0.409 | 0.409 | 0.409 | 44 |
| Severe DR | 0.286 | 0.500 | 0.364 | 28 |

The model separates Healthy from disease almost perfectly (99% recall). Among disease stages, most confusion falls on **adjacent severity levels** (e.g. Mild ↔ Moderate) rather than across the full spectrum — consistent with diabetic retinopathy being a progressive, ordinal condition rather than distinct unrelated categories. See [Limitations](#limitations) below.

---

## Pipeline overview

```
Raw fundus image
   │
   ▼
Preprocessing        crop to retina → CLAHE contrast enhancement → edge enhancement → resize 224×224 → normalise
   │
   ▼
Augmentation          rotation, flip, zoom, brightness/contrast jitter (train split only)
   │                  + inverse-frequency class weighting (handles imbalance)
   ▼
Transfer learning     EfficientNetB0 (ImageNet-pretrained) + custom classification head
   │                  Phase 1: frozen backbone, train head only
   │                  Phase 2: unfreeze top layers, fine-tune at low LR
   ▼
Evaluation             accuracy/loss curves, confusion matrix, per-class precision/recall/F1
```

<p align="center">
  <img src="outputs/preprocessing_demo.png" width="90%" alt="Preprocessing pipeline example">
</p>

---

## Dataset

Sourced from Kaggle: 2,750 RGB fundus photographs (256×256), organised into 5 class folders. Imbalanced — Healthy (1000 images) outnumbers Severe DR (190 images) by 5.3x, addressed via augmentation + class weighting rather than naive oversampling.

| Class | Images |
|---|---:|
| Healthy | 1000 |
| Moderate DR | 900 |
| Mild DR | 370 |
| Proliferate DR | 290 |
| Severe DR | 190 |

Dataset is not included in this repository — download it from Kaggle and point `config.py` at the extracted folder (see [Setup](#setup)).

---

## Project structure

```
dr_project/
├── src/
│   ├── config.py               All paths & hyperparameters
│   ├── data_preprocessing.py   Dataset indexing, stratified split, CLAHE/crop/edge-enhancement
│   ├── data_augmentation.py    tf.data augmentation pipeline + class weighting
│   ├── model.py                EfficientNetB0 backbone + transfer-learning head
│   ├── train.py                Two-phase training loop with callbacks
│   ├── evaluate.py             Curves, confusion matrix, classification report
│   └── utils.py                Reproducibility (seeding) and GPU checks
├── notebooks/
│   └── DR_Stage_Detection.ipynb  End-to-end runnable notebook
├── outputs/                    Sample results (figures shown above)   
├── requirements.txt
```
---

## Setup

**Recommended: Google Colab (free GPU)**

```python
# 1. Mount Drive / upload the dataset zip and this repo
from google.colab import drive
drive.mount('/content/drive')

# 2. Extract the dataset so /content/data directly contains the 5 class folders
import zipfile
with zipfile.ZipFile('archive.zip') as z:
    z.extractall('/content/data')

# 3. Point the code at your paths
import os, sys
os.environ["DR_DATA_DIR"]   = "/content/data"
os.environ["DR_OUTPUT_DIR"] = "/content/outputs"
os.environ["DR_MODEL_DIR"]  = "/content/saved_models"
sys.path.append("/content/dr_project/src")

!pip install -q opencv-python-headless
```

Then open and run `notebooks/DR_Stage_Detection.ipynb` top to bottom.

**Local setup**

```bash
git clone https://github.com/<your-username>/dr-stage-detection.git
cd dr-stage-detection
pip install -r requirements.txt

export DR_DATA_DIR=/path/to/dataset
export DR_OUTPUT_DIR=./outputs
export DR_MODEL_DIR=./saved_models

python src/train.py
python src/evaluate.py
```

---

## Tech stack

TensorFlow / Keras · OpenCV · scikit-learn · pandas · matplotlib

---

## Limitations

- **Class imbalance** is only partially resolved — the rarest classes (Mild, Proliferate, Severe DR) still have the weakest recall.
- **Dataset provenance is undocumented** (imaging hardware, patient demographics unknown), so generalisation to other clinics/populations is unverified.
- This is an **academic prototype, not a validated clinical tool** 

## Possible improvements

- Ordinal regression instead of plain multi-class classification (DR stages are ordered, not categorical)
- Grad-CAM visualisation for model interpretability
- Backbone comparison (ResNet50V2 / MobileNetV2 are already supported via `config.BACKBONE`)
- External validation on a second, independently-sourced dataset

---

## License

This project is provided for academic/educational purposes. The dataset is subject to its original Kaggle license and is not redistributed here.

