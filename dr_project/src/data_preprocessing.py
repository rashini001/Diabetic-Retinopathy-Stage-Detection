"""
data_preprocessing.py
----------------------

Responsibilities:
  1. Walk the dataset folder and build a labelled file index.
  2. Produce a stratified train / validation / test split.
  3. Basic image preprocessing: contrast enhancement (CLAHE), edge
     enhancement, resizing and normalisation.
"""

import os
import random
from collections import Counter

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    CLASS_NAMES,
    DATA_DIR,
    IMG_SIZE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    TEST_SPLIT,
    VAL_SPLIT,
    RANDOM_SEED,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# 1. Build a labelled file index
# ---------------------------------------------------------------------------
def build_dataframe(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    Scan `data_dir` for the five class sub-folders and return a DataFrame
    with columns [filepath, label, label_idx].

    Raises a clear error early if a class folder is missing, rather than
    silently training on 4 classes -- a common source of confusing bugs.
    """
    rows = []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            raise FileNotFoundError(
                f"Expected class folder not found: {cls_dir}\n"
                f"Check DR_DATA_DIR / config.DATA_DIR and the exact "
                f"folder names in config.CLASS_NAMES."
            )
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                rows.append(
                    {
                        "filepath": os.path.join(cls_dir, fname),
                        "label": cls,
                        "label_idx": idx,
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No images found under {data_dir}")
    return df


# ---------------------------------------------------------------------------
# 2. Stratified split
# ---------------------------------------------------------------------------
def stratified_split(df: pd.DataFrame):
    """
    Split into train / val / test using stratification on `label`, so each
    split preserves the original (imbalanced) class ratios. Balancing is
    then handled separately at training time (see data_augmentation.py),
    which keeps the validation/test sets honest measures of real-world
    performance.
    """
    train_val, test = train_test_split(
        df,
        test_size=TEST_SPLIT,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )
    # VAL_SPLIT is expressed relative to the *original* dataset, so we
    # rescale it relative to the remaining train_val portion.
    val_relative = VAL_SPLIT / (1 - TEST_SPLIT)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val["label"],
        random_state=RANDOM_SEED,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# 3. Image preprocessing
# ---------------------------------------------------------------------------
def apply_clahe(img_bgr: np.ndarray) -> np.ndarray:
    """
    Contrast-Limited Adaptive Histogram Equalisation on the L channel of
    LAB colour space. Retinal fundus images are typically low-contrast
    and unevenly illuminated (bright centre, dark edges); CLAHE makes
    lesions (haemorrhages, exudates, microaneurysms) more visible without
    blowing out already-bright regions the way global histogram
    equalisation would.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_edge_enhancement(img_bgr: np.ndarray, amount: float = 0.6) -> np.ndarray:
    """
    Unsharp-mask style edge enhancement: subtract a Gaussian-blurred copy
    from the original to emphasise fine vascular structure and lesion
    boundaries, which helps the CNN pick up early-stage (Mild) features.
    """
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(img_bgr, 1 + amount, blurred, -amount, 0)
    return sharpened


def crop_to_circle_bounds(img_bgr: np.ndarray) -> np.ndarray:
    """
    Fundus photographs usually have a large black background around the
    circular retina. Cropping to the retina's bounding box removes dead
    pixels the network would otherwise have to learn to ignore, and
    effectively increases the retina's resolution after the subsequent
    resize step.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return img_bgr
    x, y, w, h = cv2.boundingRect(coords)
    if w < 10 or h < 10:
        return img_bgr
    return img_bgr[y : y + h, x : x + w]


def preprocess_image(
    filepath: str,
    img_size=IMG_SIZE,
    use_clahe: bool = True,
    use_edge_enhancement: bool = True,
    use_crop: bool = True,
) -> np.ndarray:
    """
    Full single-image preprocessing pipeline used both at training time
    (via the tf.data pipeline in data_augmentation.py) 

    Returns a float32 RGB array in [0, 1], shape (img_size[0], img_size[1], 3).
    """
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")

    if use_crop:
        img = crop_to_circle_bounds(img)
    if use_clahe:
        img = apply_clahe(img)
    if use_edge_enhancement:
        img = apply_edge_enhancement(img)

    img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return img


# ---------------------------------------------------------------------------
# 4. EDA helpers
# ---------------------------------------------------------------------------
def class_distribution(df: pd.DataFrame) -> pd.Series:
    """Class counts"""
    return df["label"].value_counts()


def dataset_summary(df: pd.DataFrame) -> dict:
    """
    total images, per-class
    counts, and the imbalance ratio (largest class / smallest class).
    """
    counts = Counter(df["label"])
    largest = max(counts.values())
    smallest = min(counts.values())
    return {
        "total_images": len(df),
        "num_classes": len(counts),
        "class_counts": dict(counts),
        "imbalance_ratio": round(largest / smallest, 2),
    }


if __name__ == "__main__":
    # Quick manual sanity check: `python data_preprocessing.py`
    df = build_dataframe()
    print(dataset_summary(df))
    train, val, test = stratified_split(df)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
