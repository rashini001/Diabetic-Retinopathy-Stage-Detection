"""
data_augmentation.py
---------------------

Two complementary strategies are used together, which is standard
practice for medical imaging tasks with class imbalance like this one
(Healthy=1000, Mild DR=370, Moderate DR=900, Proliferate DR=290,
Severe DR=190):

  1. On-the-fly geometric/photometric augmentation (rotation, flips,
     zoom, brightness/contrast jitter) applied to every image, which
     increases effective dataset diversity and reduces overfitting.
  2. Class weighting during training, which is a cleaner way to handle
     imbalance than naive oversampling for a 5-way ordinal-ish medical
     classification task: it does not duplicate raw pixels (which can
     encourage memorisation) and does not throw away majority-class
     data (which under-sampling would).

Both the tf.data pipeline (fast, GPU-friendly, used for the real
training run) and a small pure-numpy/OpenCV augmentation function
(no TensorFlow dependency, used for report figures and unit testing)
are provided.
"""

import numpy as np
import cv2
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from config import IMG_SIZE, BATCH_SIZE, NUM_CLASSES, CLASS_NAMES, RANDOM_SEED
from data_preprocessing import preprocess_image


# ---------------------------------------------------------------------------
# Class weights (balancing strategy)
# ---------------------------------------------------------------------------
def get_class_weights(train_df: pd.DataFrame) -> dict:
    """
    Compute inverse-frequency class weights so that the loss function
    penalises mistakes on minority classes (Mild/Proliferate/Severe DR)
    more heavily than mistakes on the majority classes (Healthy,
    Moderate DR). Returned as {label_idx: weight}, the format
    tf.keras.Model.fit(..., class_weight=...) expects.
    """
    labels = train_df["label_idx"].values
    classes = np.arange(NUM_CLASSES)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ---------------------------------------------------------------------------
# Pure numpy/OpenCV augmentation (no TF dependency) - used for figures/tests
# ---------------------------------------------------------------------------
def augment_image_cv(img: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Apply a random combination of rotation, horizontal flip, zoom and
    brightness jitter to a single float32 [0,1] RGB image using only
    numpy/OpenCV. Mirrors (a subset of) what the tf.data pipeline below
    does on the GPU during real training, so it can be used to produce
    "sample augmented images" figures for the report without needing
    TensorFlow installed.
    """
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    # Random rotation up to +/-20 degrees
    angle = rng.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random horizontal flip
    if rng.random() < 0.5:
        img = cv2.flip(img, 1)

    # Random zoom (crop then resize back)
    zoom = rng.uniform(0.9, 1.1)
    if zoom != 1.0:
        nh, nw = int(h / zoom), int(w / zoom)
        nh, nw = min(nh, h), min(nw, w)
        y0 = rng.integers(0, max(1, h - nh + 1))
        x0 = rng.integers(0, max(1, w - nw + 1))
        img = img[y0 : y0 + nh, x0 : x0 + nw]
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    # Random brightness jitter
    brightness = rng.uniform(0.85, 1.15)
    img = np.clip(img * brightness, 0.0, 1.0)

    return img.astype(np.float32)


# ---------------------------------------------------------------------------
# tf.data pipeline (used for actual model training)
# ---------------------------------------------------------------------------
def build_tf_dataset(df: pd.DataFrame, training: bool, batch_size: int = BATCH_SIZE):
    """
    Build a tf.data.Dataset that reads image paths from `df`, applies the
    CPU-side preprocessing from data_preprocessing.py, one-hot encodes the
    label, and (when training=True) applies tf.keras augmentation layers.

    Imported lazily inside the function so that this module can still be
    imported (and its pure-numpy functions unit-tested) in environments
    without TensorFlow installed.
    """
    import tensorflow as tf

    filepaths = df["filepath"].values
    labels = df["label_idx"].values

    augmenter = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(20 / 360, fill_mode="reflect"),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomZoom(0.1, fill_mode="reflect"),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
        ],
        name="augmentation",
    )

    def _load(filepath, label):
        def _py_preprocess(fp):
            fp = fp.numpy().decode("utf-8")
            return preprocess_image(fp)

        img = tf.py_function(_py_preprocess, [filepath], tf.float32)
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        label = tf.one_hot(label, NUM_CLASSES)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    # Quick manual check that does NOT require TensorFlow.
    from data_preprocessing import build_dataframe, preprocess_image

    df = build_dataframe()
    cw = get_class_weights(df)
    print("class weights:", {CLASS_NAMES[k]: round(v, 2) for k, v in cw.items()})

    sample = preprocess_image(df.iloc[0]["filepath"])
    aug = augment_image_cv(sample, seed=0)
    print("augmented sample shape:", aug.shape)
