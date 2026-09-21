"""
utils.py
--------
Small shared helpers used across the codebase . Kept separate
from config.py so it can hold behaviour (functions), not just constants.
"""

import os
import random

import numpy as np

from config import RANDOM_SEED


def set_global_seed(seed: int = RANDOM_SEED):
    """
    Seed python, numpy and TensorFlow, so that the
    train/val/test split, weight initialisation and augmentation are
    reproducible run-to-run.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def check_gpu():
    """
    Print whether a GPU is visible to TensorFlow. Training this pipeline
    on CPU is possible but slow (expect hours rather than minutes).
    """
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"{len(gpus)} GPU(s) available: {[g.name for g in gpus]}")
        else:
            print("No GPU detected -- training will be slow. In Colab: "
                  "Runtime > Change runtime type > select a GPU.")
    except ImportError:
        print("TensorFlow is not installed in this environment.")


if __name__ == "__main__":
    set_global_seed()
    check_gpu()
