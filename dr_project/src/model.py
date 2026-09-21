"""
model.py
--------

Provides a small registry of ImageNet-pretrained backbones so the
architecture choice is a one-line change in config.py (BACKBONE), and a
two-phase build/compile helper: a frozen-backbone "head only" phase and
a "fine-tune the top N layers" phase.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

from config import IMG_SIZE, NUM_CLASSES, BACKBONE, FINE_TUNE_AT_LAYER


BACKBONE_REGISTRY = {
    "EfficientNetB0": tf.keras.applications.EfficientNetB0,
    "ResNet50V2": tf.keras.applications.ResNet50V2,
    "MobileNetV2": tf.keras.applications.MobileNetV2,
    "VGG16": tf.keras.applications.VGG16,
}

PREPROCESS_REGISTRY = {
    "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
    "ResNet50V2": tf.keras.applications.resnet_v2.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "VGG16": tf.keras.applications.vgg16.preprocess_input,
}


def build_model(backbone_name: str = BACKBONE, num_classes: int = NUM_CLASSES):
    """
    Build a transfer-learning classifier:
        pretrained backbone (frozen) -> GAP -> Dropout -> Dense(softmax)

    Returns (model, backbone) so the caller can later flip
    `backbone.trainable = True` for the fine-tuning phase in train.py.

    Note: our own preprocessing (data_preprocessing.preprocess_image)
    already scales images to [0, 1]. Each backbone's own
    `preprocess_input` (see PREPROCESS_REGISTRY) expects a different
    input convention (e.g. EfficientNet expects [0,255], others expect
    [-1,1]), so we apply it as the first layer of the model itself. This
    keeps the tf.data pipeline backbone-agnostic -- only model.py needs
    to know which convention this particular backbone wants.
    """
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Choose one of {list(BACKBONE_REGISTRY)}"
        )

    backbone_fn = BACKBONE_REGISTRY[backbone_name]
    preprocess_fn = PREPROCESS_REGISTRY[backbone_name]

    backbone = backbone_fn(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling=None,
    )
    backbone.trainable = False  # phase 1: frozen backbone

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")
    x = layers.Rescaling(scale=255.0)(inputs)  # undo our [0,1] scaling ...
    x = preprocess_fn(x)                        # ... then apply backbone-specific prep
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    x = layers.Dense(128, activation="relu", name="head_dense")(x)
    x = layers.Dropout(0.3, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"DR_{backbone_name}")
    return model, backbone


def unfreeze_for_fine_tuning(model: tf.keras.Model, backbone: tf.keras.Model,
                              fine_tune_at: int = FINE_TUNE_AT_LAYER):
    """
    Phase 2 of training: unfreeze only the top `abs(fine_tune_at)` layers
    of the backbone (a negative index counts from the end) and keep
    everything below frozen. Fine-tuning the whole network from the start
    on ~2.7k images would overfit fast and destroy useful low-level
    ImageNet filters (edges, textures) that are still relevant to fundus
    images; unfreezing only the top layers lets the network adapt its
    higher-level, more task-specific features to diabetic retinopathy.

    BatchNorm layers are explicitly kept frozen even within the unfrozen
    block, which is standard practice -- re-estimating BN statistics on a
    small medical dataset is a common source of training instability.
    """
    backbone.trainable = True
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone.layers[fine_tune_at:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model


def count_trainable_params(model: tf.keras.Model) -> int:
    return int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))


if __name__ == "__main__":
    model, backbone = build_model()
    model.summary()
    print("Trainable params (head-only phase):", count_trainable_params(model))
    unfreeze_for_fine_tuning(model, backbone)
    print("Trainable params (fine-tune phase):", count_trainable_params(model))
