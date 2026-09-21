"""
train.py
--------

Runs the full two-phase transfer-learning training strategy:
  Phase 1 - train only the new classification head, backbone frozen.
  Phase 2 - unfreeze the top layers of the backbone and fine-tune with
            a much lower learning rate.

"""

import json
import os

import tensorflow as tf

from config import (
    BATCH_SIZE,
    EPOCHS_HEAD,
    EPOCHS_FINE_TUNE,
    LEARNING_RATE_HEAD,
    LEARNING_RATE_FINE_TUNE,
    MODEL_DIR,
    LOG_DIR,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR,
    BACKBONE,
)
from data_preprocessing import build_dataframe, stratified_split, dataset_summary
from data_augmentation import build_tf_dataset, get_class_weights
from model import build_model, unfreeze_for_fine_tuning, count_trainable_params


def make_callbacks(phase_name: str):
    ckpt_path = os.path.join(MODEL_DIR, f"best_model_{phase_name}.keras")
    return [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-8,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(LOG_DIR, f"history_{phase_name}.csv")
        ),
    ]


def main():
    print("Building dataset index ...")
    df = build_dataframe()
    print(dataset_summary(df))

    train_df, val_df, test_df = stratified_split(df)
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    class_weights = get_class_weights(train_df)
    print("class weights:", class_weights)

    train_ds = build_tf_dataset(train_df, training=True, batch_size=BATCH_SIZE)
    val_ds = build_tf_dataset(val_df, training=False, batch_size=BATCH_SIZE)
    test_ds = build_tf_dataset(test_df, training=False, batch_size=BATCH_SIZE)

    # ---- Phase 1: train the head only ------------------------------------
    print(f"\n=== Phase 1: training head on frozen {BACKBONE} ===")
    model, backbone = build_model()
    print("Trainable params (phase 1):", count_trainable_params(model))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weights,
        callbacks=make_callbacks("phase1_head"),
    )

    # ---- Phase 2: fine-tune the top of the backbone -----------------------
    print(f"\n=== Phase 2: fine-tuning top layers of {BACKBONE} ===")
    unfreeze_for_fine_tuning(model, backbone)
    print("Trainable params (phase 2):", count_trainable_params(model))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE_TUNE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        class_weight=class_weights,
        callbacks=make_callbacks("phase2_finetune"),
    )

    # ---- Save final model + combined history -------------------------------
    final_path = os.path.join(MODEL_DIR, "dr_final_model.keras")
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    combined_history = {
        "phase1": history_head.history,
        "phase2": history_ft.history,
    }
    with open(os.path.join(LOG_DIR, "combined_history.json"), "w") as f:
        json.dump(combined_history, f, indent=2)

    # Persist the test split itself, so evaluate.py can be run later without
    # re-doing the (seeded, therefore reproducible) split from scratch.
    test_df.to_csv(os.path.join(LOG_DIR, "test_split.csv"), index=False)

    print("\nTraining complete. Run evaluate.py next to generate report figures.")
    return model, test_ds, test_df


if __name__ == "__main__":
    main()
