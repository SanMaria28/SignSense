"""
Indian Traffic Sign – Enhanced CNN Training Pipeline (v3 – disk-based)
=======================================================================
Usage:
    python ml_pipeline/train.py --zip "archive (5).zip" --output models

Key improvements over v1/v2:
  • Lazy disk-based tf.data pipeline — images are loaded one batch at a time,
    so RAM usage is ~200-400 MB regardless of dataset size (v2 needed 7+ GB).
  • Background-invariant training: every image is randomly composited onto one
    of several background colours at load time, making the model robust to both
    dark real-world signs AND clean white-bg clipart.
  • MobileNetV2 transfer learning (two-phase: frozen head → fine-tune top-40).
  • No horizontal flip in augmentation (flipping changes arrow direction meaning).
  • Saves training_summary.json with preprocessing="mobilenet_v2" so app.py
    knows which preprocessing to apply at inference time.
"""

import os
import sys
import json
import argparse
import zipfile
import shutil
import csv
import tempfile
import logging
import random
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 96
BATCH_SIZE    = 32
PHASE1_LR     = 1e-3
PHASE2_LR     = 2e-5
PHASE1_EPOCHS = 15
PHASE2_EPOCHS = 20
VAL_SPLIT     = 0.15
SEED          = 42

# Background colours used for random compositing (RGB tuples 0-255)
BG_COLOURS = [
    [255, 255, 255],   # white
    [230, 230, 230],   # light gray
    [60,  60,  60 ],   # dark gray
    [180, 200, 220],   # light blue
    [245, 235, 210],   # cream
]

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_zip(zip_path: str, dest_dir: str) -> str:
    log.info(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)
    entries = list(Path(dest_dir).iterdir())
    return str(entries[0]) if len(entries) == 1 and entries[0].is_dir() else dest_dir


def read_class_map(dataset_root: str) -> dict[int, str]:
    csv_path = Path(dataset_root) / "traffic_sign.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"traffic_sign.csv not found at {csv_path}")
    class_map: dict[int, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            class_map[int(row["ClassId"].strip())] = row["Name"].strip()
    log.info(f"Loaded {len(class_map)} classes")
    return class_map


def collect_paths(dataset_root: str, class_map: dict[int, str]):
    """Return parallel lists (path_strings, label_ints, sorted_ids)."""
    images_root = Path(dataset_root) / "Images"
    sorted_ids  = sorted(class_map.keys())
    id_to_idx   = {cid: idx for idx, cid in enumerate(sorted_ids)}
    paths, labels = [], []
    for cid in sorted_ids:
        class_dir = images_root / str(cid)
        if not class_dir.exists():
            log.warning(f"Folder missing for class {cid}, skipping.")
            continue
        files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        log.info(f"  Class {cid:>2} ({class_map[cid]:40s})  -> {len(files)} images")
        for fp in files:
            paths.append(str(fp))
            labels.append(id_to_idx[cid])
    log.info(f"Total image paths collected: {len(paths)}")
    return paths, labels, sorted_ids


def make_tf_dataset(paths: list, labels: list, augment: bool, shuffle: bool):
    """
    Build a tf.data.Dataset that:
      1. Reads each image file from disk.
      2. Randomly composites it onto one of BG_COLOURS (background invariance).
      3. Applies optional augmentation.
      4. Applies MobileNetV2 preprocessing (-> [-1, 1]).
    """
    import tensorflow as tf

    bg_tensor = tf.constant(BG_COLOURS, dtype=tf.float32)  # shape (5, 3)

    def load_and_composite(path, label):
        # Read & decode
        raw = tf.io.read_file(path)
        img = tf.image.decode_png(raw, channels=4)          # always RGBA
        img = tf.cast(img, tf.float32)

        # Split alpha
        rgb   = img[:, :, :3]                               # [H,W,3] 0-255
        alpha = img[:, :, 3:4] / 255.0                     # [H,W,1] 0-1

        # Pick a random background colour
        bg_idx = tf.random.uniform((), 0, len(BG_COLOURS), dtype=tf.int32)
        bg_rgb = bg_tensor[bg_idx]                          # shape (3,)
        bg     = tf.ones_like(rgb) * bg_rgb                 # [H,W,3]

        # Alpha composite: out = alpha*fg + (1-alpha)*bg
        composited = alpha * rgb + (1.0 - alpha) * bg       # [H,W,3]
        composited = tf.clip_by_value(composited, 0.0, 255.0)

        # Resize
        composited = tf.image.resize(composited, [IMG_SIZE, IMG_SIZE])
        return composited, label

    # Augmentation layers (applied BEFORE preprocess_input)
    augment_layer = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomBrightness(0.25),
        tf.keras.layers.RandomContrast(0.25),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
    ], name="augmentation")

    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input  # -> [-1,1]

    def aug_and_preprocess(img, label):
        img = augment_layer(img, training=True)
        img = preprocess(img)
        return img, label

    def just_preprocess(img, label):
        img = preprocess(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 8192), seed=SEED)
    ds = ds.map(load_and_composite, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(aug_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(just_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int):
    import tensorflow as tf
    from tensorflow.keras import layers, models  # type: ignore

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # Phase 1: freeze backbone

    inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(512, activation="relu")(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="TrafficSign_MobileNetV2_v3")
    return model, base_model


# ── main training function ────────────────────────────────────────────────────

def train(zip_path: str, output_dir: str):
    try:
        import tensorflow as tf
        log.info(f"TensorFlow {tf.__version__}")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            log.info(f"GPU: {gpus[0].name}")
        else:
            log.info("No GPU detected — training on CPU.")
    except ImportError:
        log.error("TensorFlow not found.  pip install tensorflow")
        sys.exit(1)

    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    )

    tmp_dir = tempfile.mkdtemp(prefix="traffic_sign_")
    try:
        # ── 1. Extract ────────────────────────────────────────────────────────
        dataset_root = extract_zip(zip_path, tmp_dir)

        # ── 2. Class map ──────────────────────────────────────────────────────
        class_map = read_class_map(dataset_root)

        # ── 3. Collect file paths (lightweight – no images in RAM yet) ────────
        log.info("Collecting image paths ...")
        all_paths, all_labels, sorted_ids = collect_paths(dataset_root, class_map)
        num_classes = len(sorted_ids)

        # ── 4. Train / val split ──────────────────────────────────────────────
        rng = np.random.default_rng(SEED)
        idx = rng.permutation(len(all_paths))
        split = int(len(all_paths) * (1 - VAL_SPLIT))
        train_idx, val_idx = idx[:split], idx[split:]

        train_paths  = [all_paths[i]  for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_paths    = [all_paths[i]  for i in val_idx]
        val_labels   = [all_labels[i] for i in val_idx]
        log.info(f"Train: {len(train_paths)}  |  Val: {len(val_paths)}")

        # ── 5. Build tf.data pipelines ────────────────────────────────────────
        log.info("Building tf.data pipelines ...")
        train_ds = make_tf_dataset(train_paths, train_labels, augment=True,  shuffle=True)
        val_ds   = make_tf_dataset(val_paths,   val_labels,   augment=False, shuffle=False)

        # ── 6. Build output directory & save class map ────────────────────────
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        idx_to_name = {str(i): class_map[cid] for i, cid in enumerate(sorted_ids)}
        class_map_path = out / "class_map.json"
        with open(class_map_path, "w", encoding="utf-8") as f:
            json.dump(idx_to_name, f, indent=2, ensure_ascii=False)
        log.info(f"Class map saved -> {class_map_path}")

        # ── 7. Build model ────────────────────────────────────────────────────
        model, base_model = build_model(num_classes)
        ckpt_path = str(out / "best_checkpoint.h5")

        # ── 8. Phase 1 — train head (backbone frozen) ─────────────────────────
        log.info("=== PHASE 1: Training classification head (backbone frozen) ===")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(PHASE1_LR),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        p1_cb = [
            EarlyStopping(monitor="val_accuracy", patience=5,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=2, min_lr=1e-6, verbose=1),
            ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                            save_best_only=True, verbose=1),
        ]
        h1 = model.fit(train_ds, validation_data=val_ds,
                       epochs=PHASE1_EPOCHS, callbacks=p1_cb)
        p1_best = max(h1.history.get("val_accuracy", [0])) * 100
        log.info(f"Phase 1 best val accuracy: {p1_best:.2f}%")

        # ── 9. Phase 2 — fine-tune top 40 backbone layers ────────────────────
        log.info("=== PHASE 2: Fine-tuning top 40 backbone layers ===")
        base_model.trainable = True
        fine_tune_at = len(base_model.layers) - 40
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(PHASE2_LR),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        p2_cb = [
            EarlyStopping(monitor="val_accuracy", patience=6,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                              patience=2, min_lr=1e-8, verbose=1),
            ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                            save_best_only=True, verbose=1),
        ]
        h2 = model.fit(train_ds, validation_data=val_ds,
                       epochs=PHASE2_EPOCHS, callbacks=p2_cb)
        p2_best = max(h2.history.get("val_accuracy", [0])) * 100
        log.info(f"Phase 2 best val accuracy: {p2_best:.2f}%")

        # ── 10. Save model + summary ──────────────────────────────────────────
        model_path = out / "traffic_model.h5"
        model.save(str(model_path))
        log.info(f"Model saved -> {model_path}")

        summary = {
            "architecture":      "MobileNetV2_transfer_learning",
            "num_classes":       num_classes,
            "image_size":        IMG_SIZE,
            "total_images":      len(all_paths),
            "best_val_accuracy": round(float(p2_best), 2),
            "best_val_loss":     round(float(min(h2.history.get("val_loss", [0]))), 4),
            "phase1_epochs":     len(h1.history["loss"]),
            "phase2_epochs":     len(h2.history["loss"]),
            "preprocessing":     "mobilenet_v2",
        }
        summary_path = out / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Training summary:\n{json.dumps(summary, indent=2)}")
        log.info(f"Training complete!  Model at: {model_path}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info("Temp directory cleaned up.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an enhanced MobileNetV2 CNN on the Indian Traffic Sign dataset."
    )
    parser.add_argument("--zip",    default=r"archive (5).zip",
                        help="Path to dataset zip (default: archive (5).zip)")
    parser.add_argument("--output", default="models",
                        help="Output directory (default: models)")
    args = parser.parse_args()

    train(os.path.abspath(args.zip), os.path.abspath(args.output))
