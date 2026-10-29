"""경량 CNN 학습과 TFLite 변환(fp32 / fp16 / int8).

    python -m src.classify.train_cnn
"""
import argparse
import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.classify.augment import augment, center_crop
from src.classify.cnn import KERAS_PATH, TFLITE_PATHS, build_model
from src.classify.datasets import load_eyes
from src.config import MODELS


class AugmentedEyes(keras.utils.Sequence):
    """매 에폭 새로 증강. raw / CLAHE 문맥 패치 중 하나를 무작위로 고른다."""

    def __init__(self, variants: list[np.ndarray], y: np.ndarray, batch: int, seed: int):
        super().__init__()
        self.variants, self.y, self.batch = variants, y, batch
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(y))

    def __len__(self) -> int:
        return int(np.ceil(len(self.y) / self.batch))

    def __getitem__(self, i):
        idx = self.order[i * self.batch:(i + 1) * self.batch]
        src = self.rng.integers(len(self.variants), size=len(idx))
        x = np.stack([augment(self.variants[s][j], self.rng) for s, j in zip(src, idx)])
        return x[..., None].astype(np.float32), self.y[idx].astype(np.float32)

    def on_epoch_end(self) -> None:
        self.order = self.rng.permutation(len(self.y))


def export_tflite(model, rep_patches: np.ndarray) -> dict:
    sizes = {}
    for kind, path in TFLITE_PATHS.items():
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        if kind == "fp16":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
        elif kind == "int8":
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.representative_dataset = lambda: (
                [p[None, ..., None].astype(np.float32)] for p in rep_patches)
            conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type = tf.uint8   # 입력은 픽셀값 그대로 (scale 1, zero 0)
            conv.inference_output_type = tf.uint8
        path.write_bytes(conv.convert())
        sizes[kind] = path.stat().st_size
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    keras.utils.set_random_seed(args.seed)

    raw, y = load_eyes("train", "raw")
    clahe, _ = load_eyes("train", "clahe")
    va, y_va = load_eyes("val", "raw")
    x_va = center_crop(va)[..., None].astype(np.float32)
    pos = y.mean()
    class_weight = {0: 0.5 / (1 - pos), 1: 0.5 / pos}
    print(f"train {len(y)} (closed {pos:.2f}), val {len(y_va)}")

    model = build_model(args.width)
    model.compile(optimizer=keras.optimizers.Adam(2e-3), loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    model.summary(print_fn=lambda s: None)
    print("params:", model.count_params())
    hist = model.fit(
        AugmentedEyes([raw, clahe], y, args.batch, args.seed),
        validation_data=(x_va, y_va.astype(np.float32)),
        epochs=args.epochs, class_weight=class_weight, verbose=2,
        callbacks=[keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5),
                   keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    MODELS.mkdir(exist_ok=True)
    model.save(KERAS_PATH)
    loss, acc, auc = model.evaluate(x_va, y_va, verbose=0)
    rng = np.random.default_rng(args.seed)
    rep = center_crop(raw[rng.choice(len(raw), 300, replace=False)])
    sizes = export_tflite(model, rep)
    info = {"params": int(model.count_params()), "epochs_run": len(hist.history["loss"]),
            "val_loss": loss, "val_acc": acc, "val_auc": auc, "tflite_bytes": sizes}
    (MODELS / "eye_cnn.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
