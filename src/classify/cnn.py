"""경량 CNN 눈 상태 분류기: 모델 정의와 TFLite 추론."""
import os
from pathlib import Path

import numpy as np

from src.config import MODELS
from src.detect.eye_mouth import PATCH

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

KERAS_PATH = MODELS / "eye_cnn.keras"
TFLITE_PATHS = {
    "fp32": MODELS / "eye_cnn_fp32.tflite",
    "fp16": MODELS / "eye_cnn_fp16.tflite",
    "int8": MODELS / "eye_cnn_int8.tflite",
}
DEFAULT_TFLITE = TFLITE_PATHS["int8"]


def build_model(width: int = 16, dropout: float = 0.3):
    """32×32 회색 → 감은 눈 확률. 약 2.5만 파라미터."""
    from tensorflow import keras
    from tensorflow.keras import layers

    def block(x, filters, n=1):
        for _ in range(n):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        return layers.MaxPool2D()(x)

    inp = keras.Input((PATCH, PATCH, 1))
    x = layers.Rescaling(1 / 255.0)(inp)
    x = block(x, width, 2)       # 16×16
    x = block(x, width * 2)      # 8×8
    x = block(x, width * 4)      # 4×4
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="eye_cnn")


class CnnClassifier:
    """TFLite 인터프리터로 추론한다(Keras 호출보다 프레임당 오버헤드가 훨씬 작다)."""

    def __init__(self, path: Path | str = DEFAULT_TFLITE, threads: int = 1):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        self.batch = None

    def _resize(self, n: int) -> None:
        if self.batch != n:
            self.interp.resize_tensor_input(self.inp["index"], [n, PATCH, PATCH, 1])
            self.interp.allocate_tensors()
            self.inp = self.interp.get_input_details()[0]
            self.out = self.interp.get_output_details()[0]
            self.batch = n

    def predict_closed(self, patches: np.ndarray) -> np.ndarray:
        """(N, 32, 32) uint8 → 감은 눈 확률 (N,)."""
        n = len(patches)
        if n == 0:
            return np.zeros(0, np.float32)
        self._resize(n)
        x = patches.reshape(n, PATCH, PATCH, 1)
        x = _quantize(x.astype(np.float32), self.inp)
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = _dequantize(self.interp.get_tensor(self.out["index"]), self.out)
        return y.reshape(n).astype(np.float32)


def _quantize(x: np.ndarray, detail: dict) -> np.ndarray:
    if detail["dtype"] in (np.int8, np.uint8):
        scale, zero = detail["quantization"]
        info = np.iinfo(detail["dtype"])
        return np.clip(np.round(x / scale + zero), info.min, info.max).astype(detail["dtype"])
    return x.astype(detail["dtype"])


def _dequantize(y: np.ndarray, detail: dict) -> np.ndarray:
    if detail["dtype"] in (np.int8, np.uint8):
        scale, zero = detail["quantization"]
        return (y.astype(np.float32) - zero) * scale
    return y.astype(np.float32)
