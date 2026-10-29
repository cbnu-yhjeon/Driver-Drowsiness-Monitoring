"""얼굴 검출: Viola-Jones(Haar) · LBP cascade, CLAHE 저조도 보정."""
from pathlib import Path

import cv2
import numpy as np

from src.config import MODELS

Box = tuple[int, int, int, int]  # x, y, w, h

CASCADE_FILES = {
    "haar": "haarcascade_frontalface_default.xml",
    "lbp": "lbpcascade_frontalface_improved.xml",
}


def load_cascade(name: str) -> cv2.CascadeClassifier:
    path = MODELS / "cascades" / name
    if not path.exists() and name.startswith("haar"):
        path = Path(cv2.data.haarcascades) / name  # pip opencv 에 동봉된 Haar cascade
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise FileNotFoundError(f"cascade 를 읽지 못함: {path} (python -m scripts.download_data 실행)")
    return cascade


class Preprocessor:
    """BGR/Gray → Gray (+ CLAHE)."""

    def __init__(self, clahe: bool = True, clip: float = 2.0, tile: int = 8):
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)) if clahe else None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray) if self.clahe is not None else gray


class FaceDetector:
    """가장 큰 정면 얼굴 하나를 찾는다. 속도를 위해 축소 영상에서 검출한다."""

    def __init__(self, kind: str = "haar", detect_width: int = 320,
                 scale_factor: float = 1.1, min_neighbors: int = 3, min_size_frac: float = 0.12):
        if kind not in CASCADE_FILES:
            raise ValueError(f"detector 는 {list(CASCADE_FILES)} 중 하나: {kind}")
        self.kind = kind
        self.cascade = load_cascade(CASCADE_FILES[kind])
        self.detect_width = detect_width
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size_frac = min_size_frac

    def detect(self, gray: np.ndarray) -> list[Box]:
        h, w = gray.shape[:2]
        s = self.detect_width / w if 0 < self.detect_width < w else 1.0
        small = cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s != 1.0 else gray
        m = max(int(min(small.shape[:2]) * self.min_size_frac), 20)
        found = self.cascade.detectMultiScale(small, scaleFactor=self.scale_factor,
                                              minNeighbors=self.min_neighbors, minSize=(m, m))
        boxes = [tuple(int(round(v / s)) for v in b) for b in found]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def detect_largest(self, gray: np.ndarray) -> Box | None:
        boxes = self.detect(gray)
        return boxes[0] if boxes else None
