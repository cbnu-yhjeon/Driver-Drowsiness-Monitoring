"""하품 판별: 입 영역 HoG + SVM.

입 벌림 비율(어두운 영역의 세로/가로) 규칙은 실제 얼굴에서 AUC 0.44~0.69 로 쓸 수 없어서
(experiments/eval_yawn 참고) 눈 상태와 같은 HoG+SVM 방식으로 바꿨다.
학습은 experiments/eval_yawn 이 한다.
"""
import cv2
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import MODELS

YAWN_PATH = MODELS / "yawn_svm.joblib"
MOUTH = 64
# 64×64 창, 16×16 블록, 8px 이동, 8×8 셀, 9방향 → 7×7 블록 × 36 = 1764차원
HOG = cv2.HOGDescriptor((MOUTH, MOUTH), (16, 16), (8, 8), (8, 8), 9)


def mouth_features(rois: list[np.ndarray]) -> np.ndarray:
    feats = []
    for r in rois:
        r = cv2.resize(r, (MOUTH, MOUTH), interpolation=cv2.INTER_AREA)
        feats.append(HOG.compute(np.ascontiguousarray(r)).ravel())
    return np.stack(feats).astype(np.float32)


class YawnClassifier:
    def __init__(self, model):
        self.model = model

    @classmethod
    def fit(cls, rois: list[np.ndarray], y: np.ndarray, c: float = 3.0, seed: int = 0) -> "YawnClassifier":
        m = make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", probability=True, random_state=seed))
        return cls(m.fit(mouth_features(rois), y))

    @classmethod
    def load(cls, path=YAWN_PATH) -> "YawnClassifier":
        return cls(joblib.load(path))

    def save(self, path=YAWN_PATH) -> None:
        joblib.dump(self.model, path, compress=3)

    def predict_yawn(self, rois: list[np.ndarray]) -> np.ndarray:
        """입 영역 회색 영상들 → 하품 확률."""
        return self.model.predict_proba(mouth_features(rois))[:, 1]
