"""고전 기준선: HoG 특징 + SVM(RBF) 눈 상태 분류.

    python -m src.classify.hog_svm          # 학습 → models/eye_svm.joblib
"""
import argparse
import json
import time

import cv2
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.classify.augment import augment, center_crop
from src.classify.datasets import load_eyes
from src.config import MODELS
from src.detect.eye_mouth import PATCH

SVM_PATH = MODELS / "eye_svm.joblib"
# 32×32 창, 16×16 블록, 8px 이동, 8×8 셀, 9방향 → 3×3 블록 × 4셀 × 9 = 324차원
HOG = cv2.HOGDescriptor((PATCH, PATCH), (16, 16), (8, 8), (8, 8), 9)


def hog_features(patches: np.ndarray) -> np.ndarray:
    return np.stack([HOG.compute(np.ascontiguousarray(p)).ravel() for p in patches]).astype(np.float32)


class HogSvmClassifier:
    """SVM 결정값을 검증 분할에서 맞춘 로지스틱(Platt)으로 확률화한다."""

    def __init__(self, svm, platt: LogisticRegression):
        self.svm, self.platt = svm, platt

    @classmethod
    def load(cls, path=SVM_PATH) -> "HogSvmClassifier":
        d = joblib.load(path)
        return cls(d["svm"], d["platt"])

    def save(self, path=SVM_PATH) -> None:
        joblib.dump({"svm": self.svm, "platt": self.platt}, path, compress=3)

    def decision(self, patches: np.ndarray) -> np.ndarray:
        return self.svm.decision_function(hog_features(patches))

    def predict_closed(self, patches: np.ndarray) -> np.ndarray:
        """(N, 32, 32) uint8 → 감은 눈 확률 (N,)."""
        return self.platt.predict_proba(self.decision(patches).reshape(-1, 1))[:, 1]


def build_training_set(ctx: np.ndarray, y: np.ndarray, copies: int, rng) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [center_crop(ctx)], [y]
    for _ in range(copies):
        xs.append(np.stack([augment(c, rng) for c in ctx]))
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=1, help="증강 사본 수")
    ap.add_argument("--max-train", type=int, default=24000, help="SVM 학습 표본 상한 (RBF SVM 은 O(n^2))")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    # raw 와 CLAHE 패치를 모두 학습에 넣어 전처리 유무에 둔감하게
    xs, ys = [], []
    for variant in ("raw", "clahe"):
        ctx, y = load_eyes("train", variant)
        x, yy = build_training_set(ctx, y, args.copies, rng)
        xs.append(x), ys.append(yy)
    x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
    if len(y_tr) > args.max_train:
        keep = rng.choice(len(y_tr), args.max_train, replace=False)
        x_tr, y_tr = x_tr[keep], y_tr[keep]
    x_tr = hog_features(x_tr)
    ctx_va, y_va = load_eyes("val", "raw")
    x_va = hog_features(center_crop(ctx_va))
    print(f"train {x_tr.shape}, val {x_va.shape}")

    best = None
    for c in (1.0, 3.0, 10.0):
        t = time.time()
        m = make_pipeline(StandardScaler(), SVC(C=c, gamma="scale"))
        m.fit(x_tr, y_tr)
        f1 = f1_score(y_va, m.predict(x_va))
        print(f"C={c:<5} val F1={f1:.4f} ({time.time() - t:.0f}s, SV={m[-1].n_support_.sum()})")
        if best is None or f1 > best[0]:
            best = (f1, c, m)
    f1, c, m = best
    platt = LogisticRegression().fit(m.decision_function(x_va).reshape(-1, 1), y_va)
    MODELS.mkdir(exist_ok=True)
    HogSvmClassifier(m, platt).save()
    (MODELS / "eye_svm.json").write_text(json.dumps({"C": c, "val_f1": f1, "n_train": int(len(y_tr))}, indent=2))
    print(f"best C={c} val F1={f1:.4f} -> {SVM_PATH}")


if __name__ == "__main__":
    main()
