"""실험 공통: 테스트 분할 원본 이미지, 조건 합성, 지표, 결과 저장."""
import json
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd

from src.classify.augment import add_glare, simulate_low_light
from src.classify.datasets import PREPARED, SPLITS
from src.config import ROOT
from src.detect.eye_mouth import PATCH_IOD

RAW = ROOT / "data" / "raw" / "closed_open_eyes"
RESULTS = ROOT / "results"
CONDITIONS = ("normal", "lowlight", "glare")


@dataclass
class TestImage:
    gray: np.ndarray
    y: int
    centers: list[tuple[float, float]]
    iod: float


@lru_cache(maxsize=1)
def test_images(split: str = "test") -> list[TestImage]:
    m = pd.read_csv(PREPARED / "faces.csv", keep_default_na=False)
    m = m[m.split == SPLITS[split]]
    out = []
    for fname, rows in m.groupby("file"):
        df = pd.read_parquet(RAW / fname)
        for r in rows.itertuples():
            img = cv2.imdecode(np.frombuffer(df.iloc[r.row].Image_data["file"], np.uint8), cv2.IMREAD_GRAYSCALE)
            out.append(TestImage(img, int(r.y), [(r.lx, r.ly), (r.rx, r.ry)], float(r.iod)))
    return out


def apply_condition(img: TestImage, cond: str, seed: int) -> np.ndarray:
    """조건을 원본(전처리 전) 영상에 합성. 같은 seed 면 같은 결과."""
    rng = np.random.default_rng(seed)
    if cond == "normal":
        return img.gray
    if cond == "lowlight":
        return simulate_low_light(img.gray, rng)
    if cond == "glare":
        g = img.gray.copy()
        s = int(PATCH_IOD * img.iod)
        for cx, cy in img.centers:
            x0, y0 = max(int(cx - s / 2), 0), max(int(cy - s / 2), 0)
            roi = g[y0:y0 + s, x0:x0 + s]
            if roi.size:
                g[y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]] = add_glare(roi, rng)
        return g
    raise ValueError(cond)


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp, tn = int((y & pred).sum()), int((~y & ~pred).sum())
    fp, fn = int((~y & pred).sum()), int((y & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(len(y)), "acc": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def save(name: str, data) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print("->", RESULTS / f"{name}.json")
