"""준비된 눈 패치 데이터 로드 (scripts/prepare_data.py 산출물) 와 CEW 로더."""
from pathlib import Path

import cv2
import numpy as np

from src.config import ROOT
from src.detect.eye_mouth import PATCH

PREPARED = ROOT / "data" / "prepared"
SPLITS = {"train": 0, "val": 1, "test": 2}


def load_eyes(split: str | None = None, variant: str = "raw") -> tuple[np.ndarray, np.ndarray]:
    """문맥 패치 (N, 48, 48) 와 라벨 (1 = 감음). variant: raw | clahe."""
    d = np.load(PREPARED / "eyes.npz")
    ctx, y, s = d[f"ctx_{variant}"], d["y"], d["split"]
    if split is not None:
        keep = s == SPLITS[split]
        ctx, y = ctx[keep], y[keep]
    return ctx, y


def load_cew_patches(root: Path | str = ROOT / "data" / "cew" / "dataset_B_Eye_Images") -> tuple[np.ndarray, np.ndarray]:
    """CEW 24×24 눈 패치(수동 다운로드)를 32×32 로 키워 반환. 폴더: closed*/open* 하위의 jpg."""
    root = Path(root)
    xs, ys = [], []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        label = 1 if sub.name.lower().startswith("closed") else 0 if sub.name.lower().startswith("open") else None
        if label is None:
            continue
        for f in sorted(sub.glob("*.jpg")):
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                xs.append(cv2.resize(img, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR))
                ys.append(label)
    if not xs:
        raise FileNotFoundError(f"CEW 패치를 찾지 못함: {root}")
    return np.stack(xs), np.array(ys, dtype=np.uint8)
