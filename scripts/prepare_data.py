"""원본 parquet → 눈 문맥 패치(npz) + 이미지별 메타(얼굴 박스 · 정답 눈 위치).

- 분할은 parquet 파일 단위(같은 프롬프트 묶음이 train/test 에 섞이지 않게) 70/15/15.
- 눈 패치 배율은 양눈 거리(IOD) 기준: 정답 bbox 크기를 쓰면 감은 눈/뜬 눈의 bbox 크기 차이가
  배율 단서로 새어 들어간다.
- Haar / LBP 얼굴 검출 결과를 같이 저장해 기하 비율 보정(calibrate_geometry)에 쓴다.

    python -m scripts.prepare_data
"""
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.classify.augment import CTX, CTX_SCALE
from src.classify.datasets import PREPARED, SPLITS
from src.config import ROOT
from src.detect.eye_mouth import PATCH_IOD, crop_square
from src.detect.face import FaceDetector, Preprocessor

RAW = ROOT / "data" / "raw" / "closed_open_eyes"
MIN_IOD = 16


def split_of(rank: int) -> int:
    """클래스 안에서의 파일 순번 → 0 train / 1 val / 2 test (20개 주기로 14:3:3)."""
    r = rank % 20
    return 0 if r < 14 else 1 if r < 17 else 2


def decode(buf: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def eye_centers(row) -> list[tuple[float, float]]:
    cs = [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in (row.Left_eye_react, row.Right_eye_react)]
    return sorted(cs)  # 영상 기준 왼쪽 → 오른쪽


def process_file(args) -> tuple[list, list]:
    path, split = args
    cv2.setNumThreads(1)
    df = pd.read_parquet(path)
    det = {k: FaceDetector(k) for k in ("haar", "lbp")}
    pre_raw, pre_clahe = Preprocessor(clahe=False), Preprocessor(clahe=True)
    patches, meta = [], []
    for i, row in enumerate(df.itertuples()):
        img = decode(row.Image_data["file"])
        if img is None:
            continue
        g_raw, g_clahe = pre_raw(img), pre_clahe(img)
        (lx, ly), (rx, ry) = cs = eye_centers(row)
        iod = float(np.hypot(rx - lx, ry - ly))
        if iod < MIN_IOD:
            continue
        side = PATCH_IOD * iod * CTX_SCALE
        y = 1 if row.Label == "closed_eyes" else 0
        rec = dict(file=Path(path).name, row=i, y=y, split=split, lx=lx, ly=ly, rx=rx, ry=ry, iod=iod,
                   w=img.shape[1], h=img.shape[0])
        for k, d in det.items():
            for pre_name, g in (("raw", g_raw), ("clahe", g_clahe)):
                boxes = d.detect(g)
                hit = next((b for b in boxes if _contains(b, cs)), None)
                rec[f"{k}_{pre_name}_n"] = len(boxes)
                rec[f"{k}_{pre_name}_box"] = "" if hit is None else " ".join(map(str, hit))
        meta.append(rec)
        for e, c in enumerate(cs):
            patches.append((crop_square(g_raw, c, side, CTX), crop_square(g_clahe, c, side, CTX),
                            y, split, len(meta) - 1, e))
    return patches, meta


def _contains(box, centers) -> bool:
    x, y, w, h = box
    return all(x <= cx <= x + w and y <= cy <= y + h for cx, cy in centers)


def main() -> None:
    files = sorted(RAW.glob("dataset_*.parquet"))
    # 파일 라벨(앞 행)로 클래스를 나누고, 클래스 안 순번으로 분할
    by_label: dict[str, list[Path]] = {}
    for f in files:
        lab = pd.read_parquet(f, columns=["Label"]).Label.iloc[0]
        by_label.setdefault(lab, []).append(f)
    jobs = [(str(f), split_of(i)) for fs in by_label.values() for i, f in enumerate(fs)]
    print({k: len(v) for k, v in by_label.items()}, "files")
    with Pool() as pool:
        results = pool.map(process_file, jobs)
    patches, meta, offset = [], [], 0
    for ps, ms in results:
        patches += [(a, b, y, s, img + offset, e) for a, b, y, s, img, e in ps]
        meta += ms
        offset += len(ms)
    PREPARED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PREPARED / "eyes.npz",
        ctx_raw=np.stack([p[0] for p in patches]), ctx_clahe=np.stack([p[1] for p in patches]),
        y=np.array([p[2] for p in patches], np.uint8), split=np.array([p[3] for p in patches], np.uint8),
        img=np.array([p[4] for p in patches], np.int32), eye=np.array([p[5] for p in patches], np.uint8))
    m = pd.DataFrame(meta)
    m.to_csv(PREPARED / "faces.csv", index=False)
    inv = {v: k for k, v in SPLITS.items()}
    print(m.groupby([m.split.map(inv), "y"]).size().unstack().rename(columns={0: "open", 1: "closed"}))
    for k in ("haar", "lbp"):
        for p in ("raw", "clahe"):
            print(f"{k}/{p} face detected: {(m[f'{k}_{p}_box'] != '').mean():.3f}")
    print(f"patches: {len(patches)} -> {PREPARED / 'eyes.npz'}")


if __name__ == "__main__":
    main()
