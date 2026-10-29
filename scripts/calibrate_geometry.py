"""얼굴 박스 → 눈 중심 기하 비율을 학습 분할에서 추정하고 검증 분할에서 오차를 잰다.

    python -m scripts.calibrate_geometry
"""
import json

import numpy as np
import pandas as pd

from src.classify.datasets import PREPARED, SPLITS
from src.config import ROOT
from src.detect.eye_mouth import GEOMETRY_FILE

RESULTS = ROOT / "results"


def relative(m: pd.DataFrame, col: str) -> pd.DataFrame:
    m = m[m[col] != ""].copy()
    b = np.array([list(map(float, s.split())) for s in m[col]])
    x, y, w, h = b.T
    out = pd.DataFrame({"split": m.split.values, "iod": m.iod.values, "w": w})
    for side in ("l", "r"):
        out[f"{side}x"] = (m[f"{side}x"].values - x) / w
        out[f"{side}y"] = (m[f"{side}y"].values - y) / h
    return out


def main() -> None:
    m = pd.read_csv(PREPARED / "faces.csv", keep_default_na=False)
    geometry, report = {}, {}
    for det in ("haar", "lbp"):
        # 전처리(raw/clahe)에 따라 박스가 거의 같으므로 둘을 합쳐 추정
        rel = pd.concat([relative(m, f"{det}_{p}_box") for p in ("raw", "clahe")])
        tr, va = rel[rel.split == SPLITS["train"]], rel[rel.split == SPLITS["val"]]
        g = {"left_eye": [float(tr.lx.median()), float(tr.ly.median())],
             "right_eye": [float(tr.rx.median()), float(tr.ry.median())]}
        geometry[det] = g
        # 검증 오차: IOD 대비 눈 중심 위치 오차
        err = []
        for side, key in (("l", "left_eye"), ("r", "right_eye")):
            dx = (va[f"{side}x"] - g[key][0]) * va.w
            dy = (va[f"{side}y"] - g[key][1]) * va.w  # 얼굴 박스는 정사각형
            err.append(np.hypot(dx, dy) / va.iod)
        err = np.concatenate(err)
        iod_ratio = (tr.iod / tr.w)
        report[det] = {"geometry": g, "n_train": int(len(tr)), "n_val": int(len(va)),
                       "val_center_err_iod_median": float(np.median(err)),
                       "val_center_err_iod_p90": float(np.percentile(err, 90)),
                       "iod_over_face_w_median": float(iod_ratio.median())}
        print(det, json.dumps(report[det], indent=1))
    GEOMETRY_FILE.write_text(json.dumps(geometry, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "geometry.json").write_text(json.dumps(report, indent=2))
    print("->", GEOMETRY_FILE)


if __name__ == "__main__":
    main()
