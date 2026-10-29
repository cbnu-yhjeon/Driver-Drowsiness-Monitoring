"""하품 판별: 입 벌림 비율 규칙(README 초안) vs 입 영역 HoG+SVM. 데이터: c3rl/yawning-people 일부.

얼굴 검출(Haar + CLAHE) → 기하 비율 눈 위치 → 입 영역. 이미지 이름 해시로 학습 70 / 시험 30.
규칙은 학습 분할에서 F1 최대 임계값을, SVM 은 학습 분할 5겹 교차검증으로 C 를 고른다.
학습된 SVM 은 models/yawn_svm.joblib 로 저장되어 실행 경로에서 쓰인다.

    python -m experiments.eval_yawn [--montage out.png]
"""
import argparse
import hashlib

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments.common import binary_metrics, save
from src.classify.yawn_svm import YawnClassifier, mouth_features
from src.config import ROOT
from src.detect.eye_mouth import EyeLocator, FaceGeometry, mouth_box, mouth_opening_ratio
from src.detect.face import FaceDetector, Preprocessor

YAWN = ROOT / "data" / "raw" / "yawning"


def is_test(name: str) -> bool:
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 10 >= 7


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    cands = np.unique(s)
    f1s = [binary_metrics(y, s >= c)["f1"] for c in cands]
    return float(cands[int(np.argmax(f1s))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--montage", help="입 영역 모아보기 이미지 (위 2줄 하품, 아래 2줄 하품 아님)")
    args = ap.parse_args()
    det, pre = FaceDetector("haar"), Preprocessor(clahe=True)
    loc = EyeLocator(FaceGeometry.load("haar"))
    rois, ys, test, n_total = [], [], [], {0: 0, 1: 0}
    for label, sub in ((1, "yawning"), (0, "notyawning")):
        for f in sorted((YAWN / sub).glob("*.png")):
            n_total[label] += 1
            g = pre(cv2.imread(str(f)))
            box = det.detect_largest(g)
            if box is None:
                continue
            x, y, w, h = mouth_box(loc.locate(g, box), g.shape)
            rois.append(g[y:y + h, x:x + w])
            ys.append(label)
            test.append(is_test(f.name))
    ys, test = np.array(ys), np.array(test)
    face_rate = {"yawning": float((ys == 1).sum() / n_total[1]), "not_yawning": float((ys == 0).sum() / n_total[0])}
    print("images", n_total, "face rate", face_rate, "train", (~test).sum(), "test", test.sum())

    # 1) 규칙: 입 벌림 비율
    ratio = np.array([mouth_opening_ratio(r) for r in rois])
    thr = best_threshold(ys[~test], ratio[~test])
    rule = {"threshold": thr, "test_auc": float(roc_auc_score(ys[test], ratio[test])),
            "test": binary_metrics(ys[test], ratio[test] >= thr)}
    # 2) HoG + SVM
    x = mouth_features(rois)
    grid = GridSearchCV(make_pipeline(StandardScaler(), SVC(gamma="scale")), {"svc__C": [1, 3, 10]},
                        cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1")
    grid.fit(x[~test], ys[~test])
    c = grid.best_params_["svc__C"]
    clf = YawnClassifier.fit([r for r, t in zip(rois, test) if not t], ys[~test], c=c)
    prob = clf.predict_yawn([r for r, t in zip(rois, test) if t])
    svm = {"C": c, "cv_f1": float(grid.best_score_), "test_auc": float(roc_auc_score(ys[test], prob)),
           "test": binary_metrics(ys[test], prob >= 0.5)}
    clf.save()
    out = {"images": n_total, "face_rate": face_rate, "n_train": int((~test).sum()), "n_test": int(test.sum()),
           "rule_mouth_ratio": rule, "hog_svm": svm}
    for name, r in (("rule", rule), ("hog_svm", svm)):
        print(f"{name:8s} test AUC={r['test_auc']:.3f} F1={r['test']['f1']:.3f} acc={r['test']['acc']:.3f}")
    save("yawn", out)

    if args.montage:
        pick = [i for i in np.flatnonzero(ys == 1)[:16]] + [i for i in np.flatnonzero(ys == 0)[:16]]
        tiles = [cv2.resize(rois[i], (96, 96)) for i in pick]
        cv2.imwrite(args.montage, np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, 32, 8)]))


if __name__ == "__main__":
    main()
