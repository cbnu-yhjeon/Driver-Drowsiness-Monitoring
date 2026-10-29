"""G1 · G3 · D1~D5: 정지 영상 전체 파이프라인 (얼굴 검출 → 눈 위치 추정 → 분류).

정답 눈 위치를 쓰지 않는다. 가장 큰 얼굴이 정답 눈을 포함하지 않으면 검출 실패로 센다.
LK 추적(D3 이후)은 정지 영상에서 효과가 없으므로 프레임 단위 수치는 D2 와 같고,
시간축 효과는 eval_temporal 에서 본다.

    python -m experiments.eval_pipeline
"""
import itertools
import time

import numpy as np

from experiments.common import CONDITIONS, apply_condition, binary_metrics, save, test_images
from src.classify.cnn import CnnClassifier
from src.classify.hog_svm import HogSvmClassifier
from src.detect.eye_mouth import EyeLocator, FaceGeometry
from src.detect.face import FaceDetector, Preprocessor

D_CONFIGS = {  # 정지 영상에서 의미 있는 축만: 검출기, CLAHE, 분류기
    "D1": ("haar", False, "hog_svm"),
    "D2": ("haar", False, "cnn"),
    "D3": ("haar", False, "cnn"),
    "D4": ("haar", True, "cnn"),
    "D5": ("lbp", True, "cnn"),
}


def contains(box, centers) -> bool:
    x, y, w, h = box
    return all(x <= cx <= x + w and y <= cy <= y + h for cx, cy in centers)


def main() -> None:
    imgs = test_images()
    clfs = {"hog_svm": HogSvmClassifier.load(), "cnn": CnnClassifier()}
    rows, loc_err = [], {}
    for det_kind, clahe, cond in itertools.product(("haar", "lbp"), (False, True), CONDITIONS):
        det = FaceDetector(det_kind)
        pre = Preprocessor(clahe=clahe)
        locators = {"geometry": EyeLocator(FaceGeometry.load(det_kind)),
                    "haar_eye": EyeLocator(FaceGeometry.load(det_kind), refine=True)}
        found, ys, t_det = [], [], 0.0
        patches = {k: [] for k in locators}
        errs = {k: [] for k in locators}
        for i, im in enumerate(imgs):
            g = pre(apply_condition(im, cond, seed=i))
            t = time.perf_counter()
            box = det.detect_largest(g)
            t_det += time.perf_counter() - t
            ok = box is not None and contains(box, im.centers)
            found.append(ok)
            ys.append(im.y)
            if not ok:
                continue
            for k, loc in locators.items():
                eyes = loc.locate(g, box)
                patches[k].append(loc.patches(g, eyes))
                errs[k] += [np.hypot(p[0] - q[0], p[1] - q[1]) / im.iod for p, q in zip(eyes.centers, im.centers)]
        found, ys = np.array(found), np.array(ys)
        for k in locators:
            loc_err[f"{det_kind}/{'clahe' if clahe else 'raw'}/{cond}/{k}"] = {
                "median_iod": float(np.median(errs[k])) if errs[k] else None,
                "p90_iod": float(np.percentile(errs[k], 90)) if errs[k] else None}
        for (lname, ps), (cname, clf) in itertools.product(patches.items(), clfs.items()):
            if not ps:
                continue
            prob = clf.predict_closed(np.concatenate(ps)).reshape(-1, 2).mean(axis=1)
            pred = prob >= 0.5
            m = binary_metrics(ys[found], pred)
            correct = int((pred == ys[found].astype(bool)).sum())
            rows.append({"detector": det_kind, "clahe": clahe, "condition": cond, "locator": lname,
                         "classifier": cname, "face_rate": float(found.mean()),
                         "e2e_acc": correct / len(ys),   # 얼굴 못 찾은 이미지는 오답 처리
                         "det_ms": t_det / len(imgs) * 1e3, **m})
            print(f"{det_kind:4s} clahe={clahe!s:5s} {cond:8s} {lname:8s} {cname:7s} face={found.mean():.3f} "
                  f"acc|face={m['acc']:.4f} F1={m['f1']:.4f} e2e={correct / len(ys):.4f}")

    def pick(det, clahe, clf, cond, loc="geometry"):
        return next(r for r in rows if r["detector"] == det and r["clahe"] == clahe and r["classifier"] == clf
                    and r["condition"] == cond and r["locator"] == loc)

    d_table = {d: {cond: pick(*cfg, cond) for cond in CONDITIONS} for d, cfg in D_CONFIGS.items()}
    for d, per in d_table.items():
        print(d, {c: f"face={v['face_rate']:.3f} F1={v['f1']:.3f} e2e={v['e2e_acc']:.3f}" for c, v in per.items()})
    save("pipeline", {"n_images": len(imgs), "rows": rows, "eye_localization_error": loc_err, "D": d_table})


if __name__ == "__main__":
    main()
