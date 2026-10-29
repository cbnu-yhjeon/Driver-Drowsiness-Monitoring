"""G1 · G3: 눈 패치 분류 성능 (정답 눈 위치에서 자른 패치).

분류기: HoG+SVM, CNN(TFLite fp32 / fp16 / int8)
조건: 일반 / 저조도(합성) / 안경 반사(합성) × 전처리 없음 / CLAHE

    python -m experiments.eval_eye_state
"""
import time

import numpy as np

from experiments.common import CONDITIONS, apply_condition, binary_metrics, save, test_images
from src.classify.cnn import TFLITE_PATHS, CnnClassifier
from src.classify.hog_svm import HogSvmClassifier
from src.detect.eye_mouth import PATCH_IOD, crop_square
from src.detect.face import Preprocessor


def classifiers() -> dict:
    out = {"hog_svm": HogSvmClassifier.load()}
    for kind, path in TFLITE_PATHS.items():
        out[f"cnn_{kind}"] = CnnClassifier(path)
    return out


def per_call_ms(clf, patches: np.ndarray, reps: int = 300) -> float:
    pair = patches[:2]
    clf.predict_closed(pair)
    t = time.perf_counter()
    for _ in range(reps):
        clf.predict_closed(pair)
    return (time.perf_counter() - t) / reps * 1e3


def main() -> None:
    imgs = test_images()
    clfs = classifiers()
    pre = {"raw": Preprocessor(clahe=False), "clahe": Preprocessor(clahe=True)}
    results = {"n_images": len(imgs), "rows": [], "latency_ms_per_frame(2 eyes)": {}}
    for cond in CONDITIONS:
        for pname, p in pre.items():
            patches, ys = [], []
            for i, im in enumerate(imgs):
                g = p(apply_condition(im, cond, seed=i))
                for c in im.centers:
                    patches.append(crop_square(g, c, PATCH_IOD * im.iod))
                    ys.append(im.y)
            patches, ys = np.stack(patches), np.array(ys)
            for name, clf in clfs.items():
                prob = clf.predict_closed(patches)
                m = binary_metrics(ys, prob >= 0.5)
                results["rows"].append({"condition": cond, "preprocess": pname, "classifier": name, **m})
                print(f"{cond:9s} {pname:6s} {name:10s} acc={m['acc']:.4f} F1={m['f1']:.4f} "
                      f"FPR={m['fpr']:.3f} FNR={m['fnr']:.3f}")
    for name, clf in clfs.items():
        results["latency_ms_per_frame(2 eyes)"][name] = per_call_ms(clf, patches)
    print({k: round(v, 3) for k, v in results["latency_ms_per_frame(2 eyes)"].items()})
    save("eye_state", results)


if __name__ == "__main__":
    main()
