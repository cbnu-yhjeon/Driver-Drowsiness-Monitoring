"""학습된 모델(models/)을 쓰는 통합 테스트. 모델이 없으면 건너뛴다."""
import numpy as np
import pytest

from src.classify.cnn import TFLITE_PATHS
from src.config import MODELS, Config, preset
from src.detect.eye_mouth import PATCH
from src.temporal.alert import AlertLevel

needs_models = pytest.mark.skipif(
    not (TFLITE_PATHS["int8"].exists() and (MODELS / "eye_svm.joblib").exists()
         and (MODELS / "yawn_svm.joblib").exists()),
    reason="학습된 모델 없음")


@needs_models
def test_cnn_int8_matches_fp32():
    from src.classify.cnn import CnnClassifier
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (8, PATCH, PATCH), dtype=np.uint8)
    p8 = CnnClassifier(TFLITE_PATHS["int8"]).predict_closed(x)
    p32 = CnnClassifier(TFLITE_PATHS["fp32"]).predict_closed(x)
    assert p8.shape == (8,) and ((0 <= p8) & (p8 <= 1)).all()
    assert np.abs(p8 - p32).max() < 0.1
    # 배치 크기가 바뀌어도 동작
    assert CnnClassifier().predict_closed(x[:2]).shape == (2,)


@needs_models
def test_svm_probabilities():
    from src.classify.hog_svm import HogSvmClassifier
    x = np.random.default_rng(1).integers(0, 255, (4, PATCH, PATCH), dtype=np.uint8)
    p = HogSvmClassifier.load().predict_closed(x)
    assert p.shape == (4,) and ((0 <= p) & (p <= 1)).all()


@needs_models
@pytest.mark.parametrize("name", ["D1", "D4", "D5"])
def test_pipeline_reports_unknown_without_face(name):
    from src.pipeline import DrowsinessPipeline, draw
    pipe = DrowsinessPipeline(preset(name, Config.load()), fps=15)
    blank = np.full((240, 320, 3), 90, np.uint8)
    for i in range(30):
        r = pipe.process(blank, i / 15)
    assert r.face is None and r.closed is None
    assert r.judgement.level == AlertLevel.UNKNOWN
    assert {"preprocess", "face", "temporal", "total"} <= set(r.timings_ms)
    assert draw(blank, r, fps=15.0).shape == blank.shape
