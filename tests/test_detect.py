import cv2
import numpy as np
import pytest

from src.classify.augment import CTX, augment, center_crop
from src.config import Config, PRESETS, preset
from src.detect.eye_mouth import EyeLocator, FaceGeometry, PATCH, crop_square, mouth_box, mouth_opening_ratio
from src.detect.tracker import FaceTracker


def test_crop_square_center_and_scale():
    img = np.zeros((200, 200), np.uint8)
    cv2.rectangle(img, (90, 90), (110, 110), 255, -1)   # 한 변 21px
    p = crop_square(img, (100, 100), 42, 32)             # 흰 사각형이 가운데 절반을 차지
    assert p.shape == (32, 32)
    assert p[16, 16] == 255 and p[2, 2] == 0
    assert 0.2 < (p > 128).mean() < 0.3


def test_crop_square_large_downscale_keeps_offsets():
    img = np.zeros((600, 600), np.uint8)
    cv2.circle(img, (400, 200), 40, 255, -1)
    p = crop_square(img, (400, 200), 200, 32)            # k < 0.5 경로
    ys, xs = np.nonzero(p > 128)
    assert abs(xs.mean() - 15.5) < 1.0 and abs(ys.mean() - 15.5) < 1.0


def test_center_crop_matches_ctx_scale():
    ctx = np.random.default_rng(0).integers(0, 255, (5, CTX, CTX), dtype=np.uint8)
    assert center_crop(ctx).shape == (5, PATCH, PATCH)


def test_augment_shape_and_dtype():
    rng = np.random.default_rng(0)
    ctx = rng.integers(0, 255, (CTX, CTX), dtype=np.uint8)
    for _ in range(20):
        p = augment(ctx, rng)
        assert p.shape == (PATCH, PATCH) and p.dtype == np.uint8


def test_eye_locator_geometry():
    loc = EyeLocator(FaceGeometry((0.3, 0.4), (0.7, 0.4)))
    eyes = loc.locate(np.zeros((300, 300), np.uint8), (50, 50, 200, 200))
    (lx, ly), (rx, ry) = eyes.centers
    assert (lx, ly) == (110, 130) and (rx, ry) == (190, 130)
    assert eyes.iod == pytest.approx(80)
    assert loc.patches(np.zeros((300, 300), np.uint8), eyes).shape == (2, PATCH, PATCH)
    x, y, w, h = mouth_box(eyes, (300, 300))
    assert y > ly and x < 150 < x + w


def test_mouth_ratio_closed_vs_open():
    closed = np.full((80, 80), 170, np.uint8)
    cv2.line(closed, (20, 30), (60, 30), 40, 2)               # 다문 입: 가는 선
    yawn = np.full((80, 80), 170, np.uint8)
    cv2.ellipse(yawn, (40, 40), (14, 22), 0, 0, 360, 30, -1)  # 벌린 입: 세로 타원
    assert mouth_opening_ratio(closed) < 0.15
    assert mouth_opening_ratio(yawn) > 0.4


class _OnceDetector:
    """첫 호출에만 얼굴을 돌려주는 가짜 검출기 → 이후는 LK 추적만으로 따라가야 한다."""

    def __init__(self, box):
        self.box, self.calls = box, 0

    def detect_largest(self, gray):
        self.calls += 1
        return self.box if self.calls == 1 else None


def test_tracker_follows_translation_without_detection():
    rng = np.random.default_rng(1)
    tex = cv2.GaussianBlur(rng.integers(0, 255, (400, 400), dtype=np.uint8), (0, 0), 2)
    tr = FaceTracker(_OnceDetector((100, 100, 120, 120)), track=True, redetect_every=1000, max_track_frames=100)
    assert tr.update(tex).source == "detect"
    for i in range(1, 11):
        frame = np.roll(tex, (2 * i, 3 * i), axis=(0, 1))   # 아래 2px, 오른쪽 3px 씩
        r = tr.update(frame)
    assert r.source == "track"
    x, y, w, h = r.box
    assert abs(x - 130) <= 2 and abs(y - 120) <= 2 and abs(w - 120) <= 4


def test_tracker_gives_up_after_max_frames():
    tex = np.random.default_rng(2).integers(0, 255, (300, 300), dtype=np.uint8)
    tex = cv2.GaussianBlur(tex, (0, 0), 2)
    tr = FaceTracker(_OnceDetector((80, 80, 100, 100)), track=True, redetect_every=1000, max_track_frames=3)
    tr.update(tex)
    sources = [tr.update(tex).source for _ in range(5)]
    assert sources[:3] == ["track"] * 3 and sources[3] == "lost"


def test_presets_and_config():
    cfg = Config.load()
    assert cfg.temporal.closure_warning_s == 1.5
    assert preset("D1").classifier == "svm" and preset("D1").temporal.mode == "closure"
    assert preset("D5").detector == "lbp"
    assert len(PRESETS) == 5
    with pytest.raises(ValueError):
        Config.from_dict({"nope": 1})
