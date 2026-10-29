"""프레임 한 장 → 얼굴 · 눈 · 입 → 눈 상태 → 시간 창 판단. 단계별 소요 시간도 잰다."""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import Config
from src.detect.eye_mouth import EyeLocator, EyeRegions, FaceGeometry, mouth_box
from src.detect.face import Box, FaceDetector, Preprocessor
from src.detect.tracker import FaceTracker
from src.temporal.alert import AlertLevel, DrowsinessJudge, Judgement


def load_classifier(kind: str):
    if kind == "cnn":
        from src.classify.cnn import CnnClassifier
        return CnnClassifier()
    if kind == "svm":
        from src.classify.hog_svm import HogSvmClassifier
        return HogSvmClassifier.load()
    raise ValueError(f"classifier 는 cnn | svm: {kind}")


@dataclass
class FrameResult:
    t: float
    face: Box | None = None
    face_source: str = "lost"
    eyes: EyeRegions | None = None
    p_closed: float | None = None
    closed: bool | None = None
    mouth: Box | None = None
    yawn_score: float | None = None
    judgement: Judgement | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class DrowsinessPipeline:
    def __init__(self, cfg: Config, fps: float = 15.0, classifier=None, yawn_classifier=None,
                 refine_eyes: bool = False):
        self.cfg = cfg
        self.pre = Preprocessor(cfg.clahe, cfg.clahe_clip, cfg.clahe_tile)
        detector = FaceDetector(cfg.detector, cfg.detect_width, min_neighbors=cfg.min_neighbors)
        self.tracker = FaceTracker(detector, cfg.track, cfg.redetect_every,
                                   max_track_frames=max(1, round(cfg.max_track_s * fps)))
        self.locator = EyeLocator(FaceGeometry.load(cfg.detector), refine=refine_eyes)
        self.classifier = classifier if classifier is not None else load_classifier(cfg.classifier)
        self.yawn = None
        if cfg.temporal.mode == "full":
            from src.classify.yawn_svm import YawnClassifier
            self.yawn = yawn_classifier if yawn_classifier is not None else YawnClassifier.load()
        self.judge = DrowsinessJudge(cfg.temporal)

    def process(self, frame: np.ndarray, t: float) -> FrameResult:
        r = FrameResult(t)
        tm = r.timings_ms
        t0 = last = time.perf_counter()

        def lap(stage: str) -> None:
            nonlocal last
            now = time.perf_counter()
            tm[stage] = (now - last) * 1e3
            last = now

        gray = self.pre(frame)
        lap("preprocess")
        tr = self.tracker.update(gray)
        r.face, r.face_source = tr.box, tr.source
        lap("face")
        if r.face is not None:
            r.eyes = self.locator.locate(gray, r.face)
            patches = self.locator.patches(gray, r.eyes)
            lap("eyes")
            r.p_closed = float(np.mean(self.classifier.predict_closed(patches)))
            r.closed = r.p_closed >= self.cfg.closed_threshold
            lap("classify")
            if self.yawn is not None:
                r.mouth = mouth_box(r.eyes, gray.shape)
                x, y, w, h = r.mouth
                r.yawn_score = float(self.yawn.predict_yawn([gray[y:y + h, x:x + w]])[0])
                lap("yawn")
        r.judgement = self.judge.update(t, r.closed, r.yawn_score)
        lap("temporal")
        tm["total"] = (last - t0) * 1e3
        return r


# ---------------------------------------------------------------- 화면 표시
COLORS = {AlertLevel.NORMAL: (80, 200, 80), AlertLevel.CAUTION: (0, 200, 255),
          AlertLevel.WARNING: (40, 40, 240), AlertLevel.UNKNOWN: (160, 160, 160)}
LABELS = {AlertLevel.NORMAL: "NORMAL", AlertLevel.CAUTION: "CAUTION",
          AlertLevel.WARNING: "WARNING", AlertLevel.UNKNOWN: "NO FACE"}


def draw(frame: np.ndarray, r: FrameResult, fps: float | None = None) -> np.ndarray:
    """OpenCV 기본 글꼴은 한글을 못 그리므로 화면 문구는 영어로 둔다."""
    out = frame.copy()
    j = r.judgement
    color = COLORS[j.level]
    if r.face is not None:
        x, y, w, h = r.face
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(out, r.face_source, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if r.eyes is not None:
        eye_color = (0, 0, 255) if r.closed else (255, 200, 0)
        for ex, ey, ew, eh in r.eyes.boxes():
            cv2.rectangle(out, (ex, ey), (ex + ew, ey + eh), eye_color, 1)
    if r.mouth is not None:
        mx, my, mw, mh = r.mouth
        cv2.rectangle(out, (mx, my), (mx + mw, my + mh), (200, 120, 255), 1)
    H, W = out.shape[:2]
    cv2.rectangle(out, (0, 0), (W, 34), color, -1)
    cv2.putText(out, LABELS[j.level], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    c, yw = j.closure, j.yawn
    lines = [f"PERCLOS {c.perclos:5.1%}  closed {c.closure_s:4.1f}s  blinks/min {c.blinks_per_min:4.1f}",
             f"yawns {yw.count}  p(yawn) {yw.score:.2f}  p(closed) {'-' if r.p_closed is None else f'{r.p_closed:.2f}'}"]
    if fps:
        lines.append(f"{fps:4.1f} FPS  ({r.timings_ms.get('total', 0):.1f} ms/frame)")
    for i, s in enumerate(lines):
        cv2.putText(out, s, (10, H - 12 - 20 * (len(lines) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, s, (10, H - 12 - 20 * (len(lines) - 1 - i)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (30, 30, 30), 1, cv2.LINE_AA)
    return out
