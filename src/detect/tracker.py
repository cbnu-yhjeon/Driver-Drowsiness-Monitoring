"""검출 사이 얼굴 위치를 Lucas-Kanade 광류로 유지한다.

N 프레임마다 재검출하고, 그 사이에는 얼굴 안 특징점의 중앙값 이동량·스케일 변화로 박스를 옮긴다.
검출이 계속 실패하면 max_track_frames 까지만 추적으로 버티고 그 뒤에는 '잃음'으로 본다.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from src.detect.face import Box, FaceDetector

LK_PARAMS = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


@dataclass
class TrackResult:
    box: Box | None
    source: str  # detect | track | lost


class FaceTracker:
    def __init__(self, detector: FaceDetector, track: bool = True, redetect_every: int = 5,
                 max_track_frames: int = 30, min_points: int = 8, fb_max_err: float = 1.0):
        self.detector = detector
        self.track = track
        self.redetect_every = max(1, redetect_every)
        self.max_track_frames = max_track_frames
        self.min_points = min_points
        self.fb_max_err = fb_max_err
        self.reset()

    def reset(self) -> None:
        self.box: np.ndarray | None = None  # float x, y, w, h
        self.prev_gray: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.frame_idx = 0
        self.frames_since_detect = 0

    def update(self, gray: np.ndarray) -> TrackResult:
        """gray: 검출에 쓰는 전처리된 회색 영상."""
        due = self.box is None or not self.track or self.frame_idx % self.redetect_every == 0
        self.frame_idx += 1
        if due:
            found = self.detector.detect_largest(gray)
            if found is not None:
                self._init(gray, found)
                return TrackResult(self._int_box(), "detect")
            if not self.track:
                self.box = None
                return TrackResult(None, "lost")
        if self.box is None or not self._step(gray):
            self.reset()
            return TrackResult(None, "lost")
        return TrackResult(self._int_box(), "track")

    def _init(self, gray: np.ndarray, box: Box) -> None:
        self.box = np.array(box, dtype=np.float32)
        self.prev_gray = gray
        self.points = self._features(gray, self.box)
        self.frames_since_detect = 0

    def _features(self, gray: np.ndarray, box: np.ndarray) -> np.ndarray | None:
        x, y, w, h = box
        mask = np.zeros_like(gray)
        # 얼굴 가장자리(배경·머리카락)를 피해 안쪽만
        x0, y0 = int(x + 0.15 * w), int(y + 0.1 * h)
        x1, y1 = int(x + 0.85 * w), int(y + 0.95 * h)
        mask[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 255
        return cv2.goodFeaturesToTrack(gray, maxCorners=60, qualityLevel=0.01,
                                       minDistance=max(3, int(w / 20)), mask=mask)

    def _step(self, gray: np.ndarray) -> bool:
        self.frames_since_detect += 1
        if self.frames_since_detect > self.max_track_frames:
            return False
        if self.points is None or len(self.points) < self.min_points:
            return False
        p0 = self.points
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **LK_PARAMS)
        if p1 is None:
            return False
        p0r, st_b, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev_gray, p1, None, **LK_PARAMS)
        fb = np.linalg.norm((p0 - p0r).reshape(-1, 2), axis=1)
        good = (st.ravel() == 1) & (st_b.ravel() == 1) & (fb < self.fb_max_err)
        if good.sum() < self.min_points:
            return False
        a, b = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
        shift = np.median(b - a, axis=0)
        scale = self._scale_change(a, b)
        x, y, w, h = self.box
        cx, cy = x + w / 2 + shift[0], y + h / 2 + shift[1]
        w, h = w * scale, h * scale
        self.box = np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float32)
        self.points = b.reshape(-1, 1, 2)
        self.prev_gray = gray
        return True

    @staticmethod
    def _scale_change(a: np.ndarray, b: np.ndarray) -> float:
        n = len(a)
        if n < 2:
            return 1.0
        i, j = np.triu_indices(n, k=1)
        da = np.linalg.norm(a[i] - a[j], axis=1)
        db = np.linalg.norm(b[i] - b[j], axis=1)
        ok = da > 1e-3
        if not ok.any():
            return 1.0
        return float(np.clip(np.median(db[ok] / da[ok]), 0.8, 1.25))

    def _int_box(self) -> Box:
        return tuple(int(round(v)) for v in self.box)
