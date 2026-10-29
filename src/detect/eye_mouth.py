"""얼굴 박스 → 눈 · 입 영역.

눈 위치는 얼굴 박스 안의 기하 비율(학습 데이터로 보정한 값)로 정하고, 원하면 Haar eye cascade 로 다듬는다.
Haar eye 는 감은 눈을 자주 놓치므로 기본값은 기하 비율만 쓴다(experiments/eval_pipeline 비교 결과).
눈 패치 크기는 양눈 사이 거리(IOD)에 비례 → 학습 패치와 실행 시 패치의 배율이 같다.
"""
import json
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from src.config import MODELS
from src.detect.face import Box, load_cascade

PATCH = 32          # 분류기 입력 크기
PATCH_IOD = 0.75    # 눈 패치 한 변 = 0.75 × IOD
GEOMETRY_FILE = MODELS / "face_geometry.json"


@dataclass(frozen=True)
class FaceGeometry:
    """얼굴 박스 기준 상대 좌표. left/right 는 영상 기준(왼쪽 = x 가 작은 쪽)."""
    left_eye: tuple[float, float] = (0.31, 0.40)
    right_eye: tuple[float, float] = (0.69, 0.40)

    @classmethod
    def load(cls, detector: str) -> "FaceGeometry":
        if GEOMETRY_FILE.exists():
            data = json.loads(GEOMETRY_FILE.read_text())
            if detector in data:
                g = data[detector]
                return cls(tuple(g["left_eye"]), tuple(g["right_eye"]))
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)


def crop_square(gray: np.ndarray, center: tuple[float, float], side: float, out: int = PATCH,
                angle: float = 0.0) -> np.ndarray:
    """center 를 중심으로 한 변 side 인 정사각형을 out×out 으로 잘라낸다(서브픽셀, 경계 복제)."""
    k = out / side
    m = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, k)
    m[0, 2] += out / 2 - center[0]
    m[1, 2] += out / 2 - center[1]
    if k < 0.5:  # 크게 줄일 때 계단 현상 방지 (warpAffine 은 INTER_AREA 를 지원하지 않음)
        r = int(np.ceil(side * 0.75))
        cx, cy = int(center[0]), int(center[1])
        y0, x0 = max(cy - r, 0), max(cx - r, 0)
        sub = gray[y0:cy + r, x0:cx + r]
        if sub.size:  # 잘라낸 부분만 흐리고, 원점 이동만큼 변환 행렬을 보정
            gray = cv2.GaussianBlur(sub, (0, 0), 0.4 / k)
            m[:, 2] += m[:, :2] @ np.array([x0, y0], dtype=np.float64)
    return cv2.warpAffine(gray, m, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


@dataclass
class EyeRegions:
    centers: list[tuple[float, float]]  # [영상 왼쪽 눈, 오른쪽 눈]
    side: float                         # 패치 한 변(px)
    iod: float
    refined: list[bool]

    def boxes(self) -> list[Box]:
        s = self.side
        return [(int(cx - s / 2), int(cy - s / 2), int(s), int(s)) for cx, cy in self.centers]


class EyeLocator:
    def __init__(self, geometry: FaceGeometry, refine: bool = False, eyeglasses: bool = False):
        self.geometry = geometry
        self.cascade = None
        if refine:
            self.cascade = load_cascade("haarcascade_eye_tree_eyeglasses.xml" if eyeglasses else "haarcascade_eye.xml")

    def locate(self, gray: np.ndarray, face: Box) -> EyeRegions:
        x, y, w, h = face
        g = self.geometry
        centers = [(x + g.left_eye[0] * w, y + g.left_eye[1] * h),
                   (x + g.right_eye[0] * w, y + g.right_eye[1] * h)]
        refined = [False, False]
        iod_geo = float(np.hypot(centers[1][0] - centers[0][0], centers[1][1] - centers[0][1]))
        if self.cascade is not None:
            centers, refined = self._refine(gray, face, centers)
        iod = float(np.hypot(centers[1][0] - centers[0][0], centers[1][1] - centers[0][1]))
        iod = float(np.clip(iod, 0.7 * iod_geo, 1.3 * iod_geo))  # 다듬기가 엉뚱하게 튀는 것 방지
        return EyeRegions(centers, PATCH_IOD * iod, iod, refined)

    def _refine(self, gray, face, centers):
        x, y, w, h = face
        y0, y1 = int(y + 0.15 * h), int(y + 0.6 * h)
        roi = gray[max(y0, 0):max(y1, 0), max(x, 0):max(x + w, 0)]
        if roi.size == 0:
            return centers, [False, False]
        found = self.cascade.detectMultiScale(roi, 1.1, 4, minSize=(int(w * 0.12), int(w * 0.12)),
                                              maxSize=(int(w * 0.4), int(w * 0.4)))
        out, refined = list(centers), [False, False]
        tol = 0.12 * w
        for ex, ey, ew, eh in found:
            c = (max(x, 0) + ex + ew / 2, max(y0, 0) + ey + eh / 2)
            side = 0 if c[0] < x + w / 2 else 1
            if np.hypot(c[0] - centers[side][0], c[1] - centers[side][1]) < tol and not refined[side]:
                out[side], refined[side] = c, True
        return out, refined

    @staticmethod
    def patches(gray: np.ndarray, eyes: EyeRegions) -> np.ndarray:
        return np.stack([crop_square(gray, c, eyes.side) for c in eyes.centers])


# ---------------------------------------------------------------- 입 (하품)
MOUTH_SIZE = 64


def mouth_box(eyes: EyeRegions, frame_shape) -> Box:
    """양 눈 중점 아래 IOD 0.55~2.0 구간, 폭 1.4 IOD (크게 벌린 입이 아래로 내려가는 것까지 포함)."""
    (lx, ly), (rx, ry) = eyes.centers
    mx, ey, iod = (lx + rx) / 2, (ly + ry) / 2, eyes.iod
    x0, x1 = mx - 0.7 * iod, mx + 0.7 * iod
    y0, y1 = ey + 0.55 * iod, ey + 2.0 * iod
    H, W = frame_shape[:2]
    x0, y0 = int(np.clip(x0, 0, W - 1)), int(np.clip(y0, 0, H - 1))
    x1, y1 = int(np.clip(x1, x0 + 1, W)), int(np.clip(y1, y0 + 1, H))
    return x0, y0, x1 - x0, y1 - y0


def mouth_opening_ratio(roi_gray: np.ndarray) -> float:
    """입 안쪽 어두운 영역의 세로 길이 / ROI 폭 (README 초안의 규칙 기반 기준선).

    다문 입은 입술 사이 어두운 선이 가로로 얇게, 하품은 어두운 타원이 세로로 크게 나타난다.
    실제 얼굴에서는 수염·그림자·치아 때문에 잘 맞지 않아 실행 경로에서는 YawnClassifier 를 쓴다.
    """
    if roi_gray.size == 0:
        return 0.0
    roi = cv2.resize(roi_gray, (MOUTH_SIZE, MOUTH_SIZE), interpolation=cv2.INTER_AREA)
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    lo, med = np.percentile(roi, 2), np.median(roi)
    if med - lo < 10:  # 대비가 거의 없음
        return 0.0
    dark = (roi < lo + 0.45 * (med - lo)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(dark, connectivity=8)
    best = 0.0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx = cents[i][0]
        if area < 12 or abs(cx - MOUTH_SIZE / 2) > MOUTH_SIZE * 0.25:
            continue  # 가운데에 있는 덩어리만
        if x == 0 or x + w >= MOUTH_SIZE:
            continue  # 좌우 경계에 닿으면 그림자/배경
        best = max(best, h / MOUTH_SIZE)
    return float(best)
