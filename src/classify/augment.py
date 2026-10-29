"""눈 패치 증강과 조건 합성(저조도 · 안경 반사).

학습 패치는 '문맥 패치'(눈 패치 한 변 S 의 1.5배 영역을 48px 로 저장)에서 잘라낸다.
가운데 32px 이 실행 시 패치와 같은 배율이다.
"""
import cv2
import numpy as np

from src.detect.eye_mouth import PATCH, crop_square

CTX = 48                      # 문맥 패치 크기
CTX_SCALE = CTX / PATCH       # 1.5


def center_crop(ctx: np.ndarray) -> np.ndarray:
    """(N, 48, 48) → (N, 32, 32)."""
    o = (CTX - PATCH) // 2
    return ctx[..., o:o + PATCH, o:o + PATCH]


def simulate_low_light(gray: np.ndarray, rng: np.random.Generator,
                       gain: float = 0.25, gamma: float = 1.4, noise: float = 5.0) -> np.ndarray:
    """어두운 차 안: 밝기를 낮추고(감마·이득) 센서 잡음을 더한 뒤 8비트로 양자화."""
    x = (gray.astype(np.float32) / 255.0) ** gamma * gain * 255.0
    x += rng.normal(0, noise, gray.shape)
    return np.clip(x, 0, 255).astype(np.uint8)


def add_glare(patch: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """안경 렌즈 반사를 흉내 낸 밝은 타원(합성). 패치 크기에 상대적으로 그린다."""
    h, w = patch.shape[:2]
    mask = np.zeros((h, w), np.float32)
    c = (int(w * rng.uniform(0.3, 0.7)), int(h * rng.uniform(0.3, 0.6)))
    axes = (int(w * rng.uniform(0.12, 0.3)), int(h * rng.uniform(0.06, 0.15)))
    cv2.ellipse(mask, c, axes, rng.uniform(-40, 40), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(w, h) / 16)
    out = patch.astype(np.float32) + mask * strength * rng.uniform(120, 220)
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(ctx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """문맥 패치 1장 → 증강된 32×32 패치 1장."""
    shift = rng.normal(0, 0.10 * PATCH, 2)       # 눈 위치 추정 오차 (기하 비율 오차 중앙값 ≈ 0.1 패치)
    scale = rng.uniform(0.85, 1.2)               # 얼굴 박스 크기 오차
    angle = rng.uniform(-12, 12)                 # 고개 기울임
    p = crop_square(ctx, (CTX / 2 + shift[0], CTX / 2 + shift[1]), PATCH * scale, PATCH, angle)
    if rng.random() < 0.5:
        p = p[:, ::-1]
    p = p.astype(np.float32)
    if rng.random() < 0.35:                      # 저조도
        p = (p / 255.0) ** rng.uniform(1.0, 1.8) * rng.uniform(0.15, 0.6) * 255.0
        p += rng.normal(0, rng.uniform(1, 7), p.shape)
    else:                                        # 일반 밝기·대비 변화
        p = (p - 128) * rng.uniform(0.7, 1.3) + 128 + rng.uniform(-30, 30)
    p = np.clip(p, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:
        p = cv2.GaussianBlur(p, (0, 0), rng.uniform(0.4, 1.2))
    if rng.random() < 0.12:
        p = add_glare(p, rng, strength=rng.uniform(0.4, 1.0))
    return np.ascontiguousarray(p)
