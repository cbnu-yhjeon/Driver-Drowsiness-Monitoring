"""G4: CPU 처리 속도 (D1~D5). 640×480 합성 주행 영상에서 단계별 소요 시간을 잰다.

합성 영상: 시험 분할 얼굴 사진을 640×480 화면 안에서 천천히 움직이고(이동 · 크기 변화),
4초마다 다른 사람으로 바꿔 재검출 · 추적 경로를 모두 지나게 한다.
노트북 1코어 상황을 보려고 OpenCV · TFLite 스레드를 1개로 묶은 결과도 같이 잰다.

    python -m experiments.benchmark_fps [--frames 900] [--save-video data/synthetic_drive.mp4]
"""
import argparse
import platform
import time

import cv2
import numpy as np

from experiments.common import save, test_images
from src.classify.cnn import CnnClassifier
from src.config import PRESETS, Config, preset
from src.pipeline import DrowsinessPipeline

W, H, FPS = 640, 480, 15


def synthetic_video(n_frames: int, seed: int = 0) -> list[np.ndarray]:
    imgs = test_images()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(imgs))
    frames = []
    for i in range(n_frames):
        im = imgs[order[(i // (4 * FPS)) % len(order)]].gray
        t = i / FPS
        side = int(340 + 30 * np.sin(2 * np.pi * t / 7))
        cx = W / 2 + 80 * np.sin(2 * np.pi * t / 5)
        cy = H / 2 + 25 * np.sin(2 * np.pi * t / 3)
        face = cv2.resize(im, (side, side), interpolation=cv2.INTER_AREA)
        canvas = np.full((H, W), 60, np.uint8)
        x0, y0 = int(cx - side / 2), int(cy - side / 2)
        xs, ys = max(x0, 0), max(y0, 0)
        xe, ye = min(x0 + side, W), min(y0 + side, H)
        canvas[ys:ye, xs:xe] = face[ys - y0:ye - y0, xs - x0:xe - x0]
        noise = rng.normal(0, 2, canvas.shape)
        frames.append(cv2.cvtColor(np.clip(canvas + noise, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return frames


def bench(cfg: Config, frames, classifier=None) -> dict:
    pipe = DrowsinessPipeline(cfg, fps=FPS, classifier=classifier)
    for i, f in enumerate(frames[:15]):  # 예열
        pipe.process(f, i / FPS)
    pipe = DrowsinessPipeline(cfg, fps=FPS, classifier=pipe.classifier, yawn_classifier=pipe.yawn)
    stages: dict[str, list[float]] = {}
    sources = {"detect": 0, "track": 0, "lost": 0}
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        r = pipe.process(f, i / FPS)
        sources[r.face_source] += 1
        for k, v in r.timings_ms.items():
            stages.setdefault(k, []).append(v)
    wall = time.perf_counter() - t0
    return {"fps": len(frames) / wall, "ms_mean": {k: float(np.mean(v)) for k, v in stages.items()},
            "ms_p95_total": float(np.percentile(stages["total"], 95)), "face_source": sources}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--save-video", help="합성 영상을 mp4 로 저장 (앱 동작 확인용)")
    args = ap.parse_args()
    frames = synthetic_video(args.frames)
    if args.save_video:
        vw = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames:
            vw.write(f)
        vw.release()
        print("->", args.save_video)
    cpu = platform.processor() or platform.machine()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    out = {"cpu": cpu, "frame_size": [W, H], "frames": args.frames, "default_threads": {}, "single_thread": {}}
    for threads, key in ((None, "default_threads"), (1, "single_thread")):
        cv2.setNumThreads(threads if threads else -1)
        for d in PRESETS:
            cfg = preset(d)
            clf = CnnClassifier(threads=1) if cfg.classifier == "cnn" else None
            r = bench(cfg, frames, clf)
            out[key][d] = r
            ms = ", ".join(f"{k}={v:.2f}" for k, v in r["ms_mean"].items())
            print(f"[{key}] {d}: {r['fps']:.1f} FPS  ({ms})  {r['face_source']}")
    save("fps", out)


if __name__ == "__main__":
    main()
