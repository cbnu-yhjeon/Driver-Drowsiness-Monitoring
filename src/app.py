"""웹캠 / 영상 파일 실시간 졸음 감지.

    python -m src.app                          # 웹캠 0번, 최종안(configs/default.yaml)
    python -m src.app --source drive.mp4       # 영상 파일 (시간은 영상 FPS 기준)
    python -m src.app --preset D1 --no-display --log results/run.csv

개인정보: 기본값으로는 영상을 저장하지 않는다. --log 는 판정 결과(숫자)만 기록한다.
--output 은 시연 영상을 만들 때만 명시적으로 쓴다.
"""
import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from src.config import Config, preset
from src.pipeline import DrowsinessPipeline, draw
from src.temporal.alert import AlertLevel


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="웹캠 번호 또는 영상 경로")
    ap.add_argument("--config", type=Path, help="설정 YAML (기본 configs/default.yaml)")
    ap.add_argument("--preset", choices=["D1", "D2", "D3", "D4", "D5"], help="실험 구성")
    ap.add_argument("--no-display", action="store_true", help="창 없이 실행")
    ap.add_argument("--log", type=Path, help="프레임별 판정 CSV (영상은 저장하지 않음)")
    ap.add_argument("--output", type=Path, help="주석을 그린 영상 저장 (시연용, 얼굴이 담김)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--width", type=int, default=640, help="처리 폭(px). 0 이면 원본")
    return ap.parse_args(argv)


def open_source(src: str) -> tuple[cv2.VideoCapture, bool]:
    live = src.isdigit()
    cap = cv2.VideoCapture(int(src) if live else src)
    if not cap.isOpened():
        sys.exit(f"입력을 열 수 없음: {src}" + (" (WSL 에는 웹캠이 없으면 영상 파일을 쓰세요)" if live else ""))
    return cap, live


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = Config.load(args.config)
    if args.preset:
        cfg = preset(args.preset, cfg)
    cap, live = open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fps_hint = src_fps if 1 < src_fps < 121 else 15.0
    pipe = DrowsinessPipeline(cfg, fps=fps_hint)

    display = not args.no_display
    writer, log_file, log = None, None, None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(args.log, "w", newline="")
        log = csv.writer(log_file)
        log.writerow(["frame", "t", "level", "face_source", "p_closed", "closed", "closure_s", "perclos",
                      "valid_s", "yawn_score", "yawns", "ms_total", "reasons"])
    recent = deque(maxlen=30)
    prev_level, idx, t_start = AlertLevel.NORMAL, 0, time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.width and frame.shape[1] > args.width:
                s = args.width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            t = time.monotonic() - t_start if live else idx / fps_hint
            tic = time.perf_counter()
            r = pipe.process(frame, t)
            recent.append(time.perf_counter() - tic)
            fps = len(recent) / sum(recent)
            j = r.judgement
            if j.level == AlertLevel.WARNING and prev_level != AlertLevel.WARNING:
                print(f"\a[{t:7.1f}s] WARNING: {', '.join(j.reasons)}", flush=True)
            elif j.level != prev_level:
                print(f"[{t:7.1f}s] {j.level.value} {', '.join(j.reasons)}", flush=True)
            prev_level = j.level
            if log:
                log.writerow([idx, f"{t:.3f}", j.level.value, r.face_source,
                              "" if r.p_closed is None else f"{r.p_closed:.3f}",
                              "" if r.closed is None else int(r.closed), f"{j.closure.closure_s:.2f}",
                              f"{j.closure.perclos:.3f}", f"{j.closure.valid_s:.1f}",
                              "" if r.yawn_score is None else f"{r.yawn_score:.3f}", j.yawn.count,
                              f"{r.timings_ms['total']:.2f}", "; ".join(j.reasons)])
            if display or args.output:
                vis = draw(frame, r, fps)
                if args.output:
                    if writer is None:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 fps_hint, (vis.shape[1], vis.shape[0]))
                    writer.write(vis)
                if display:
                    try:
                        cv2.imshow("Driver Drowsiness Monitoring (q: quit)", vis)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                    except cv2.error:
                        print("화면 출력을 쓸 수 없어 --no-display 로 계속합니다 (opencv-python-headless?)")
                        display = False
            idx += 1
            if args.max_frames and idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer:
            writer.release()
        if log_file:
            log_file.close()
        if display:
            cv2.destroyAllWindows()
    if recent:
        print(f"frames {idx}, 최근 처리 속도 {len(recent) / sum(recent):.1f} FPS")


if __name__ == "__main__":
    main()
