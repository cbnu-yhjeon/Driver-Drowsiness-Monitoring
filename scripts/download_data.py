"""공개 데이터 일부를 내려받는다.

- 눈 상태: MichalMlodawski/closed-open-eyes (HuggingFace, ODC-BY)
  얼굴 512x512 + 양쪽 눈 bbox + open/closed 라벨. parquet 1개에 100장, 파일마다 라벨이 하나.
- 하품: c3rl/yawning-people (HuggingFace, MIT)
- Cascade: OpenCV 저장소의 Haar / LBP cascade xml

CEW는 공식 페이지가 자동 다운로드를 막고 있어 수동으로 받아 data/cew/ 에 풀어야 한다
(src/classify/datasets.py 의 load_cew_patches 참고).

    python -m scripts.download_data --eye-files 30 --yawn-images 400
"""
import argparse
import csv
import random
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
HF = "https://huggingface.co/datasets"
EYES_REPO = "MichalMlodawski/closed-open-eyes"
YAWN_REPO = "c3rl/yawning-people"
OPENCV_RAW = "https://raw.githubusercontent.com/opencv/opencv/4.x/data"
CASCADES = {
    "haarcascade_frontalface_default.xml": f"{OPENCV_RAW}/haarcascades/haarcascade_frontalface_default.xml",
    "haarcascade_eye.xml": f"{OPENCV_RAW}/haarcascades/haarcascade_eye.xml",
    "haarcascade_eye_tree_eyeglasses.xml": f"{OPENCV_RAW}/haarcascades/haarcascade_eye_tree_eyeglasses.xml",
    "lbpcascade_frontalface_improved.xml": f"{OPENCV_RAW}/lbpcascades/lbpcascade_frontalface_improved.xml",
}
# 1~632번 파일은 closed_eyes, 633번 이후는 open_eyes (라벨은 행 단위로 다시 확인한다)
CLOSED_RANGE = (1, 620)
OPEN_RANGE = (650, 1260)


def fetch(url: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dst)
    return dst


def spread(lo: int, hi: int, n: int) -> list[int]:
    """[lo, hi] 구간에서 고르게 n개 (생성 순서에 따른 편향을 줄이기 위해)."""
    step = (hi - lo) / max(n - 1, 1)
    return sorted({round(lo + i * step) for i in range(n)})


def download_eyes(n_per_class: int, workers: int) -> None:
    ids = spread(*CLOSED_RANGE, n_per_class) + spread(*OPEN_RANGE, n_per_class)
    out = DATA / "closed_open_eyes"
    jobs = [(f"{HF}/{EYES_REPO}/resolve/main/data/dataset_{i:03d}.parquet",
             out / f"dataset_{i:03d}.parquet") for i in ids]
    with ThreadPoolExecutor(workers) as ex:
        for p in ex.map(lambda j: fetch(*j), jobs):
            print("  ", p.name)


def download_yawn(n_per_class: int, workers: int, seed: int = 0) -> None:
    out = DATA / "yawning"
    meta = fetch(f"{HF}/{YAWN_REPO}/resolve/main/dataset.csv", out / "dataset.csv")
    with open(meta, newline="") as f:
        rows = list(csv.DictReader(f))
    rng = random.Random(seed)
    picked = []
    for label in ("1", "0"):
        cand = [r for r in rows if r["yawn_label"] == label]
        picked += rng.sample(cand, min(n_per_class, len(cand)))
    jobs = []
    for r in picked:
        sub = "yawning" if r["yawn_label"] == "1" else "notyawning"
        name = f"{r['image_id']}.png"
        jobs.append((f"{HF}/{YAWN_REPO}/resolve/main/images/{sub}/{name}", out / sub / name))
    with ThreadPoolExecutor(workers) as ex:
        done = sum(1 for _ in ex.map(lambda j: fetch(*j), jobs))
    print(f"   yawning images: {done}")


def download_cascades() -> None:
    for name, url in CASCADES.items():
        print("  ", fetch(url, ROOT / "models" / "cascades" / name).name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eye-files", type=int, default=30, help="클래스별 parquet 파일 수 (파일당 100장)")
    ap.add_argument("--yawn-images", type=int, default=400, help="클래스별 하품 이미지 수")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    print("[cascades]")
    download_cascades()
    print("[closed-open-eyes]")
    download_eyes(args.eye_files, args.workers)
    print("[yawning-people]")
    download_yawn(args.yawn_images, args.workers)


if __name__ == "__main__":
    main()
