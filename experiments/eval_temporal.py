"""G2: 시간 창 판단의 졸음 이벤트 정밀도 · 재현율 · 시간당 오경보 (합성 운전 세션).

실제 졸음 영상(NTHU-DDD)은 사용 신청이 필요해 아직 없다. 대신
  1) 운전자 행동을 합성한다: 평상시 깜빡임(0.1~0.4초)과 가끔의 긴 감음(재채기 등 0.5~1.2초),
     졸음 구간의 느린 깜빡임 · 미세수면(1.5~4초) · 하품
  2) 프레임 판정 오류를 eval_pipeline / eval_yawn 에서 잰 비율(FPR, FNR, 얼굴 놓침)로 덧씌운다
     (오류는 독립이 아니라 몇 프레임씩 몰려 나오게 Markov 연쇄로 만든다)
  3) 실제 DrowsinessJudge 를 그대로 돌려 경보 시점을 정답 졸음 구간과 비교한다.

임계값은 검증용 세션(seed 0~)에서 고르고, 다른 세션(seed 10000~)에서 보고한다.

    python -m experiments.eval_temporal
"""
import itertools
import json
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

from experiments.common import RESULTS, save
from src.config import Config, preset
from src.temporal.alert import AlertLevel, DrowsinessJudge

FPS = 15
SESSION_S = 600
GRACE_S = 5.0       # 졸음 구간이 끝난 뒤 이 시간 안의 경보도 맞힌 것으로 본다
N_SESSIONS = 30     # 졸음 포함 세션 수 (같은 수의 평상시 세션도 만든다)


@dataclass
class Session:
    closed: np.ndarray       # 실제 눈 감김 (프레임)
    yawning: np.ndarray      # 실제 하품
    episodes: list           # 졸음 구간 [(시작 s, 끝 s)]


def _intervals(rng, dur_s, rate_per_min, sampler):
    """구간 [0, dur_s) 안에 Poisson 으로 사건을 놓고 각 길이를 sampler 로 뽑는다."""
    out, t = [], 0.0
    while True:
        t += rng.exponential(60 / rate_per_min)
        if t >= dur_s:
            return out
        d = sampler()
        out.append((t, t + d))
        t += d


def simulate_driver(rng, drowsy: bool) -> Session:
    n = SESSION_S * FPS
    closed = np.zeros(n, bool)
    yawning = np.zeros(n, bool)
    episodes = []
    # 졸음 세션: 평상시 → 졸음 구간(40~120초)이 2~3번 끼어든다
    segs, t = [], 0.0
    while t < SESSION_S:
        a = rng.uniform(60, 200) if drowsy else SESSION_S
        segs.append(("alert", t, min(t + a, SESSION_S)))
        t += a
        if drowsy and t < SESSION_S - 30:
            d = rng.uniform(40, 120)
            segs.append(("drowsy", t, min(t + d, SESSION_S)))
            episodes.append((t, min(t + d, SESSION_S)))
            t += d

    def put(arr, s0, ivs):
        for a, b in ivs:
            arr[int((s0 + a) * FPS):int((s0 + b) * FPS)] = True

    for kind, s0, s1 in segs:
        dur = s1 - s0
        if kind == "alert":
            put(closed, s0, _intervals(rng, dur, 17, lambda: np.clip(rng.lognormal(np.log(0.2), 0.35), 0.08, 0.45)))
            # 졸음이 아닌 긴 감음(재채기 · 눈 비비기): 5분에 1번꼴, 0.5~1.2초 (가정)
            put(closed, s0, _intervals(rng, dur, 0.2, lambda: rng.uniform(0.5, 1.2)))
            put(yawning, s0, _intervals(rng, dur, 0.05, lambda: rng.uniform(3, 6)))
        else:
            put(closed, s0, _intervals(rng, dur, 20, lambda: np.clip(rng.lognormal(np.log(0.45), 0.4), 0.15, 1.2)))
            put(closed, s0, _intervals(rng, dur, 1.5, lambda: rng.uniform(1.5, 4.0)))   # 미세수면
            put(yawning, s0, _intervals(rng, dur, 1.2, lambda: rng.uniform(3, 6)))
    return Session(closed, yawning, episodes)


def bursty_errors(rng, n: int, rate: float, mean_len: float = 3.0) -> np.ndarray:
    """주변 비율 rate, 평균 길이 mean_len 프레임으로 몰려 나오는 오류 마스크."""
    if rate <= 0:
        return np.zeros(n, bool)
    rate = min(rate, 0.95)
    p_exit = 1 / mean_len
    p_enter = rate * p_exit / (1 - rate)
    out = np.zeros(n, bool)
    state = rng.random() < rate
    u = rng.random(n)
    for i in range(n):
        state = (u[i] >= p_exit) if state else (u[i] < p_enter)
        out[i] = state
    return out


def observe(rng, s: Session, rates: dict, track: bool, max_track_s: float = 2.0):
    """정답 → 프레임 판정 (None = 얼굴 놓침)."""
    n = len(s.closed)
    flip_open = bursty_errors(rng, n, rates["fpr"])    # 뜬 눈을 감음으로
    flip_closed = bursty_errors(rng, n, rates["fnr"])  # 감은 눈을 뜸으로
    obs = np.where(s.closed, ~flip_closed, flip_open)
    miss = bursty_errors(rng, n, rates["miss"], mean_len=FPS * 1.0)
    if track:  # 놓친 구간 중 max_track_s 이내는 LK 추적이 메운다
        run = 0
        bridged = np.zeros(n, bool)
        for i in range(n):
            run = run + 1 if miss[i] else 0
            bridged[i] = miss[i] and run <= max_track_s * FPS
        miss &= ~bridged
    y_hit = bursty_errors(rng, n, 1 - rates["yawn_tpr"], mean_len=FPS * 0.5)
    y_fa = bursty_errors(rng, n, rates["yawn_fpr"], mean_len=FPS * 0.5)
    yawn = np.where(s.yawning, ~y_hit, y_fa).astype(float)
    closed = [None if m else bool(o) for o, m in zip(obs, miss)]
    yscore = [None if m else y for y, m in zip(yawn, miss)]
    return closed, yscore


def run_judge(cfg: Config, closed, yscore) -> np.ndarray:
    judge = DrowsinessJudge(cfg.temporal)
    return np.array([judge.update(i / FPS, c, y).level.value for i, (c, y) in enumerate(zip(closed, yscore))])


def score(levels: np.ndarray, episodes, levels_set=(AlertLevel.WARNING.value,)) -> dict:
    on = np.isin(levels, levels_set)
    onsets = np.flatnonzero(on & ~np.r_[False, on[:-1]]) / FPS
    inside = lambda t: any(a <= t <= b + GRACE_S for a, b in episodes)
    tp_onsets = sum(inside(t) for t in onsets)
    detected, latency = 0, []
    for a, b in episodes:
        hits = [t for t in onsets if a <= t <= b + GRACE_S]
        if not hits and on[int(a * FPS):int(min(b + GRACE_S, SESSION_S) * FPS)].any():
            hits = [a]  # 구간 시작 전부터 이어진 경보
        if hits:
            detected += 1
            latency.append(hits[0] - a)
    alert_s = SESSION_S - sum(min(b + GRACE_S, SESSION_S) - a for a, b in episodes)
    return {"episodes": len(episodes), "detected": detected, "onsets": len(onsets), "tp_onsets": tp_onsets,
            "fa": len(onsets) - tp_onsets, "alert_h": alert_s / 3600, "latency": latency}


def aggregate(scores: list[dict]) -> dict:
    ep = sum(s["episodes"] for s in scores)
    det = sum(s["detected"] for s in scores)
    on = sum(s["onsets"] for s in scores)
    tp = sum(s["tp_onsets"] for s in scores)
    fa = sum(s["fa"] for s in scores)
    hours = sum(s["alert_h"] for s in scores)
    lat = [x for s in scores for x in s["latency"]]
    rec, prec = det / ep if ep else 0.0, tp / on if on else 1.0
    return {"recall": rec, "precision": prec, "f1": 2 * rec * prec / (rec + prec) if rec + prec else 0.0,
            "false_alarms_per_h": fa / hours if hours else 0.0, "median_latency_s": float(np.median(lat)) if lat else None,
            "episodes": ep, "alarms": on, "alert_hours": hours}


def evaluate(args) -> dict:
    cfg, rates, seed0 = args
    warn, any_alert = [], []
    for k in range(N_SESSIONS * 2):
        rng = np.random.default_rng(seed0 + k)
        s = simulate_driver(rng, drowsy=k < N_SESSIONS)
        closed, ys = observe(rng, s, rates, cfg.track, cfg.max_track_s)
        levels = run_judge(cfg, closed, ys)
        warn.append(score(levels, s.episodes))
        any_alert.append(score(levels, s.episodes, (AlertLevel.WARNING.value, AlertLevel.CAUTION.value)))
    return {"warning": aggregate(warn), "caution_or_warning": aggregate(any_alert)}


def frame_rates() -> dict:
    """D 구성 · 조건별 프레임 판정 오류율 (eval_pipeline, eval_yawn 결과)."""
    pipe = json.loads((RESULTS / "pipeline.json").read_text())["D"]
    yawn = json.loads((RESULTS / "yawn.json").read_text())["hog_svm"]["test"]
    return {d: {cond: {"fpr": v["fpr"], "fnr": v["fnr"], "miss": 1 - v["face_rate"],
                       "yawn_tpr": yawn["recall"], "yawn_fpr": yawn["fpr"]}
                for cond, v in per.items()} for d, per in pipe.items()}


def main() -> None:
    rates = frame_rates()
    base = Config.load()
    # 1) 임계값 고르기 (검증 세션, D4 · 일반 조건): 재현율 ≥ 0.95 중 오경보가 가장 적은 조합
    grid = list(itertools.product((1.0, 1.5, 2.0, 2.5), (0.2, 0.25, 0.3, 0.35)))
    jobs = [(preset("D4", base).with_temporal(closure_warning_s=c, perclos_warning=p), rates["D4"]["normal"], 0)
            for c, p in grid]
    with Pool() as pool:
        sweep = pool.map(evaluate, jobs)
    rows = [{"closure_warning_s": c, "perclos_warning": p, **r["warning"]} for (c, p), r in zip(grid, sweep)]
    ok = [r for r in rows if r["recall"] >= 0.95] or rows
    best = min(ok, key=lambda r: (r["false_alarms_per_h"], -r["recall"]))
    print("threshold sweep (val):")
    for r in rows:
        print(f"  closure≥{r['closure_warning_s']}s PERCLOS≥{r['perclos_warning']:.2f}: recall={r['recall']:.3f} "
              f"prec={r['precision']:.3f} FA/h={r['false_alarms_per_h']:.2f}")
    print("chosen:", {k: best[k] for k in ("closure_warning_s", "perclos_warning")})
    tuned = base.with_temporal(closure_warning_s=best["closure_warning_s"], perclos_warning=best["perclos_warning"])

    # 2) D1~D5 × 조건 (시험 세션, 고른 임계값)
    keys = [(d, cond) for d in ("D1", "D2", "D3", "D4", "D5") for cond in ("normal", "lowlight", "glare")]
    jobs = [(preset(d, tuned), rates[d][cond], 10_000) for d, cond in keys]
    with Pool() as pool:
        res = pool.map(evaluate, jobs)
    table: dict[str, dict] = {}
    for (d, cond), r in zip(keys, res):
        table.setdefault(d, {})[cond] = r
        w = r["warning"]
        print(f"{d} {cond:8s} WARNING recall={w['recall']:.3f} prec={w['precision']:.3f} "
              f"FA/h={w['false_alarms_per_h']:.2f} latency={w['median_latency_s']}")
    save("temporal", {"setup": {"fps": FPS, "session_s": SESSION_S, "sessions_per_job": 2 * N_SESSIONS,
                                "grace_s": GRACE_S},
                      "sweep_val": rows, "chosen": {k: best[k] for k in ("closure_warning_s", "perclos_warning")},
                      "frame_rates": rates, "test": table})


if __name__ == "__main__":
    main()
