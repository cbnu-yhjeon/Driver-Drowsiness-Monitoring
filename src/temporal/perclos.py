"""시간축 신호: PERCLOS, 연속 감음 시간, 하품 횟수.

모든 update 는 (t초, 프레임 판정) 을 받는다. 판정 None 은 '알 수 없음'(얼굴/눈 못 찾음)으로,
PERCLOS 분모에서 빠지고 감음 구간을 끝내지도 늘리지도 않는다.
"""
from collections import deque
from dataclasses import dataclass


@dataclass
class ClosureStats:
    closure_s: float        # 지금 이어지고 있는 감음 시간 (뜬 상태면 0)
    perclos: float          # 창 안 유효 프레임 중 감은 프레임 비율
    valid_s: float          # 창 안 유효 프레임이 차지하는 시간
    blinks_per_min: float   # 창 안에서 끝난 감음 구간 수 (분당)
    mean_blink_s: float     # 그 평균 길이


class EyeClosureMonitor:
    def __init__(self, window_s: float = 60.0, gap_s: float = 0.1):
        self.window_s = window_s
        self.gap_s = gap_s
        self.samples: deque[tuple[float, bool]] = deque()
        self.n_closed = 0
        self.events: deque[tuple[float, float]] = deque()  # (끝난 시각, 길이)
        self.closed_since: float | None = None
        self.last_closed_t = 0.0
        self.last_t: float | None = None
        self.dt = 1 / 15                                     # 프레임 간격 추정 (EMA)

    def update(self, t: float, closed: bool | None) -> ClosureStats:
        if self.last_t is not None and t > self.last_t:
            self.dt = 0.9 * self.dt + 0.1 * min(t - self.last_t, 1.0)
        self.last_t = t
        if closed is not None:
            self.samples.append((t, closed))
            self.n_closed += closed
        while self.samples and self.samples[0][0] <= t - self.window_s:
            self.n_closed -= self.samples.popleft()[1]
        while self.events and self.events[0][0] <= t - self.window_s:
            self.events.popleft()

        if closed:
            if self.closed_since is None:
                self.closed_since = t
            self.last_closed_t = t
        elif self.closed_since is not None and t - self.last_closed_t > self.gap_s:
            # 뜬 상태(또는 알 수 없음)가 gap 보다 길게 이어짐 → 감음 구간 종료
            self.events.append((self.last_closed_t, self.last_closed_t - self.closed_since + self.dt))
            self.closed_since = None
        # gap 안의 짧은 '뜸' 판정 동안에도 구간은 유지되고, 길이는 마지막 감음 프레임까지로 센다
        closure = 0.0 if self.closed_since is None else self.last_closed_t - self.closed_since + self.dt

        n = len(self.samples)
        span = min(self.window_s, t - self.samples[0][0] + self.dt) if n else 0.0
        mins = max(span, 1e-6) / 60
        return ClosureStats(
            closure_s=closure,
            perclos=self.n_closed / n if n else 0.0,
            valid_s=min(n * self.dt, span),
            blinks_per_min=len(self.events) / mins if span >= 10 else 0.0,
            mean_blink_s=sum(d for _, d in self.events) / len(self.events) if self.events else 0.0,
        )


@dataclass
class YawnStats:
    yawning: bool
    count: int      # 창 안 하품 횟수
    score: float    # 이번 프레임 하품 확률


class YawnMonitor:
    """하품 점수(확률)가 임계값 이상으로 min_s 이상 이어지면 하품 1회."""

    def __init__(self, threshold: float = 0.5, min_s: float = 1.0, window_s: float = 300.0, gap_s: float = 0.3):
        self.threshold, self.min_s, self.window_s, self.gap_s = threshold, min_s, window_s, gap_s
        self.open_since: float | None = None
        self.last_open_t = 0.0
        self.counted = False
        self.events: deque[float] = deque()

    def update(self, t: float, score: float | None) -> YawnStats:
        while self.events and self.events[0] <= t - self.window_s:
            self.events.popleft()
        is_open = score is not None and score >= self.threshold
        if is_open:
            if self.open_since is None:
                self.open_since, self.counted = t, False
            self.last_open_t = t
            if not self.counted and t - self.open_since >= self.min_s:
                self.events.append(t)   # 길이 조건을 넘는 순간 1회로 센다
                self.counted = True
        elif self.open_since is not None and t - self.last_open_t > self.gap_s:
            self.open_since = None
        yawning = self.open_since is not None and self.counted
        return YawnStats(yawning, len(self.events), 0.0 if score is None else score)
