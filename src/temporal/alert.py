"""경보 단계 판단: 정상 · 주의 · 경고 · 판단 불가."""
from dataclasses import dataclass, field
from enum import Enum

from src.config import TemporalConfig
from src.temporal.perclos import ClosureStats, EyeClosureMonitor, YawnMonitor, YawnStats


class AlertLevel(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    WARNING = "WARNING"
    UNKNOWN = "UNKNOWN"


@dataclass
class Judgement:
    level: AlertLevel
    reasons: list[str] = field(default_factory=list)
    closure: ClosureStats | None = None
    yawn: YawnStats | None = None
    unknown_s: float = 0.0


class DrowsinessJudge:
    """mode='closure': 연속 감음만 (D1·D2). mode='full': PERCLOS + 연속 감음 + 하품 (D3~D5)."""

    def __init__(self, cfg: TemporalConfig):
        if cfg.mode not in ("closure", "full"):
            raise ValueError(f"temporal.mode 는 closure | full: {cfg.mode}")
        self.cfg = cfg
        self.eyes = EyeClosureMonitor(cfg.perclos_window_s, cfg.closure_gap_s)
        self.yawns = YawnMonitor(cfg.yawn_threshold, cfg.yawn_min_s, cfg.yawn_window_s)
        self.warning_until = -1.0
        self.last_seen: float | None = None
        self.start: float | None = None

    def update(self, t: float, closed: bool | None, yawn_score: float | None = None) -> Judgement:
        c = self.cfg
        if self.start is None:
            self.start = t
        if closed is not None:
            self.last_seen = t
        unknown_s = t - (self.last_seen if self.last_seen is not None else self.start)
        eye = self.eyes.update(t, closed)
        yawn = self.yawns.update(t, yawn_score if c.mode == "full" else None)

        warn, caution = [], []
        if eye.closure_s >= c.closure_warning_s:
            warn.append(f"눈 감음 {eye.closure_s:.1f}s")
        if c.mode == "full":
            if eye.valid_s >= c.perclos_min_valid_s:
                if eye.perclos >= c.perclos_warning:
                    warn.append(f"PERCLOS {eye.perclos:.0%}")
                elif eye.perclos >= c.perclos_caution:
                    caution.append(f"PERCLOS {eye.perclos:.0%}")
            if yawn.count >= c.yawn_caution_count:
                caution.append(f"하품 {yawn.count}회")

        if warn:
            self.warning_until = t + c.warning_hold_s
            level, reasons = AlertLevel.WARNING, warn
        elif t < self.warning_until:
            level, reasons = AlertLevel.WARNING, ["경고 유지"]
        elif unknown_s >= c.unknown_after_s:
            level, reasons = AlertLevel.UNKNOWN, [f"얼굴/눈 없음 {unknown_s:.1f}s"]
        elif caution:
            level, reasons = AlertLevel.CAUTION, caution
        else:
            level, reasons = AlertLevel.NORMAL, []
        return Judgement(level, reasons, eye, yawn, unknown_s)
