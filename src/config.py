"""설정 로드와 실험 구성(D1~D5) 프리셋."""
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class TemporalConfig:
    mode: str = "full"
    perclos_window_s: float = 60.0
    perclos_min_valid_s: float = 10.0
    perclos_caution: float = 0.15
    perclos_warning: float = 0.25
    closure_warning_s: float = 1.5
    closure_gap_s: float = 0.1
    yawn_threshold: float = 0.5
    yawn_min_s: float = 1.0
    yawn_window_s: float = 300.0
    yawn_caution_count: int = 3
    unknown_after_s: float = 1.0
    warning_hold_s: float = 3.0


@dataclass(frozen=True)
class Config:
    detector: str = "haar"
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    detect_width: int = 320
    min_neighbors: int = 3
    track: bool = True
    redetect_every: int = 5
    max_track_s: float = 2.0
    classifier: str = "cnn"
    closed_threshold: float = 0.5
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text()) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        raw = dict(raw)
        temporal = TemporalConfig(**_known(TemporalConfig, raw.pop("temporal", {}) or {}))
        return cls(temporal=temporal, **_known(cls, raw))

    def with_temporal(self, **kw) -> "Config":
        return replace(self, temporal=replace(self.temporal, **kw))


def _known(dc, raw: dict) -> dict:
    names = {f.name for f in fields(dc)} - {"temporal"}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"알 수 없는 설정 키: {sorted(unknown)}")
    return raw


def preset(name: str, base: Config | None = None) -> Config:
    """README 실험 설계표의 D1~D5."""
    base = base or Config.load()
    name = name.upper()
    if name == "D1":
        return replace(base, detector="haar", track=False, clahe=False, classifier="svm").with_temporal(mode="closure")
    if name == "D2":
        return replace(base, detector="haar", track=False, clahe=False, classifier="cnn").with_temporal(mode="closure")
    if name == "D3":
        return replace(base, detector="haar", track=True, clahe=False, classifier="cnn").with_temporal(mode="full")
    if name == "D4":
        return replace(base, detector="haar", track=True, clahe=True, classifier="cnn").with_temporal(mode="full")
    if name == "D5":
        return replace(base, detector="lbp", track=True, clahe=True, classifier="cnn").with_temporal(mode="full")
    raise ValueError(f"알 수 없는 프리셋: {name} (D1~D5)")


PRESETS = ("D1", "D2", "D3", "D4", "D5")
