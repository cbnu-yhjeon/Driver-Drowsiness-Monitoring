from src.config import TemporalConfig
from src.temporal.alert import AlertLevel, DrowsinessJudge
from src.temporal.perclos import EyeClosureMonitor, YawnMonitor

FPS = 15


def run(judge, states, t0=0.0, mouth=None):
    """states: 프레임별 True(감음)/False(뜸)/None(알 수 없음). 마지막 판정과 전체 단계 목록을 돌려준다."""
    levels = []
    j = None
    for i, s in enumerate(states):
        j = judge.update(t0 + i / FPS, s, None if mouth is None else mouth[i])
        levels.append(j.level)
    return j, levels


def frames(seconds, value):
    return [value] * round(seconds * FPS)


def test_blink_is_not_a_warning():
    judge = DrowsinessJudge(TemporalConfig())
    states = (frames(2, False) + frames(0.3, True)) * 10
    _, levels = run(judge, states)
    assert AlertLevel.WARNING not in levels


def test_long_closure_triggers_warning_after_threshold():
    judge = DrowsinessJudge(TemporalConfig(closure_warning_s=1.5))
    j, levels = run(judge, frames(5, False) + frames(2.0, True))
    assert j.level == AlertLevel.WARNING
    first = levels.index(AlertLevel.WARNING)
    assert abs(first / FPS - (5 + 1.5)) < 0.15


def test_single_open_frame_inside_closure_is_ignored():
    mon = EyeClosureMonitor(gap_s=0.1)
    states = frames(1.0, True) + [False] + frames(1.0, True)
    for i, s in enumerate(states):
        st = mon.update(i / FPS, s)
    assert st.closure_s > 1.9


def test_two_open_frames_end_closure():
    mon = EyeClosureMonitor(gap_s=0.1)
    states = frames(1.0, True) + [False] * 3
    for i, s in enumerate(states):
        st = mon.update(i / FPS, s)
    assert st.closure_s == 0.0
    assert len(mon.events) == 1


def test_perclos_caution_only_in_full_mode():
    pattern = (frames(0.8, False) + frames(0.2, True)) * 70   # 20% 감음, 개별 감음은 짧음
    j_full, _ = run(DrowsinessJudge(TemporalConfig(mode="full")), pattern)
    j_cl, _ = run(DrowsinessJudge(TemporalConfig(mode="closure")), pattern)
    assert j_full.level == AlertLevel.CAUTION
    assert abs(j_full.closure.perclos - 0.2) < 0.03
    assert j_cl.level == AlertLevel.NORMAL


def test_perclos_needs_minimum_valid_time():
    judge = DrowsinessJudge(TemporalConfig(perclos_min_valid_s=10))
    j, _ = run(judge, (frames(0.6, False) + frames(0.4, True)) * 5)  # 5초뿐
    assert j.level == AlertLevel.NORMAL


def test_unknown_after_face_lost():
    judge = DrowsinessJudge(TemporalConfig(unknown_after_s=1.0))
    j, levels = run(judge, frames(2, False) + frames(1.5, None))
    assert j.level == AlertLevel.UNKNOWN
    assert levels[round(2.5 * FPS)] == AlertLevel.NORMAL


def test_warning_is_held():
    judge = DrowsinessJudge(TemporalConfig(warning_hold_s=3.0))
    j, levels = run(judge, frames(2, True) + frames(2, False))
    assert j.level == AlertLevel.WARNING
    j, _ = run(judge, frames(2, False), t0=4.0)
    assert j.level == AlertLevel.NORMAL


def test_yawn_counter():
    mon = YawnMonitor(threshold=0.4, min_s=1.0)
    ratios = ([0.1] * 30 + [0.6] * 30) * 3 + [0.6] * 5   # 2초 하품 3번 + 짧은 벌림
    for i, r in enumerate(ratios):
        st = mon.update(i / FPS, r)
    assert st.count == 3


def test_three_yawns_cause_caution():
    judge = DrowsinessJudge(TemporalConfig(yawn_threshold=0.4, yawn_caution_count=3))
    mouth = ([0.1] * 45 + [0.6] * 30) * 3
    j, _ = run(judge, [False] * len(mouth), mouth=mouth)
    assert j.level == AlertLevel.CAUTION
    assert j.yawn.count == 3
