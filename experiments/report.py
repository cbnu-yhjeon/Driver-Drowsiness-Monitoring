"""results/*.json → results/RESULTS.md (표 정리).

    python -m experiments.report
"""
import json

from experiments.common import RESULTS

COND_KO = {"normal": "일반", "lowlight": "저조도(합성)", "glare": "안경 반사(합성)"}


def load(name: str):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def eye_state(d) -> list[str]:
    out = ["## 1. 눈 상태 분류 (G1 · G3)", "",
           f"시험 분할 {d['n_images']}장 × 양 눈 = {2 * d['n_images']} 패치. 정답 눈 위치에서 자른 패치(검출 오류 제외).", "",
           "| 조건 | 전처리 | HoG+SVM F1 | CNN fp32 F1 | CNN int8 F1 | CNN int8 FPR / FNR |", "|---|---|---|---|---|---|"]
    rows = {(r["condition"], r["preprocess"], r["classifier"]): r for r in d["rows"]}
    for cond in COND_KO:
        for pre in ("raw", "clahe"):
            g = lambda c: rows[(cond, pre, c)]
            out.append(f"| {COND_KO[cond]} | {'CLAHE' if pre == 'clahe' else '없음'} | {g('hog_svm')['f1']:.3f} | "
                       f"{g('cnn_fp32')['f1']:.3f} | {g('cnn_int8')['f1']:.3f} | "
                       f"{pct(g('cnn_int8')['fpr'])} / {pct(g('cnn_int8')['fnr'])} |")
    lat = d["latency_ms_per_frame(2 eyes)"]
    out += ["", "프레임당 분류 시간(양 눈, 1스레드): " + ", ".join(f"{k} {v:.2f} ms" for k, v in lat.items()), ""]
    return out


def pipeline(d) -> list[str]:
    out = ["## 2. 전체 파이프라인 (정지 영상, D1~D5)", "",
           "얼굴 검출 → 기하 비율 눈 위치 → 분류. 가장 큰 얼굴이 정답 눈을 포함하지 않으면 검출 실패. "
           "'종단 정확도'는 검출 실패를 오답으로 센 값. 정지 영상이라 LK 추적 효과(D3)는 여기서 드러나지 않는다.", "",
           "| 구성 | 조건 | 얼굴 검출률 | 눈 상태 F1 (검출된 얼굴) | 종단 정확도 |", "|---|---|---|---|---|"]
    for dname, per in d["D"].items():
        for cond, r in per.items():
            out.append(f"| {dname} | {COND_KO[cond]} | {pct(r['face_rate'])} | {r['f1']:.3f} | {pct(r['e2e_acc'])} |")
    le = d["eye_localization_error"]
    out += ["", "눈 위치 오차(IOD 대비 중앙값, Haar · 전처리 없음 · 일반): "
            f"기하 비율 {le['haar/raw/normal/geometry']['median_iod']:.3f}, "
            f"Haar eye 보정 {le['haar/raw/normal/haar_eye']['median_iod']:.3f}", ""]
    return out


def yawn(d) -> list[str]:
    r, s = d["rule_mouth_ratio"], d["hog_svm"]
    return ["## 3. 하품 판별", "",
            f"c3rl/yawning-people 일부 {sum(d['images'].values())}장, 얼굴 검출된 {d['n_train']} / {d['n_test']}장으로 학습 / 시험.", "",
            "| 방법 | 시험 AUC | F1 | 정확도 |", "|---|---|---|---|",
            f"| 입 벌림 비율 규칙 (README 초안) | {r['test_auc']:.3f} | {r['test']['f1']:.3f} | {pct(r['test']['acc'])} |",
            f"| 입 영역 HoG + SVM (채택) | {s['test_auc']:.3f} | {s['test']['f1']:.3f} | {pct(s['test']['acc'])} |", "",
            f"Haar 얼굴 검출률: 하품 {pct(d['face_rate']['yawning'])} vs 하품 아님 {pct(d['face_rate']['not_yawning'])} "
            "→ 크게 벌린 입은 정면 얼굴 검출을 깨뜨린다. 검출 사이를 LK 추적으로 잇는 이유.", ""]


def temporal(d) -> list[str]:
    s = d["setup"]
    out = ["## 4. 시간 창 판단 (G2, 합성 운전 세션)", "",
           f"구성 · 조건마다 {s['sessions_per_job']}개 × {s['session_s'] // 60}분 세션 (절반은 졸음 구간 포함). "
           "프레임 오류율은 2절 · 3절 측정값. 졸음 구간 끝 + "
           f"{s['grace_s']:.0f}초 안의 경고를 맞힌 것으로 본다.", "",
           "임계값 선택(검증 세션, D4 · 일반): 재현율 ≥ 0.95 를 만족하는 조합이 없어 오경보가 가장 적은 조합 중 재현율 최대 → "
           f"**연속 감음 ≥ {d['chosen']['closure_warning_s']}초, PERCLOS ≥ {d['chosen']['perclos_warning']:.0%}**", "",
           "| 연속 감음 | PERCLOS 경고 | 재현율 | 정밀도 | 시간당 오경보 |", "|---|---|---|---|---|"]
    for r in d["sweep_val"]:
        out.append(f"| {r['closure_warning_s']}초 | {r['perclos_warning']:.0%} | {r['recall']:.3f} | {r['precision']:.3f} | "
                   f"{r['false_alarms_per_h']:.2f} |")
    out += ["", "시험 세션 결과 (경고 단계 기준):", "",
            "| 구성 | 조건 | 재현율 | 정밀도 | 시간당 오경보 | 경고까지 걸린 시간(중앙값) | 주의 이상 재현율 |",
            "|---|---|---|---|---|---|---|"]
    for dname, per in d["test"].items():
        for cond, r in per.items():
            w, a = r["warning"], r["caution_or_warning"]
            lat = "-" if w["median_latency_s"] is None else f"{w['median_latency_s']:.1f}초"
            out.append(f"| {dname} | {COND_KO[cond]} | {w['recall']:.3f} | {w['precision']:.3f} | "
                       f"{w['false_alarms_per_h']:.2f} | {lat} | {a['recall']:.3f} |")
    return out + [""]


def fps(d) -> list[str]:
    out = ["## 5. CPU 속도 (G4)", "", f"CPU: {d['cpu']}, 입력 {d['frame_size'][0]}×{d['frame_size'][1]}, "
           f"합성 주행 영상 {d['frames']} 프레임.", "",
           "| 구성 | FPS (기본 스레드) | FPS (1스레드) | 프레임당 ms (1스레드: 전처리 / 얼굴 / 눈 / 분류 / 하품) |",
           "|---|---|---|---|"]
    for dname in d["single_thread"]:
        a, b = d["default_threads"][dname], d["single_thread"][dname]
        m = b["ms_mean"]
        parts = " / ".join(f"{m.get(k, 0):.2f}" for k in ("preprocess", "face", "eyes", "classify", "yawn"))
        out.append(f"| {dname} | {a['fps']:.0f} | {b['fps']:.0f} | {parts} |")
    return out + [""]


def main() -> None:
    lines = ["# 실험 결과", "", "`python -m experiments.report` 가 results/*.json 에서 만든 문서입니다.", ""]
    for name, fn in (("eye_state", eye_state), ("pipeline", pipeline), ("yawn", yawn),
                     ("temporal", temporal), ("fps", fps)):
        d = load(name)
        if d is not None:
            lines += fn(d)
    (RESULTS / "RESULTS.md").write_text("\n".join(lines))
    print("->", RESULTS / "RESULTS.md")


if __name__ == "__main__":
    main()
