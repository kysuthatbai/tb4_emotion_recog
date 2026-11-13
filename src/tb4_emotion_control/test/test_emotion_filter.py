"""Kiểm thử bộ lọc biểu cảm bằng chuỗi khung tổng hợp (15 khung/s), không cần webcam/ROS/DeepFace.

Chạy: cd ~/tb4_ws && python3 -m pytest src/tb4_emotion_control/test -q
"""

from tb4_emotion_control import emotions as E
from tb4_emotion_control.emotion_filter import EmotionFilter

DT = 1.0 / 15.0


def probs(label, p=0.8):
    rest = (1.0 - p) / (len(E.MODEL_LABELS) - 1)
    return {k: (p if k == label else rest) for k in E.MODEL_LABELS}


def run(frames, f=None, t0=0.0):
    f = f or EmotionFilter()
    return [f.update(fr, t0 + i * DT)[0] for i, fr in enumerate(frames)], f


def first(out, label):
    return out.index(label) * DT


def test_starts_without_face():
    out, _ = run([None] * 5)
    assert set(out) == {E.NO_FACE}


def test_steady_expression_becomes_stable_after_stable_time():
    out, _ = run([probs(E.HAPPY)] * 15)
    assert 0.39 <= first(out, E.HAPPY) <= 0.5
    assert out[-1] == E.HAPPY


def test_low_confidence_is_unknown():
    out, _ = run([probs(E.SAD, 0.35)] * 15)
    assert out[-1] == E.UNKNOWN


def test_close_second_is_unknown():
    p = {k: 0.0 for k in E.MODEL_LABELS}
    p[E.SAD], p[E.NEUTRAL] = 0.5, 0.45
    out, _ = run([p] * 15)
    assert out[-1] == E.UNKNOWN


def test_single_frame_flicker_does_not_switch():
    frames = [probs(E.NEUTRAL)] * 15 + ([probs(E.SAD, 0.95)] + [probs(E.NEUTRAL)] * 3) * 5
    out, _ = run(frames)
    assert E.SAD not in out


def test_switch_takes_smoothing_plus_stable_time():
    frames = [probs(E.HAPPY)] * 15 + [probs(E.SURPRISE, 0.9)] * 20
    out, _ = run(frames)
    t_switch = first(out, E.SURPRISE) - 15 * DT
    assert 0.4 <= t_switch <= 0.9
    assert out[-1] == E.SURPRISE


def test_brief_face_loss_keeps_label():
    frames = [probs(E.FEAR)] * 15 + [None] * 4 + [probs(E.FEAR)] * 5   # mất mặt 0.27 s
    out, _ = run(frames)
    assert E.NO_FACE not in out[15:]


def test_long_face_loss_gives_no_face_and_resets():
    frames = [probs(E.ANGRY)] * 15 + [None] * 10 + [probs(E.HAPPY)] * 10
    out, f = run(frames)
    assert E.NO_FACE in out[15:25]
    assert 0.45 <= first(out[15:], E.NO_FACE) <= 0.6
    # EMA đã xóa: không bị angry kéo theo, happy ổn định sau đúng stable_time
    assert out[-1] == E.HAPPY
    assert E.ANGRY not in out[25:]


def test_unnormalized_probabilities_are_ok():
    p = {k: v * 100 for k, v in probs(E.SAD).items()}   # DeepFace trả về phần trăm
    out, _ = run([p] * 15)
    assert out[-1] == E.SAD
