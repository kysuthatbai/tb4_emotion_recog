"""Kiểm thử bộ lọc nhiễu + phát hiện Wave bằng chuỗi khung tổng hợp (30 fps), không cần webcam/ROS.

Chạy: cd ~/tb4_ws && colcon test --packages-select tb4_gesture_control --event-handlers console_direct+
  hoặc: python3 -m pytest src/tb4_gesture_control/test -q
"""

import math

from tb4_gesture_control import gestures as G
from tb4_gesture_control.gesture_recognition_node import GestureFilter

DT = 1.0 / 30.0


def run(frames, f=None, t0=0.0):
    """frames: list (raw, score, cx). Trả về danh sách nhãn ổn định từng khung."""
    f = f or GestureFilter()
    return [f.update(raw, score, cx, t0 + i * DT)[0] for i, (raw, score, cx) in enumerate(frames)], f


def test_static_gesture_becomes_stable_after_votes():
    out, _ = run([(G.VICTORY, 0.9, 0.5)] * 10)
    assert out[:4] == [G.NONE] * 4          # chưa đủ 5 phiếu
    assert out[4:] == [G.VICTORY] * 6


def test_low_confidence_is_ignored():
    out, _ = run([(G.THUMB_UP, 0.4, 0.5)] * 20)
    assert set(out) == {G.NONE}


def test_flicker_never_triggers():
    frames = [(G.POINTING_UP, 0.9, 0.5), (G.NONE, 0.0, None)] * 15
    out, _ = run(frames)
    assert set(out) == {G.NONE}


def test_still_open_palm_is_stop():
    out, _ = run([(G.OPEN_PALM, 0.8, 0.5)] * 10)
    assert out[-1] == G.OPEN_PALM


def waving(seconds, freq=2.0, amp=0.1, t_offset=0.0):
    n = int(seconds / DT)
    return [(G.OPEN_PALM, 0.8, 0.5 + amp * math.sin(2 * math.pi * freq * (i * DT + t_offset)))
            for i in range(n)]


def test_wave_detected_and_not_confused_with_stop():
    out, _ = run(waving(1.5))
    assert G.WAVE in out
    assert G.OPEN_PALM not in out           # tay xòe nhưng đang chuyển động -> không phải lệnh dừng


def test_palm_after_wave_is_blocked_then_allowed():
    f = GestureFilter()
    wave = waving(1.5)
    out1, _ = run(wave, f)
    t = len(wave) * DT
    still = [(G.OPEN_PALM, 0.8, 0.5)] * 90  # giữ yên 3 s
    out2, _ = run(still, f, t0=t)
    # <= 0.5 s chờ tay hết chuyển động + 0.5 s giữ Wave + 0.5 s chặn Open_Palm + 5 khung bỏ phiếu
    first_palm = out2.index(G.OPEN_PALM) * DT
    assert 0.9 <= first_palm <= 1.8
    assert out2[0] == G.WAVE                # ngay khi vừa dừng vẫy vẫn còn là Wave
    assert G.WAVE in out1


def test_small_jitter_is_not_wave():
    out, _ = run(waving(2.0, amp=0.02))
    assert G.WAVE not in out
    assert out[-1] == G.OPEN_PALM


def test_moving_thumb_up_is_not_wave():
    frames = [(G.THUMB_UP, 0.9, 0.5 + 0.1 * math.sin(2 * math.pi * 2 * i * DT)) for i in range(45)]
    out, _ = run(frames)
    assert G.WAVE not in out
    assert out[-1] == G.THUMB_UP
