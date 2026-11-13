"""Kiểm thử quỹ đạo của từng biểu cảm bằng mô phỏng động học robot vi sai (không cần Gazebo/ROS)."""

import math

import pytest

from tb4_emotion_control import emotions as E
from tb4_emotion_control.motions import build_behaviors, SegmentRun

DT = 0.05


def simulate(seg, seconds=60.0):
    """Tích phân (v, w) của SegmentRun trên pose lý tưởng; trả về (pose cuối, số bước)."""
    x = y = yaw = 0.0
    run = SegmentRun(seg, (x, y, yaw), 0.0)
    for i in range(int(seconds / DT)):
        run.update((x, y, yaw))
        if run.finished():
            return (x, y, yaw), run, i
        v, w = run.command()
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw += w * DT
    return (x, y, yaw), run, None


def test_mapping_covers_five_motions():
    b = build_behaviors()
    assert set(b) == {E.HAPPY, E.SAD, E.ANGRY, E.SURPRISE, E.FEAR}
    assert E.NEUTRAL not in b                  # neutral = dừng, không có quỹ đạo


def test_happy_curves_left_and_sad_curves_right():
    b = build_behaviors()
    for emotion, sign in ((E.HAPPY, 1), (E.SAD, -1)):
        seg = b[emotion].segments[0]
        assert b[emotion].continuous and seg.kind == 'hold'
        (x, y, yaw), run, done = simulate(seg, seconds=3.0)
        assert done is None                    # chạy liên tục, không tự kết thúc
        assert x > 0.3 and math.copysign(1, y) == sign and math.copysign(1, yaw) == sign
        assert seg.v / abs(seg.w) == pytest.approx(0.5)   # bán kính cong


def test_angry_backs_up_a_little():
    (x, y, yaw), _, done = simulate(build_behaviors()[E.ANGRY].segments[0])
    assert done is not None
    assert -0.22 <= x <= -0.197 and abs(y) < 1e-6


def test_fear_moves_forward_a_little():
    (x, y, yaw), _, done = simulate(build_behaviors()[E.FEAR].segments[0])
    assert done is not None
    assert 0.247 <= x <= 0.27


def test_surprise_spins_one_full_turn():
    (x, y, yaw), run, done = simulate(build_behaviors()[E.SURPRISE].segments[0])
    assert done is not None
    assert 2 * math.pi - 0.01 <= run.yaw_acc <= 2 * math.pi + 0.05   # góc tích lũy qua ±pi vẫn đúng
    assert math.hypot(x, y) < 1e-6


def test_slowdown_near_goal():
    seg = build_behaviors()[E.FEAR].segments[0]
    run = SegmentRun(seg, (0.0, 0.0, 0.0), 0.0)
    assert run.command()[0] == pytest.approx(seg.v)
    run.update((seg.goal - 0.01, 0.0, 0.0))
    assert run.command()[0] == pytest.approx(seg.v * 0.4)


def test_parameters_override():
    b = build_behaviors({'back_distance': 0.1, 'spin_angle_deg': 180.0, 'curve_radius': 1.0})
    assert b[E.ANGRY].segments[0].goal == pytest.approx(0.1)
    assert b[E.SURPRISE].segments[0].goal == pytest.approx(math.pi)
    assert b[E.HAPPY].segments[0].w == pytest.approx(0.15)


def test_spin_odom_gain_scales_turn_target():
    seg = build_behaviors({'spin_odom_gain': 1.06})[E.SURPRISE].segments[0]
    assert seg.goal == pytest.approx(2 * math.pi * 1.06)
