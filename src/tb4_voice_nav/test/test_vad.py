"""Kiểm thử VAD năng lượng bằng tín hiệu tổng hợp (không cần micro).

Chạy: python3 -m pytest src/tb4_voice_nav/test -q
"""

import numpy as np

from tb4_voice_nav.audio import EnergyVAD, FRAME, RATE

RNG = np.random.default_rng(0)


def noise(seconds, db):
    return RNG.normal(0, 32768 * 10 ** (db / 20), int(seconds * RATE))


def burst(seconds, db, f=220.0):
    """'Lời nói' giả: tổng hài điều biên 4 Hz (có lên xuống như âm tiết)."""
    t = np.arange(int(seconds * RATE)) / RATE
    x = sum(np.sin(2 * np.pi * f * k * t) / k for k in (1, 2, 3)) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
    return x / np.abs(x).max() * 32768 * 10 ** (db / 20)


def segments(x):
    v = EnergyVAD(ratio=5.0)
    pcm = np.clip(x, -32768, 32767).astype(np.int16)
    out = []
    for i in range(0, len(pcm) - FRAME, FRAME):
        u = v.push(pcm[i:i + FRAME])
        if u is not None:
            out.append(len(u) / RATE)
    return out


def test_steady_noise_never_triggers():
    assert segments(noise(10, -45)) == []


def test_command_in_noise_is_one_segment():
    x = np.concatenate([noise(3, -45), noise(2, -45) + burst(2, -15), noise(2, -45)])
    out = segments(x)
    assert len(out) == 1
    assert 2.0 <= out[0] <= 3.5          # 2 s lời + 0.3 s đầu câu + 0.8 s im lặng cuối


def test_digital_silence_source():
    """Micro sạch / noise gate: im lặng tuyệt đối (0) trước câu đầu không làm mất đầu câu."""
    x = np.concatenate([np.zeros(3 * RATE), burst(1.5, -20), np.zeros(2 * RATE), burst(1.5, -20), np.zeros(RATE)])
    out = segments(x)
    assert len(out) == 2
    assert abs(out[0] - out[1]) < 0.1


def test_noise_level_change_adapts():
    """Ồn nền tăng 15 dB (bật quạt): sau vài giây ngưỡng theo kịp, không cắt câu dài 8 s liên tục."""
    x = np.concatenate([noise(3, -55), noise(10, -40)])
    assert len(segments(x)) <= 1
