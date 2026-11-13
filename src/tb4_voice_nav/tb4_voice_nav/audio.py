"""Thu âm (arecord), đọc WAV và tách câu nói bằng năng lượng (VAD). Không phụ thuộc ROS."""

from collections import deque
import math
import os
import subprocess
import wave

import numpy as np

RATE = 16000
FRAME = 480  # 30 ms


class MicSource:
    """Đọc PCM từ arecord (không cần PortAudio). source: tên nguồn PulseAudio, '' = mặc định."""

    def __init__(self, device='default', source=''):
        env = dict(os.environ)
        if source:
            env['PULSE_SOURCE'] = source
        self.proc = subprocess.Popen(
            ['arecord', '-q', '-D', device, '-f', 'S16_LE', '-r', str(RATE), '-c', '1', '-t', 'raw'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    def read(self):
        data = self.proc.stdout.read(FRAME * 2)
        if len(data) < FRAME * 2:
            err = self.proc.stderr.read().decode(errors='ignore').strip() if self.proc.poll() is not None else ''
            raise EOFError(err or 'arecord dừng')
        return np.frombuffer(data, np.int16)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def read_wav(path):
    with wave.open(path) as w:
        if w.getframerate() != RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f'{path}: cần WAV 16 kHz, mono, 16 bit')
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


class EnergyVAD:
    """Tách câu theo năng lượng với ngưỡng thích nghi theo mức ồn nền."""

    def __init__(self, ratio=3.0, min_db=-50.0, start_frames=3, end_silence=0.8, min_speech=0.35,
                 max_speech=8.0, pre_roll=0.3):
        self.ratio_db = 20 * math.log10(ratio)
        self.min_db = min_db
        self.start_frames = start_frames
        self.end_frames = int(end_silence / (FRAME / RATE))
        self.min_frames = int(min_speech / (FRAME / RATE))
        self.max_frames = int(max_speech / (FRAME / RATE))
        self.pre = deque(maxlen=int(pre_roll / (FRAME / RATE)))
        self.history = deque(maxlen=100)   # mức (dB) ~3 s gần nhất khi không nói
        self.seen_signal = False
        self.zero_run = 0
        self.noise_db = None
        self.active = False
        self.buf, self.loud, self.quiet = [], 0, 0
        self.level_db = -90.0

    @staticmethod
    def db(frame):
        rms = math.sqrt(float(np.mean(frame.astype(np.float32) ** 2)) + 1e-9) / 32768.0
        return 20 * math.log10(rms + 1e-9)

    def threshold(self):
        return max((self.noise_db if self.noise_db is not None else -60.0) + self.ratio_db, self.min_db)

    def push(self, frame):
        """Trả về mảng int16 của một câu khi câu kết thúc, ngược lại None."""
        level = self.db(frame)
        self.level_db = level
        if not self.active:
            # Ồn nền = phân vị 20% của ~3 s gần nhất: tiếng nói ngắt quãng không kéo ngưỡng lên,
            # ồn tăng thật (quạt, máy tính) thì ngưỡng theo kịp. Bỏ các khung số 0 lúc arecord vừa mở
            # (chưa có tín hiệu); sau đó khung im lặng số được kẹp -90 dB (micro sạch / có noise gate).
            if level > -100.0:
                self.seen_signal = True
            else:
                self.zero_run += 1      # im lặng số kéo dài (> 0.3 s) = nguồn sạch thật, không phải lúc mở
            if self.seen_signal or self.zero_run > 10:
                self.history.append(max(level, -90.0))
            if len(self.history) < 15:      # đang hiệu chuẩn ~0.5 s đầu
                self.pre.append(frame)
                return None
            self.noise_db = float(np.percentile(self.history, 20))
        speech = level > self.threshold()
        if not self.active:
            self.pre.append(frame)
            self.loud = self.loud + 1 if speech else 0
            if self.loud >= self.start_frames:
                self.active, self.buf, self.quiet = True, list(self.pre), 0
            return None
        self.buf.append(frame)
        self.quiet = 0 if speech else self.quiet + 1
        if self.quiet >= self.end_frames or len(self.buf) >= self.max_frames:
            self.active, self.loud = False, 0
            self.pre.clear()
            n_speech = len(self.buf) - self.quiet
            utt, self.buf = np.concatenate(self.buf), []
            return utt if n_speech >= self.min_frames else None
        return None
