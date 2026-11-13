"""
Nhận dạng giọng nói tiếng Việt -> văn bản trên /voice/text.

Luồng xử lý (chế độ micro):
  arecord (PulseAudio, 16 kHz, mono, int16) -> khung 30 ms
    -> VAD năng lượng: ước lượng mức ồn nền, câu bắt đầu khi năng lượng > ồn nền x vad_ratio trong
       vad_start_frames khung liên tiếp, kết thúc khi im lặng vad_end_silence giây (giữ 0.3 s đầu câu)
    -> hàng đợi -> luồng nhận dạng:
         whisper (mặc định, faster-whisper, CPU int8): văn bản tự do có dấu; initial_prompt chứa tên
                  các khu vực để tăng độ chính xác; bỏ kết quả "ảo giác" (no_speech_prob cao, câu kinh điển
                  kiểu "Hãy subscribe cho kênh ...")
         vosk:    nhẹ hơn, offline; ngữ pháp ràng buộc vào tập câu lệnh sinh từ zones.yaml
    -> publish std_msgs/String lên /voice/text
Chế độ file (kiểm thử không cần micro): input = file .wav hoặc thư mục .wav (16 kHz mono), mỗi file là một câu.

Dùng:
  ros2 run tb4_voice_nav speech_recognizer
  ros2 run tb4_voice_nav speech_recognizer --ros-args -p engine:=vosk
  ros2 run tb4_voice_nav speech_recognizer --ros-args -p input:=/đường/dẫn/thư_mục_wav
"""

import glob
import json
import math
import os
import queue
import re
import signal
import sys
import threading
import time
import wave

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from tb4_voice_nav.audio import EnergyVAD, MicSource, RATE, read_wav
from tb4_voice_nav.command_parser import fold
from tb4_voice_nav.zones import ZoneMap

MODELS = os.path.join(os.path.expanduser('~'), 'tb4_ws', 'models')
DEFAULT_WHISPER = os.path.join(MODELS, 'whisper', 'faster-whisper-small')
DEFAULT_VOSK = os.path.join(MODELS, 'vosk', 'vosk-model-small-vn-0.4')
# Câu Whisper hay "bịa" ra khi gặp tiếng ồn / im lặng (học từ phụ đề video tiếng Việt)
HALLUCINATIONS = re.compile(r'(subscribe|đăng ký kênh|đăng kí kênh|theo dõi|cảm ơn các bạn|ghiền mì gõ|'
                            r'like và share|hẹn gặp lại|xin chào các bạn|phụ đề)', re.I)
# Câu có dạng câu lệnh (bỏ dấu): từ chỉ di chuyển / dừng / hỏi vị trí / khu vực. Lời nói khác trong phòng bỏ qua.
COMMAND_HINT = re.compile(r'\b(di|den|toi|ve|sang|vao|qua|tu|chay|dung|huy|stop|ngung|khu|vung|phong|'
                          r'o dau|vi tri|danh sach)\b')
# Cách đọc chữ cái (cho ngữ pháp Vosk)
SPOKEN_LETTERS = {'A': 'a', 'B': 'bê', 'C': 'xê', 'D': 'đê', 'E': 'e', 'F': 'ép', 'G': 'giê', 'H': 'hát',
                  'I': 'i', 'J': 'gi', 'K': 'ca', 'L': 'e lờ', 'M': 'em mờ', 'N': 'en nờ', 'O': 'o',
                  'P': 'pê', 'Q': 'quy', 'R': 'e rờ', 'S': 'ét', 'T': 'tê', 'U': 'u', 'V': 'vê',
                  'X': 'ích xì', 'Y': 'i dài', 'Z': 'dét'}


def default_zones_file():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('tb4_voice_nav'), 'config', 'zones.yaml')


# ---------------------------------------------------------------- bộ nhận dạng
class WhisperEngine:

    def __init__(self, model, device, compute_type, beam_size, language, prompt, log, cpu_threads=4):
        from faster_whisper import WhisperModel
        try:
            self.model = WhisperModel(model, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
        except Exception as e:  # thiếu thư viện CUDA 12 -> chạy CPU
            if device == 'cpu':
                raise
            log(f'Không dùng được {device} ({e}); chuyển sang CPU int8')
            self.model = WhisperModel(model, device='cpu', compute_type='int8', cpu_threads=cpu_threads)
        self.beam_size, self.language, self.prompt = beam_size, language, prompt

    def transcribe(self, pcm):
        audio = pcm.astype(np.float32) / 32768.0
        segments, _ = self.model.transcribe(
            audio, language=self.language, beam_size=self.beam_size, initial_prompt=self.prompt,
            condition_on_previous_text=False, temperature=0.0, vad_filter=True,
            vad_parameters={'min_silence_duration_ms': 500})
        texts = [s.text.strip() for s in segments
                 if not (s.no_speech_prob > 0.6 and s.avg_logprob < -0.8)]
        text = ' '.join(t for t in texts if t).strip()
        return '' if HALLUCINATIONS.search(text) else text


class VoskEngine:

    def __init__(self, model, grammar):
        from vosk import KaldiRecognizer, Model, SetLogLevel
        SetLogLevel(-1)
        self.model = Model(model)
        if grammar:   # bỏ câu chứa từ ngoài từ vựng của mô hình (Vosk sẽ bỏ qua và in cảnh báo)
            grammar = [g for g in grammar if g == '[unk]' or
                       all(self.model.find_word(w) >= 0 for w in g.split())]
        self.grammar_size = len(grammar) if grammar else 0
        # dựng bộ nhận dạng (biên dịch ngữ pháp) một lần; FinalResult() tự reset cho câu sau
        self.rec = KaldiRecognizer(self.model, RATE, json.dumps(grammar, ensure_ascii=False)) \
            if grammar else KaldiRecognizer(self.model, RATE)

    def transcribe(self, pcm):
        self.rec.AcceptWaveform(pcm.tobytes())
        text = json.loads(self.rec.FinalResult()).get('text', '')
        return text.replace('[unk]', '').strip()


def build_prompt(zones):
    names = ', '.join(dict.fromkeys(z.name for z in zones))
    return (f'Đi đến khu vực A. Di chuyển từ khu vực A đến khu vực B. Đi đến {names}. Dừng lại.')


def build_grammar(zones):
    """Tập câu cho Vosk: lệnh x (chữ cái A-Z | tên khu vực) + dừng / hủy."""
    targets = [f'khu vực {w}' for w in SPOKEN_LETTERS.values()]
    names = [z.name.lower() for z in zones]
    targets += names + [a.lower() for z in zones for a in z.aliases]
    targets = list(dict.fromkeys(targets))
    phrases = ['dừng lại', 'dừng', 'hủy', 'robot đang ở đâu', 'có những khu vực nào']
    for verb in ('đi đến', 'đi tới', 'di chuyển đến', 'di chuyển tới', 'đến', 'về'):
        phrases += [f'{verb} {t}' for t in targets]
    # "từ X đến Y" chỉ với chữ cái + tên chính (không bí danh) để ngữ pháp không quá lớn
    ends = list(dict.fromkeys([f'khu vực {w}' for w in SPOKEN_LETTERS.values()] + names))
    for a in ends:
        for b in ends:
            if a != b:
                phrases.append(f'di chuyển từ {a} đến {b}')
    return phrases + ['[unk]']


# ---------------------------------------------------------------- node
class SpeechRecognizer(Node):

    def __init__(self):
        super().__init__('speech_recognizer')
        self.declare_parameter('engine', 'whisper')        # whisper | vosk
        self.declare_parameter('whisper_model', DEFAULT_WHISPER)  # thư mục mô hình hoặc tên: small, medium
        self.declare_parameter('whisper_device', 'cpu')    # cpu | cuda (cần thư viện CUDA 12)
        self.declare_parameter('whisper_compute_type', 'int8')
        self.declare_parameter('beam_size', 5)
        self.declare_parameter('language', 'vi')
        self.declare_parameter('vosk_model', DEFAULT_VOSK)
        self.declare_parameter('vosk_grammar', True)       # ràng buộc câu lệnh theo zones.yaml
        self.declare_parameter('zones_file', '')           # '' = config/zones.yaml của package
        self.declare_parameter('input', 'mic')             # mic | file .wav | thư mục .wav
        self.declare_parameter('file_interval', 2.0)       # s nghỉ giữa các file khi input là thư mục
        self.declare_parameter('mic_device', 'default')    # thiết bị ALSA cho arecord
        self.declare_parameter('mic_source', '')           # nguồn PulseAudio ('' = mặc định hệ thống)
        self.declare_parameter('vad_ratio', 5.0)           # năng lượng câu nói / ồn nền (5 ~ 14 dB)
        self.declare_parameter('vad_min_db', -50.0)        # ngưỡng tuyệt đối tối thiểu (dBFS)
        self.declare_parameter('vad_start_frames', 3)
        self.declare_parameter('vad_end_silence', 0.8)     # s
        self.declare_parameter('vad_min_speech', 0.35)     # s
        self.declare_parameter('vad_max_speech', 8.0)      # s
        self.declare_parameter('text_topic', '/voice/text')
        self.declare_parameter('command_filter', True)     # chỉ publish câu có dạng câu lệnh
        self.declare_parameter('push_to_talk', False)      # True: nhấn Enter rồi mới nghe (phòng ồn)
        self.declare_parameter('ptt_window', 6.0)          # s nghe sau mỗi lần nhấn Enter
        self.declare_parameter('show_level', False)        # in mức âm thanh mỗi giây (chỉnh vad_*)
        self.declare_parameter('cpu_threads', 4)           # luồng CPU cho Whisper (chừa lõi cho Gazebo/Nav2)
        self.declare_parameter('nice', 10)                 # hạ ưu tiên: nhận dạng không được làm Nav2 lỡ heartbeat

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        if p('nice') > 0:
            os.nice(p('nice'))
        zones_file = os.path.expanduser(p('zones_file') or default_zones_file())
        zones = list(ZoneMap(zones_file)) if os.path.isfile(zones_file) else []
        self.pub = self.create_publisher(String, p('text_topic'), 10)
        self.input = p('input')
        self.file_interval = p('file_interval')
        self.show_level = p('show_level')
        self.command_filter = p('command_filter')
        self.ptt_window = p('ptt_window')
        self.zone_words = [fold(n) for z in zones for n in [z.name, *z.aliases]]
        self.listen_until = math.inf     # chế độ nghe liên tục
        self.vad = EnergyVAD(ratio=p('vad_ratio'), min_db=p('vad_min_db'),
                             start_frames=p('vad_start_frames'), end_silence=p('vad_end_silence'),
                             min_speech=p('vad_min_speech'), max_speech=p('vad_max_speech'))

        engine = p('engine')
        t0 = time.monotonic()
        if engine == 'vosk':
            grammar = build_grammar(zones) if p('vosk_grammar') and zones else None
            self.engine = VoskEngine(os.path.expanduser(p('vosk_model')), grammar)
            desc = f'Vosk ({self.engine.grammar_size or "không"} câu ngữ pháp)'
        else:
            self.engine = WhisperEngine(os.path.expanduser(p('whisper_model')), p('whisper_device'),
                                        p('whisper_compute_type'), p('beam_size'), p('language'),
                                        build_prompt(zones), self.get_logger().warn, p('cpu_threads'))
            desc = f'Whisper {os.path.basename(p("whisper_model"))} ({p("whisper_device")})'
            self.engine.transcribe(np.zeros(RATE // 2, np.int16))  # khởi động trước
        self.get_logger().info(f'Bộ nhận dạng: {desc}, nạp {time.monotonic() - t0:.1f} s; '
                               f'{len(zones)} khu vực từ {zones_file}')

        self.queue = queue.Queue()
        self.stop = threading.Event()
        self.mic = None
        self.count = 0
        if self.input == 'mic':
            self.mic = MicSource(p('mic_device'), p('mic_source'))
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            if p('push_to_talk') and sys.stdin.isatty():
                self.listen_until = 0.0
                threading.Thread(target=self.ptt_loop, daemon=True).start()
                self.get_logger().info(f'Push-to-talk: nhấn Enter rồi nói trong {self.ptt_window:.0f} s')
            else:
                if p('push_to_talk'):
                    self.get_logger().warn('push_to_talk cần terminal (ros2 run), chuyển sang nghe liên tục')
                self.get_logger().info('Đang nghe micro... (nói: "Đi đến khu vực A")')
        else:
            threading.Thread(target=self.file_loop, daemon=True).start()
        self.worker = threading.Thread(target=self.recognize_loop, daemon=True)
        self.worker.start()

    def ptt_loop(self):
        while not self.stop.is_set():
            try:
                input()
            except EOFError:
                return
            self.listen_until = time.monotonic() + self.ptt_window
            self.get_logger().info(f'>>> Mời nói ({self.ptt_window:.0f} s)')

    def capture_loop(self):
        last_print = time.monotonic()
        was_active = False
        started = 0.0
        try:
            while not self.stop.is_set():
                utt = self.vad.push(self.mic.read())
                if self.vad.active and not was_active:
                    started = time.monotonic()
                    if started <= self.listen_until:
                        self.get_logger().info('... đang nghe câu lệnh')
                was_active = self.vad.active
                if utt is not None and started <= self.listen_until:
                    self.queue.put((utt, 'mic'))
                if self.show_level and time.monotonic() - last_print > 1.0:
                    last_print = time.monotonic()
                    self.get_logger().info(f'mức {self.vad.level_db:6.1f} dB | ồn nền '
                                           f'{self.vad.noise_db:6.1f} dB | ngưỡng {self.vad.threshold():6.1f} dB')
        except EOFError as e:
            if not self.stop.is_set():
                self.get_logger().error(f'Mất micro: {e}')

    def file_loop(self):
        path = os.path.expanduser(self.input)
        files = sorted(glob.glob(os.path.join(path, '*.wav'))) if os.path.isdir(path) else [path]
        for i, f in enumerate(files):
            if self.stop.is_set():
                return
            try:
                self.queue.put((read_wav(f), os.path.basename(f)))
            except (OSError, ValueError, wave.Error) as e:
                self.get_logger().error(str(e))
            # chờ câu trước nhận dạng xong + thời gian nghỉ (cho node điều hướng xử lý)
            self.queue.join()
            if i + 1 < len(files):
                self.stop.wait(self.file_interval)
        self.get_logger().info(f'Đã xử lý {len(files)} file')

    def recognize_loop(self):
        while not self.stop.is_set():
            try:
                pcm, src = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                t0 = time.monotonic()
                text = self.engine.transcribe(pcm)
                dt = time.monotonic() - t0
                dur = len(pcm) / RATE
                folded = fold(text)
                if text and self.command_filter and not (COMMAND_HINT.search(folded) or
                                                         any(w and w in folded for w in self.zone_words)):
                    self.get_logger().info(f'[{src}, {dur:.1f} s] "{text}" không giống câu lệnh, bỏ qua')
                elif text:
                    self.count += 1
                    self.get_logger().info(f'[{src}, {dur:.1f} s âm thanh, nhận dạng {dt:.2f} s] "{text}"')
                    self.pub.publish(String(data=text))
                else:
                    self.get_logger().info(f'[{src}, {dur:.1f} s] không có lời nói rõ ràng, bỏ qua')
            except Exception as e:  # noqa: BLE001  lỗi một câu không được làm chết node
                self.get_logger().error(f'Lỗi nhận dạng: {e}')
            finally:
                self.queue.task_done()

    def close(self):
        self.stop.set()
        if self.mic is not None:
            self.mic.close()


def _interrupt(*_):
    raise KeyboardInterrupt


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _interrupt)
    signal.signal(signal.SIGTERM, _interrupt)
    node = None
    try:
        node = SpeechRecognizer()
        # node không có timer: rclpy.spin() chặn trong C nên handler SIGINT của Python không bao giờ chạy
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if node is not None:
            print(f'\n[speech_recognizer] đã nhận dạng {node.count} câu')
            node.close()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
