"""
Gõ câu lệnh bằng bàn phím thay cho giọng nói (kiểm thử khi không có micro) và xem phản hồi của robot.

Mỗi dòng gõ vào được publish lên /voice/text (giống hệt đầu ra của speech_recognizer);
phản hồi trên /voice/feedback được in ra. Ctrl+D hoặc Ctrl+C để thoát.

Dùng:
  ros2 run tb4_voice_nav voice_console
  ros2 run tb4_voice_nav voice_console "Đi đến khu vực A"      # gửi 1 câu rồi thoát
"""

import sys
import threading
import time

import rclpy
import rclpy.executors
from rclpy.node import Node
from std_msgs.msg import String


class VoiceConsole(Node):

    def __init__(self):
        super().__init__('voice_console')
        self.pub = self.create_publisher(String, '/voice/text', 10)
        self.create_subscription(String, '/voice/feedback', self.on_feedback, 10)
        self.create_subscription(String, '/voice/text', self.on_text, 10)

    def on_feedback(self, msg):
        print(f'\r  robot> {msg.data}\n> ', end='', flush=True)

    def on_text(self, msg):
        print(f'\r  nghe > "{msg.data}"\n> ', end='', flush=True)

    def send(self, text):
        self.pub.publish(String(data=text))


def main():
    rclpy.init()
    node = VoiceConsole()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    end = time.monotonic() + 5.0
    while node.pub.get_subscription_count() < 2 and time.monotonic() < end:  # chờ voice_navigator
        time.sleep(0.1)
    try:
        if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
            node.send(' '.join(a for a in sys.argv[1:] if not a.startswith('--')))
            time.sleep(1.5)
            return
        print('Gõ câu lệnh (ví dụ: Đi đến khu vực A | Di chuyển từ khu vực A đến khu vực B | Dừng lại)')
        while True:
            line = input('> ').strip()
            if line:
                node.send(line)
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        # dừng executor và chờ luồng spin thoát trước khi hủy node (không thì C++ abort khi thoát)
        executor.shutdown()
        spinner.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
