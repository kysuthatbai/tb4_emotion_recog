"""
Phần 1: đọc và hiển thị hình ảnh webcam laptop bằng OpenCV (không cần ROS).

Hiện FPS và độ phân giải thực tế; q / Esc để thoát.

Dùng:
  ros2 run tb4_gesture_control webcam_view            # /dev/video0
  ros2 run tb4_gesture_control webcam_view 1          # /dev/video1
"""

import sys
import time

import cv2


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f'[webcam_view] Không mở được /dev/video{index}')
        return 1

    window = f'webcam /dev/video{index}'
    fps, last = 0.0, time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print('[webcam_view] Không đọc được khung hình')
                break
            frame = cv2.flip(frame, 1)
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1.0 / max(now - last, 1e-6)
            last = now
            h, w = frame.shape[:2]
            cv2.putText(frame, f'{w}x{h}  {fps:4.1f} fps  (q: thoat)', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27) or \
                    cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f'[webcam_view] Đã đóng, FPS trung bình ~{fps:.1f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
