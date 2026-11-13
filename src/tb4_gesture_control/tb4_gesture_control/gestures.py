"""Tên cử chỉ dùng chung giữa gesture_recognition_node và robot_behavior_node.

7 nhãn đầu là nhãn có sẵn của MediaPipe Gesture Recognizer (mô hình canned),
WAVE do gesture_recognition_node tự phát hiện từ chuyển động qua lại của bàn tay.
"""

NONE = 'None'
CLOSED_FIST = 'Closed_Fist'
OPEN_PALM = 'Open_Palm'
POINTING_UP = 'Pointing_Up'
THUMB_DOWN = 'Thumb_Down'
THUMB_UP = 'Thumb_Up'
VICTORY = 'Victory'
I_LOVE_YOU = 'ILoveYou'
WAVE = 'Wave'

ALL = (CLOSED_FIST, OPEN_PALM, POINTING_UP, THUMB_DOWN, THUMB_UP, VICTORY, I_LOVE_YOU, WAVE)

# Ý nghĩa điều khiển (hiện trên cửa sổ webcam, chỉ ASCII vì cv2.putText không vẽ được dấu tiếng Việt)
ACTIONS = {
    WAVE: 'greet (turn left-right)',
    POINTING_UP: 'select FORWARD-BACK x4',
    VICTORY: 'select CIRCLE',
    THUMB_UP: 'CONFIRM / run',
    OPEN_PALM: 'STOP now',
    CLOSED_FIST: 'CANCEL -> idle',
}
