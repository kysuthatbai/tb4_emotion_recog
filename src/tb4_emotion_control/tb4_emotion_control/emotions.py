"""Nhãn biểu cảm dùng chung giữa emotion_recognition_node và emotion_behavior_node.

7 nhãn đầu là đầu ra của mô hình cảm xúc DeepFace (CNN 48x48 huấn luyện trên FER-2013).
NO_FACE / UNKNOWN do bộ lọc thêm vào: không thấy mặt / mô hình chưa đủ chắc chắn.
"""

ANGRY = 'angry'
DISGUST = 'disgust'
FEAR = 'fear'
HAPPY = 'happy'
SAD = 'sad'
SURPRISE = 'surprise'
NEUTRAL = 'neutral'
NO_FACE = 'no_face'
UNKNOWN = 'unknown'

# Thứ tự đầu ra của mô hình DeepFace
MODEL_LABELS = (ANGRY, DISGUST, FEAR, HAPPY, SAD, SURPRISE, NEUTRAL)

# 6 biểu cảm điều khiển robot (disgust không dùng: bị bỏ qua)
CONTROL = (HAPPY, SAD, NEUTRAL, ANGRY, SURPRISE, FEAR)

# Hành động tương ứng (hiện trên cửa sổ camera, chỉ ASCII vì cv2.putText không vẽ được dấu tiếng Việt)
ACTIONS = {
    HAPPY: 'curve LEFT',
    SAD: 'curve RIGHT',
    NEUTRAL: 'STOP',
    ANGRY: 'back up a little',
    SURPRISE: 'spin 360 deg',
    FEAR: 'forward a little',
}

VIETNAMESE = {
    HAPPY: 'Vui vẻ', SAD: 'Buồn', NEUTRAL: 'Bình thường', ANGRY: 'Tức giận',
    SURPRISE: 'Ngạc nhiên', FEAR: 'Sợ hãi', DISGUST: 'Ghê tởm', NO_FACE: 'Không thấy mặt',
    UNKNOWN: 'Chưa rõ',
}

COLORS = {  # BGR
    HAPPY: (0, 215, 255), SAD: (255, 120, 40), NEUTRAL: (200, 200, 200), ANGRY: (40, 40, 230),
    SURPRISE: (255, 0, 200), FEAR: (60, 200, 60), DISGUST: (40, 120, 120), NO_FACE: (120, 120, 120),
    UNKNOWN: (150, 150, 150),
}
