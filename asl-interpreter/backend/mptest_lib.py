# mptest_lib.py — headless adapter that mirrors your model+mediapipe logic
import numpy as np
import torch, torch.nn as nn
import mediapipe as mp

mp_hands = mp.solutions.hands

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.net(x)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = CNN(26).to(DEVICE)
_model.load_state_dict(torch.load("asl_landmark_model.pt", map_location=DEVICE))
_model.eval()
_CLASSES = [chr(65 + i) for i in range(26)]

_hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

def process_frame(img_rgb_np: np.ndarray):
    """
    Input: HxWx3 uint8 RGB (browser-sent frame)
    Return: (prediction_text:str, landmarks_px:list[[x,y]], width:int, height:int)
    """
    h, w = img_rgb_np.shape[:2]
    results = _hands.process(img_rgb_np)

    left_pred = right_pred = None
    left_conf = right_conf = 0.0
    landmarks_px = []

    if results.multi_hand_landmarks:
        for hlm in results.multi_hand_landmarks:
            xs = [int(lm.x * w) for lm in hlm.landmark]
            ys = [int(lm.y * h) for lm in hlm.landmark]
            landmarks_px.extend([[x, y] for x, y in zip(xs, ys)])

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, _ in enumerate(results.multi_hand_landmarks):
            handed = results.multi_handedness[i].classification[0].label  # "Left"/"Right"
            if results.multi_hand_world_landmarks:
                world = results.multi_hand_world_landmarks[i]
                lms = np.array([[lm.x, lm.y, lm.z] for lm in world.landmark], dtype=np.float32)
            else:
                norm = results.multi_hand_landmarks[i]
                lms = np.array([[lm.x, lm.y, 0.0] for lm in norm.landmark], dtype=np.float32)

            lms = (lms - lms.mean(axis=0)) / (lms.std(axis=0) + 1e-8)
            x = torch.from_numpy(lms).float().T.unsqueeze(0).to(DEVICE)  # (1,3,21)
            with torch.no_grad():
                probs = torch.softmax(_model(x), dim=1)
                conf, idx = probs.max(dim=1)
                conf = conf.item() * 100.0
                pred = _CLASSES[idx.item()]
            if handed == "Left":  left_pred, left_conf = pred, conf
            else:                 right_pred, right_conf = pred, conf

    parts = []
    # Mirror labels like your original script’s display
    if right_pred: parts.append(f"L:{right_pred} {right_conf:.1f}%")
    if left_pred:  parts.append(f"R:{left_pred} {left_conf:.1f}%")
    return " | ".join(parts), landmarks_px, w, h
