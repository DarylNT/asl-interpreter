import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import os
import threading

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
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

    def forward(self, x):
        return self.net(x)


# get cnn model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(26).to(DEVICE)
model.load_state_dict(torch.load("asl_landmark_model.pt", map_location=DEVICE))
model.eval()

# a-z
letter_classes = [chr(65 + i) for i in range(26)]

# ElevenLabs TTS (optional). Set ELEVENLABS_API_KEY in your environment to enable.
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE", "Rachel")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")

_elevenlabs_ready = False
try:
  if ELEVENLABS_API_KEY:
    # Prefer new SDK if available, else fallback to legacy API
    try:
      from elevenlabs import generate, play, set_api_key
      set_api_key(ELEVENLABS_API_KEY)
      _elevenlabs_ready = True
    except Exception:
      _elevenlabs_ready = False
  else:
    _elevenlabs_ready = False
except Exception:
  _elevenlabs_ready = False

def _speak_elevenlabs(text: str):
  if not _elevenlabs_ready:
    return
  try:
    audio = generate(text=text, voice=ELEVENLABS_VOICE, model=ELEVENLABS_MODEL)
    play(audio, notebook=False, use_ffmpeg=False)
  except Exception as e:
    # Non-fatal: just ignore speaking errors during live demo
    pass

def speak_async(text: str):
  if not text:
    return
  if not _elevenlabs_ready:
    return
  t = threading.Thread(target=_speak_elevenlabs, args=(text,), daemon=True)
  t.start()


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    h, w = image.shape[:2]
    left_hand_pred = None
    right_hand_pred = None
    left_hand_conf = 0.0
    right_hand_conf = 0.0
    # Track last spoken letters per hand
    if 'last_left_spoken' not in globals():
      globals()['last_left_spoken'] = None
    if 'last_right_spoken' not in globals():
      globals()['last_right_spoken'] = None
    
    # normal landmarks to draw on hand
    if results.multi_hand_landmarks and results.multi_handedness:
      for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        # draw a rectangle around the hand (bounding box from 2D landmarks)
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        pad = 10
        x_min = max(0, int(min(xs)) - pad)
        y_min = max(0, int(min(ys)) - pad)
        x_max = min(w - 1, int(max(xs)) + pad)
        y_max = min(h - 1, int(max(ys)) + pad)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # which hand is it :O
        handedness = results.multi_handedness[hand_idx]
        hand_label = handedness.classification[0].label  # left or right
        
        # world landmarks for it to be actually useful
        if results.multi_hand_world_landmarks:
          hand_world_landmarks = results.multi_hand_world_landmarks[hand_idx]
          landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_landmarks.landmark], dtype=np.float32)
          landmarks = (landmarks - landmarks.mean(axis=0)) / (landmarks.std(axis=0) + 1e-8)
          
          # convert stuff on screen to tensor and predict letter
          x = torch.from_numpy(landmarks).float().T.unsqueeze(0).to(DEVICE)  # (1, 3, 21)
          with torch.no_grad():
              output = model(x)
              probabilities = torch.softmax(output, dim=1)
              confidence, pred_idx = probabilities.max(dim=1)
              confidence = confidence.item() * 100
              predicted_letter = letter_classes[pred_idx.item()]
              
              # store prediction based on which hand it is
              if hand_label == "Left":
                left_hand_pred = predicted_letter
                left_hand_conf = confidence
              else:
                right_hand_pred = predicted_letter
                right_hand_conf = confidence

    # Speak on new sign changes (per hand) when confident enough
    CONF_THRESHOLD = 60.0
    if left_hand_pred and left_hand_conf >= CONF_THRESHOLD and left_hand_pred != globals().get('last_left_spoken'):
        speak_async(left_hand_pred)
        globals()['last_left_spoken'] = left_hand_pred
    if right_hand_pred and right_hand_conf >= CONF_THRESHOLD and right_hand_pred != globals().get('last_right_spoken'):
        speak_async(right_hand_pred)
        globals()['last_right_spoken'] = right_hand_pred
    
    # make it easier to see urself
    image = cv2.flip(image, 1)
    
    # show predicted letters
    # im just gonna flip them without changing the stuff up there bc its getting them reversed
    if right_hand_pred:
      cv2.putText(image, f"Left: {right_hand_pred}", (10, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
      cv2.putText(image, f"{right_hand_conf:.1f}%", (10, 100), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if left_hand_pred:
      cv2.putText(image, f"Right: {left_hand_pred}", (w - 260, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
      cv2.putText(image, f"{left_hand_conf:.1f}%", (w - 260, 100), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('ASL Recognition', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()