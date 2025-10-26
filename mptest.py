import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
import os
import threading
import time
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

# get api key
load_dotenv()
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# basic drawing stuffs
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# speaking trackers
last_left_spoken = None
last_right_spoken = None
last_spoken_time = 0.0
left_hold_letter = None
left_hold_start = 0.0
right_hold_letter = None
right_hold_start = 0.0


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

# speak on threads bc it was freezing cam before
speak_thread = None
def speak(text: str):
  global speak_thread
  if not text:
    return
  if speak_thread is not None and speak_thread.is_alive():
    return
  def _run():
    try:
      audio = client.text_to_speech.convert(text=text, voice_id="JBFqnCBsd6RMkjVDRZzb", model_id="eleven_multilingual_v2", output_format="mp3_44100_128")
      play(audio, notebook=False, use_ffmpeg=False)
    except Exception as e:
        print(f"[TTS ERROR] ElevenLabs failed: {e}")
  speak_thread = threading.Thread(target=_run, daemon=True)
  speak_thread.start()


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

    # make image ez to process
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w = image.shape[:2]

    # vars for predictions and confidence
    left_hand_pred = None
    right_hand_pred = None
    left_hand_conf = 0.0
    right_hand_conf = 0.0
    
    # normal landmarks to draw on hand
    if results.multi_hand_landmarks and results.multi_handedness:
      for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        # draw a box around the hand
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        pad = 10
        x_min = max(0, int(min(xs)) - pad)
        y_min = max(0, int(min(ys)) - pad)
        x_max = min(w - 1, int(max(xs)) + pad)
        y_max = min(h - 1, int(max(ys)) + pad)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # which hand is it 😮
        handedness = results.multi_handedness[hand_idx]
        hand_label = handedness.classification[0].label  # left or right

        # world landmarks for prediction
        if results.multi_hand_world_landmarks:
          hand_world_landmarks = results.multi_hand_world_landmarks[hand_idx]
          landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_landmarks.landmark], dtype=np.float32)
          landmarks = (landmarks - landmarks.mean(axis=0)) / (landmarks.std(axis=0) + 1e-8)
          x = torch.from_numpy(landmarks).float().T.unsqueeze(0).to(DEVICE)  # (1, 3, 21)
          with torch.no_grad():
            output = model(x)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred_idx = probabilities.max(dim=1)
            confidence = confidence.item() * 100
            predicted_letter = letter_classes[pred_idx.item()]
          if hand_label == "Left":
            left_hand_pred = predicted_letter
            left_hand_conf = confidence
          else:
            right_hand_pred = predicted_letter
            right_hand_conf = confidence

    # speak only if pose is held for at least 0.15s
    CONF_THRESHOLD = 49.0
    HOLD_SEC = 0.15

    # left hand speak
    now = time.monotonic()
    if left_hand_pred and left_hand_conf >= CONF_THRESHOLD:
      if left_hold_letter == left_hand_pred:
        if (now - left_hold_start) >= HOLD_SEC and left_hand_pred != last_left_spoken:
          speak(left_hand_pred)
          last_left_spoken = left_hand_pred
      else:
        left_hold_letter = left_hand_pred
        left_hold_start = now
    else:
      left_hold_letter = None
      left_hold_start = 0.0

    # right hand speak
    now = time.monotonic()
    if right_hand_pred and right_hand_conf >= CONF_THRESHOLD:
      if right_hold_letter == right_hand_pred:
        if (now - right_hold_start) >= HOLD_SEC and right_hand_pred != last_right_spoken:
          speak(right_hand_pred)
          last_right_spoken = right_hand_pred
      else:
        right_hold_letter = right_hand_pred
        right_hold_start = now
    else:
      right_hold_letter = None
      right_hold_start = 0.0
    
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
    
    cv2.imshow('ASL Interpreter', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()