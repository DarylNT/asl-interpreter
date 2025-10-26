import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

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


cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
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
    
    predicted_letter = ""
    confidence = 0.0
    # normal landmarks to draw on hand
    if results.multi_hand_landmarks:
      for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
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
    
    # make it easier to see urself
    image = cv2.flip(image, 1)
    
    # show prediced letter
    if predicted_letter:
      cv2.putText(image, f"Letter: {predicted_letter}", (10, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
      cv2.putText(image, f"Confidence: {confidence:.1f}%", (10, 100), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('ASL Recognition', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()