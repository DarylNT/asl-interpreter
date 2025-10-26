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
    
    def extract_features(self, x):
        """Extract features before final classification"""
        # Get features up to the second-to-last layer
        features = x
        for layer in self.net[:-2]:  # Exclude last two layers (Linear + Linear)
            features = layer(features)
        return features


class EnhancedCNNRNN(nn.Module):
    """CNN-RNN model that combines CNN features with probabilities for temporal prediction"""
    def __init__(self, cnn_model, feature_size=128, prob_size=26, hidden_size=64, num_classes=26):
        super().__init__()
        self.cnn = cnn_model
        self.feature_extractor = nn.Sequential(
            nn.Linear(128, feature_size),  # Process CNN features
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.rnn = nn.LSTM(feature_size + prob_size, hidden_size, batch_first=True, dropout=0.3, num_layers=2)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, landmarks_sequence, prob_sequence):
        """
        Args:
            landmarks_sequence: (batch, seq_len, 3, 21) - sequence of landmarks
            prob_sequence: (batch, seq_len, 26) - sequence of CNN probabilities
        """
        batch_size, seq_len = landmarks_sequence.shape[0], landmarks_sequence.shape[1]
        features = []
        
        for i in range(seq_len):
            # Get CNN features (before final classification)
            cnn_features = self.cnn.extract_features(landmarks_sequence[:, i, :, :])
            processed_features = self.feature_extractor(cnn_features)
            
            # Combine CNN features with probabilities
            combined = torch.cat([processed_features, prob_sequence[:, i, :]], dim=1)
            features.append(combined)
        
        # Stack and process with RNN
        features = torch.stack(features, dim=1)  # (batch, seq_len, feature_size + prob_size)
        rnn_out, _ = self.rnn(features)
        
        # Use last output for classification
        last_output = rnn_out[:, -1, :]
        last_output = self.dropout(last_output)
        return self.classifier(last_output)


# get cnn model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(26).to(DEVICE)
model.load_state_dict(torch.load("asl_landmark_model.pt", map_location=DEVICE))
model.eval()

# a-z
letter_classes = [chr(65 + i) for i in range(26)]


class TemporalASLPredictor:
    """Real-time ASL predictor using CNN-RNN temporal enhancement"""
    def __init__(self, cnn_model, rnn_model, sequence_length=10):
        self.cnn_model = cnn_model
        self.rnn_model = rnn_model
        self.sequence_length = sequence_length
        self.landmarks_buffer = []
        self.probabilities_buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def predict_with_temporal_context(self, landmarks):
        """Predict letter using temporal context from CNN-RNN"""
        # Normalize landmarks
        landmarks_normalized = (landmarks - landmarks.mean(axis=0)) / (landmarks.std(axis=0) + 1e-8)
        
        # Get CNN prediction
        x = torch.from_numpy(landmarks_normalized).float().T.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cnn_output = self.cnn_model(x)
            cnn_probs = torch.softmax(cnn_output, dim=1)
        
        # Add to buffers
        self.landmarks_buffer.append(x.squeeze(0))  # Remove batch dim
        self.probabilities_buffer.append(cnn_probs.squeeze(0))
        
        # Keep only recent frames
        if len(self.landmarks_buffer) > self.sequence_length:
            self.landmarks_buffer.pop(0)
            self.probabilities_buffer.pop(0)
        
        # Use RNN when buffer is full (only if RNN model is trained)
        # For now, just use CNN with temporal averaging for better stability
        if len(self.landmarks_buffer) == self.sequence_length and self.rnn_model is not None:
            # Stack sequences
            landmarks_seq = torch.stack(self.landmarks_buffer).unsqueeze(0)  # (1, seq_len, 3, 21)
            probs_seq = torch.stack(self.probabilities_buffer).unsqueeze(0)   # (1, seq_len, 26)
            
            try:
                with torch.no_grad():
                    rnn_output = self.rnn_model(landmarks_seq, probs_seq)
                    rnn_probs = torch.softmax(rnn_output, dim=1)
                    
                    # Get enhanced prediction
                    enhanced_confidence, pred_idx = rnn_probs.max(dim=1)
                    enhanced_confidence = enhanced_confidence.item() * 100
                    predicted_letter = letter_classes[pred_idx.item()]
                    
                    return predicted_letter, enhanced_confidence
            except:
                # Fall back to temporal averaging if RNN fails
                pass
        
        # Use temporal averaging for stability when buffer is full
        if len(self.probabilities_buffer) == self.sequence_length:
            avg_probs = torch.stack(self.probabilities_buffer).mean(dim=0)
            confidence, pred_idx = avg_probs.max(dim=0)
            confidence = confidence.item() * 100
            predicted_letter = letter_classes[pred_idx.item()]
            return predicted_letter, confidence
        
        # Return CNN prediction for first few frames
        confidence, pred_idx = cnn_probs.max(dim=1)
        confidence = confidence.item() * 100
        predicted_letter = letter_classes[pred_idx.item()]
        return predicted_letter, confidence


# Initialize RNN model as None (not trained yet - would need to train this separately)
# Set it to None to use temporal averaging instead of RNN
rnn_model = None
# If you have a trained RNN model, uncomment the lines below:
# rnn_model = EnhancedCNNRNN(model).to(DEVICE)
# rnn_model.load_state_dict(torch.load("rnn_model.pt", map_location=DEVICE))
# rnn_model.eval()

# Initialize temporal predictor
temporal_predictor = TemporalASLPredictor(model, rnn_model, sequence_length=10)


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
        # draw a rectangle around the hand (bounding box from 2D landmarks)
        h, w = image.shape[:2]
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        pad = 10
        x_min = max(0, int(min(xs)) - pad)
        y_min = max(0, int(min(ys)) - pad)
        x_max = min(w - 1, int(max(xs)) + pad)
        y_max = min(h - 1, int(max(ys)) + pad)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # world landmarks for it to be actually useful
        if results.multi_hand_world_landmarks:
          hand_world_landmarks = results.multi_hand_world_landmarks[hand_idx]
          landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_world_landmarks.landmark], dtype=np.float32)
          
          # Use temporal predictor for enhanced accuracy
          predicted_letter, confidence = temporal_predictor.predict_with_temporal_context(landmarks)
    
    # make it easier to see urself
    image = cv2.flip(image, 1)
    
    # show prediced letter
    if predicted_letter:
      cv2.putText(image, f"Letter: {predicted_letter}", (10, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
      cv2.putText(image, f"Confidence: {confidence:.1f}%", (10, 100), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
      
      # Show if using temporal enhancement
      if len(temporal_predictor.landmarks_buffer) == temporal_predictor.sequence_length:
          cv2.putText(image, "Temporal Enhanced", (10, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.imshow('Enhanced ASL Recognition', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()