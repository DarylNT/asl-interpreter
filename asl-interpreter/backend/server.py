import asyncio
import json
import websockets
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Import the model classes
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
        features = x
        for layer in self.net[:-2]:
            features = layer(features)
        return features

class TemporalPredictor:
    def __init__(self, model, sequence_length=10):
        self.model = model
        self.sequence_length = sequence_length
        self.probabilities_buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def predict(self, landmarks):
        x = torch.from_numpy(landmarks).float().T.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            probs = torch.softmax(output, dim=1)
        
        # Add to buffer for temporal averaging
        self.probabilities_buffer.append(probs.squeeze(0))
        
        if len(self.probabilities_buffer) > self.sequence_length:
            self.probabilities_buffer.pop(0)
        
        # Use temporal averaging when buffer is full
        if len(self.probabilities_buffer) == self.sequence_length:
            avg_probs = torch.stack(self.probabilities_buffer).mean(dim=0)
            confidence, pred_idx = avg_probs.max(dim=0)
            confidence = confidence.item() * 100
            return pred_idx.item(), confidence, True
        
        # Return single frame prediction
        confidence, pred_idx = probs.max(dim=1)
        confidence = confidence.item() * 100
        return pred_idx.item(), confidence, False

# Initialize model and predictor
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
letter_classes = [chr(65 + i) for i in range(26)]

model = CNN(26).to(DEVICE)
model_path = Path(__file__).parent / "asl_landmark_model.pt"
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
model.eval()

predictor = TemporalPredictor(model)

async def predict_handler(websocket, path):
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                landmarks = np.array(data["landmarks"], dtype=np.float32)
                
                # Get prediction
                pred_idx, confidence, is_temporal = predictor.predict(landmarks)
                predicted_letter = letter_classes[pred_idx]
                
                # Send response
                response = {
                    "letter": predicted_letter,
                    "confidence": confidence,
                    "temporal_mode": is_temporal
                }
                await websocket.send(json.dumps(response))
                
            except Exception as e:
                print(f"Error processing prediction: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        pass

async def main():
    server = await websockets.serve(
        predict_handler,
        "127.0.0.1",
        8000,
        ping_interval=None
    )
    print("ASL Prediction Server running on ws://127.0.0.1:8000")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())