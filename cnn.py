import json
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from pytubefix import YouTube
import tempfile

# Paths
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MS-ASL")
TRAIN_JSON = os.path.join(DATASET_DIR, "MSASL_train.json")
VAL_JSON = os.path.join(DATASET_DIR, "MSASL_val.json")
CLASSES_JSON = os.path.join(DATASET_DIR, "MSASL_classes.json")
CACHE_DIR = "landmark_cache"
MODEL_SAVE_PATH = "asl_landmark_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def load_json(path):
    # read/parse json file from path
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def download_frame(url, start_time):
    # download a short clip from a YouTube URL and grab a frame at start time
    # uses a temp file in CACHE_DIR which is removed after use
    try:
        # create temp directory for vids
        temp_dir = os.path.join(CACHE_DIR, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        video_id = url.split('watch?v=')[-1].split('&')[0]
        temp_path = os.path.join(temp_dir, f"{video_id}_{start_time}.mp4")
        
        # download video with retry - retry up to 3 times
        for attempt in range(3):
            try:
                yt = YouTube(url)
                stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
                if not stream:
                    print(f"No suitable stream found for {url}")
                    return None
                    
                # download vid
                stream.download(output_path=temp_dir, filename=f"{video_id}_{start_time}.mp4")
                
                # make sure file is there
                if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                    print(f"Download failed or empty file for {url}")
                    continue
                    
                # get frame
                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    print(f"Failed to open video file {temp_path}")
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
                ok, frame = cap.read()
                cap.release()
                
                if ok and frame is not None:
                    return frame
                print(f"Failed to read frame at {start_time}s")
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < 2:  # Don't sleep on last attempt
                    import time
                    time.sleep(2)  # Wait before retry
        
        return None
            
    finally:
        # Clean up temp file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            print(f"Failed to clean up temp file: {e}")
        
        # Try to clean up empty temp directory
        try:
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception:
            pass


def extract_landmarks(frame):
    # convert frame to landmarks
    # return a 21x3 numpy array of x,y,z coords
    if frame is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]


    landmarks = []
    for point in lm.landmark:
        coords = [point.x, point.y, point.z]
        landmarks.append(coords)
    
    return np.array(landmarks)


def cache_path(video_id, t):
    # get video at time
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{video_id}_{t:.3f}.npy")


def get_landmarks(url, start_time):
    # get video id
    vid = url.split('watch?v=')[-1].split('&')[0]
    cache = cache_path(vid, start_time)
    if os.path.exists(cache):
        return np.load(cache)
    frame = download_frame(url, start_time)
    lm = extract_landmarks(frame)
    if lm is not None:
        np.save(cache, lm)
    return lm


class ASLDataset(Dataset):
    def __init__(self, json_path, classes):
        # Load samples and build a mapping from class label to index
        self.samples = load_json(json_path)

        self.class_to_idx = {}
        for i, c in enumerate(classes):
            self.class_to_idx[c] = i

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        url = s['url']
        t = s['start_time']
        label = s['label']
    # get landmakrs for sample. if none, return 0 array
        lm = get_landmarks(url, t)
        if lm is None:
            x = torch.zeros(3, 21, dtype=torch.float32)
        else:
            x = torch.from_numpy(lm).float().T
        y = torch.tensor(self.class_to_idx.get(label, 0), dtype=torch.long)
        return x, y


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(model, loader, opt, crit):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        # Standard training step: move to device, forward, backward, step
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    # Return average batch loss
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        # Move batch to device and compute predictions for accuracy
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    # Return percentage accuracy (0-100) If loader had zero samples, return 0.0
    return 0.0 if total == 0 else 100.0 * correct / total


def main():
    classes = load_json(CLASSES_JSON)
    train_ds = ASLDataset(TRAIN_JSON, classes)
    val_ds = ASLDataset(VAL_JSON, classes)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = CNN(len(classes)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(5):
        loss = train_epoch(model, train_loader, opt, crit)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, val_acc={val_acc:.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()