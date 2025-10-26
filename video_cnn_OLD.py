import json
import os
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp
from pytubefix import YouTube
import tempfile
import argparse

# Paths
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MS-ASL")
TRAIN_JSON = os.path.join(DATASET_DIR, "MSASL_train.json")
VAL_JSON = os.path.join(DATASET_DIR, "MSASL_val.json")
CLASSES_JSON = os.path.join(DATASET_DIR, "MSASL_classes.json")
CACHE_DIR = "landmark_cache"
MODEL_SAVE_PATH = "asl_landmark_model.pt"

# try different devices to run it on
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print("Using DirectML (AMD/Intel GPU)")
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
    else:
        print("Using CPU (no GPU acceleration)")
except Exception:
    DEVICE = torch.device("cpu")
    print("Using CPU (GPU initialization failed)")

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def load_json(path):
    # read/parse json file from path
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def download_video(url):
    temp_dir = os.path.join(CACHE_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    video_id = url.split('watch?v=')[-1].split('&')[0]
    temp_path = os.path.join(temp_dir, f"{video_id}.mp4")
    
    try:
        yt = YouTube(url)
        # sort by resolution
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').first()
        if not stream:
            print(f"No suitable stream found for {url}")
            return None, None
            
        # download vid
        stream.download(output_path=temp_dir, filename=f"{video_id}.mp4")
        
        # make sure file is there
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print(f"Download failed or empty file for {url}")
            return None, None
            
        return temp_path, video_id
    except Exception as e:
        print(f"Download failed: {str(e)}")
        return None, None


def extract_frames_from_video(video_path, times):
    frames = {}
    if len(times) == 0:
        return frames
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video file {video_path}")
            return frames
        
        sorted_times = sorted(times)
        
        for t in sorted_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames[t] = frame
            else:
                frames[t] = None
                
        cap.release()
    except Exception as e:
        print(f"Frame extraction failed: {str(e)}")
    
    return frames


def extract_landmarks_batch(frames_dict):
    results = {}
    if not frames_dict:
        return results
    
    for t, frame in frames_dict.items():
        # made me want to kms wouldnt stop erroring
        if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
            results[t] = None
            continue
            
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            
            if not res.multi_hand_world_landmarks:
                results[t] = None
                continue
                
            lm = res.multi_hand_world_landmarks[0]
            landmarks = []
            for point in lm.landmark:
                coords = [point.x, point.y, point.z]
                landmarks.append(coords)
            
            results[t] = np.array(landmarks)
        except Exception:
            results[t] = None
    
    return results


def extract_landmarks(frame):
    # convert frame to landmarks
    # return a 21x3 numpy array of x,y,z coords
    if frame is None or (isinstance(frame, np.ndarray) and frame.size == 0):
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_world_landmarks:
        return None
    lm = res.multi_hand_world_landmarks[0]


    landmarks = []
    for point in lm.landmark:
        coords = [point.x, point.y, point.z]
        landmarks.append(coords)
    
    return np.array(landmarks)


def cache_path(video_id, start_time, end_time=None):
    os.makedirs(CACHE_DIR, exist_ok=True)
    if end_time is None or end_time <= start_time:
        return os.path.join(CACHE_DIR, f"{video_id}_{start_time:.3f}.npy")
    else:
        return os.path.join(CACHE_DIR, f"{video_id}_{start_time:.3f}_{end_time:.3f}.npy")


def get_landmarks(url, start_time, end_time=None, step=1/30.0):
    vid = url.split('watch?v=')[-1].split('&')[0]

    # one frame
    if end_time is None or end_time <= start_time:
        cache = cache_path(vid, start_time)
        if os.path.exists(cache):
            return np.load(cache)
        
        # download video once
        temp_path, _ = download_video(url)
        if temp_path is None:
            # download failed, remove from dataset
            for json_path in [TRAIN_JSON, VAL_JSON]:
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    new_data = [item for item in data if item.get('url') != url]
                    if len(new_data) != len(data):
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(new_data, f, ensure_ascii=False, indent=2)
                        print(f"Removed unavailable video from {json_path}: {url}")
                except Exception as e:
                    print(f"Failed to update {json_path}: {e}")
            return None
        
        try:
            # extract single frame
            frames = extract_frames_from_video(temp_path, [start_time])
            frame = frames.get(start_time)
            if frame is None:
                return None
            lm = extract_landmarks(frame)
            if lm is not None:
                np.save(cache, lm)
            return lm
        finally:
            # cleanup video file
            try:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                temp_dir = os.path.dirname(temp_path)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                print(f"Failed to clean up temp file: {e}")

    # store all frames in one numpy file
    cache = cache_path(vid, start_time, end_time)
    if os.path.exists(cache):
        try:
            stacked = np.load(cache)  # shape (F, 21, 3)
            if len(stacked) > 0:
                return stacked 
        except Exception as e:
            print(f"Failed to load cache {cache}: {e}")
    
    # need to extract frames
    times = np.arange(start_time, end_time + 1e-9, step)
    temp_path, _ = download_video(url)
    if temp_path is None:
        # download failed, remove sample
        for json_path in [TRAIN_JSON, VAL_JSON]:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                new_data = [item for item in data if item.get('url') != url]
                if len(new_data) != len(data):
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(new_data, f, ensure_ascii=False, indent=2)
                    print(f"Removed unavailable video from {json_path}: {url}")
            except Exception as e:
                print(f"Failed to update {json_path}: {e}")
        return None
    
    try:
        # extract all frames at once
        frames = extract_frames_from_video(temp_path, times)
        
        # batch process landmarks for all frames at once
        landmarks_dict = extract_landmarks_batch(frames)
        
        collected = []
        for t in times:
            lm = landmarks_dict.get(t)
            if lm is not None:
                collected.append(lm)
        
        if not collected:
            # remove sample from dataset if nth there (backup)
            for json_path in [TRAIN_JSON, VAL_JSON]:
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    new_data = [item for item in data if item.get('url') != url]
                    if len(new_data) != len(data):
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(new_data, f, ensure_ascii=False, indent=2)
                        print(f"Removed unavailable video from {json_path}: {url}")
                except Exception as e:
                    print(f"Failed to update {json_path}: {e}")
            return None
        
        # save all frames together in one file
        stacked = np.stack(collected, axis=0)  # (F, 21, 3)
        try:
            np.save(cache, stacked)
        except Exception as e:
            print(f"Failed to save cache {cache}: {e}")
        
        return stacked
        
    finally:
        # get rid of video stuff
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            temp_dir = os.path.dirname(temp_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Failed to clean up temp file: {e}")

class ASLDataset(Dataset):
    def __init__(self, json_path, classes):
        # map class to index
        self.samples = load_json(json_path)[:20]

        self.class_to_idx = {}
        for i, c in enumerate(classes):
            self.class_to_idx[c] = i

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        url = s['url']
        start_time = s.get('start_time', 0.0)
        end_time = s.get('end_time', None)
        label = s.get('label')
    # get landmarks for sample. if none, return 0 array
        lm = get_landmarks(url, start_time, end_time)
        if lm is None:
            x = torch.zeros(3, 21, dtype=torch.float32)
        else:
            # if single frame or multiple (aka start+endtmie)
            if lm.ndim == 2:  # single frame (21, 3)
                x = torch.from_numpy(lm).float().T
            else:
                averaged = lm.mean(axis=0)
                x = torch.from_numpy(averaged).float().T
        if isinstance(label, int):
            y_idx = int(label)
        else:
            y_idx = self.class_to_idx.get(label, 0)
        y = torch.tensor(y_idx, dtype=torch.long)
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
    total_batches = len(loader)
    # Per-class Welford state: {class_idx: {count, mean, M2}}
    class_stats = {}

    for i, (x, y) in enumerate(loader, start=1):
        # Skip empty batches (possible when all samples in a batch are filtered out)
        if x.numel() == 0 or y.numel() == 0:
            print(f"Batch {i}/{total_batches} - skipped empty batch")
            continue
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        # Compute loss for backward (mean reduction)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        batch_loss = loss.item()
        total_loss += batch_loss

        # Per-sample losses for Welford (no reduction)
        with torch.no_grad():
            per_sample_loss = F.cross_entropy(out, y, reduction='none').detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            # Update Welford for each class present in this batch
            for cls in np.unique(y_cpu):
                mask = (y_cpu == cls)
                values = per_sample_loss[mask]
                # Initialize if needed
                if int(cls) not in class_stats:
                    class_stats[int(cls)] = {"count": 0, "mean": 0.0, "M2": 0.0}
                st = class_stats[int(cls)]
                # Update sequentially
                for v in values:
                    st_count = st["count"] + 1
                    delta = v - st["mean"]
                    mean = st["mean"] + delta / st_count
                    delta2 = v - mean
                    M2 = st["M2"] + delta * delta2
                    st["count"], st["mean"], st["M2"] = st_count, mean, M2

        # display progress after each batch
        print(f"Batch {i}/{total_batches} - batch_loss={batch_loss:.4f} - avg_loss={(total_loss/i):.4f}")
    # avg batch loss
    return total_loss / len(loader), class_stats


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        if x.numel() == 0 or y.numel() == 0:
            continue
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    # percent accuracy
    return 0.0 if total == 0 else 100.0 * correct / total


def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    classes = load_json(CLASSES_JSON)
    train_ds = ASLDataset(TRAIN_JSON, classes)
    val_ds = ASLDataset(VAL_JSON, classes)
    
    # parallel processing
    num_workers = 4  # Set to 0 if you have issues on Windows
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=num_workers, pin_memory=True)

    model = CNN(len(classes)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    best = 0.0
    for epoch in range(5):
        loss, class_stats = train_epoch(model, train_loader, opt, crit)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, val_acc={val_acc:.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # more welford ugh
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            # welford to count/mean/variance
            export = {}
            for cls_idx, st in class_stats.items():
                count = st.get("count", 0)
                mean = st.get("mean", 0.0)
                M2 = st.get("M2", 0.0)
                variance = (M2 / (count - 1)) if count > 1 else 0.0
                # Try to include class name if available
                cls_name = None
                try:
                    if 0 <= int(cls_idx) < len(classes):
                        cls_name = classes[int(cls_idx)]
                except Exception:
                    cls_name = None
                export[str(cls_idx)] = {
                    "class_name": cls_name,
                    "count": int(count),
                    "mean_loss": float(mean),
                    "variance_loss": float(variance)
                }
            stats_path = os.path.join(CACHE_DIR, f"class_loss_stats_epoch_{epoch+1}.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(export, f, indent=2, ensure_ascii=False)
            print(f"Saved per-class loss stats to {stats_path}")
        except Exception as e:
            print(f"Failed to save class stats: {e}")
        # reload dataset after each epoch so any unavailable videos removed from the JSON
        train_ds = ASLDataset(TRAIN_JSON, classes)
        val_ds = ASLDataset(VAL_JSON, classes)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, num_workers=num_workers, pin_memory=True)
        print(f"After epoch {epoch+1}: reloaded {len(train_ds)} training samples and {len(val_ds)} validation samples")


if __name__ == '__main__':
    main()