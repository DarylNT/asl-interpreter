import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp

# tensorflow is annoying me
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# dirs needed
ROBOFLOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RoboFlow")
TRAIN_DIR = os.path.join(ROBOFLOW_DIR, "train")
VAL_DIR = os.path.join(ROBOFLOW_DIR, "valid")
TRAIN_CSV = os.path.join(TRAIN_DIR, "_classes.csv")
VAL_CSV = os.path.join(VAL_DIR, "_classes.csv")
CACHE_DIR = "roboflow_cache"
MODEL_PATH = "asl_landmark_model.pt"

# try different devices to run it on
try:
    import torch_directml
    DEVICE = torch_directml.device()
except:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe Hands will be created lazily inside the dataset workers only if needed


class RoboFlowDataset(Dataset):
    def __init__(self, image_dir, csv_path):
        import csv
        self.image_dir = image_dir
        self.classes = [chr(65 + i) for i in range(26)]  # a-z
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []  # (image_path, label_idx)
        self.hands = None  # lazy init to avoid creating TFLite interpreter in every worker unnecessarily
        
        # load csv files to map letters to imgs
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                filename = row.get('filename', '').strip()
                if not filename:
                    continue
                img_path = os.path.join(image_dir, filename)
                if not os.path.exists(img_path):
                    continue
                
                # get letter from csv
                for letter in self.classes:
                    if row.get(f' {letter}', '').strip() == '1':
                        self.samples.append((img_path, self.class_to_idx[letter]))
                        break
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        cache_file = self._get_cache_path(img_path)
        
        # if already in cache
        if os.path.exists(cache_file):
            lm = np.load(cache_file)
        else:
            # get landmarks from img (only when cache is missing)
            image = cv2.imread(img_path)
            if image is None:
                return None
            
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # create mediapipe hands on first use in this worker
            if self.hands is None:
                mp_hands = mp.solutions.hands
                self.hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
            results = self.hands.process(rgb)
            
            if not results.multi_hand_world_landmarks:
                return None
            
            lm = np.array([[p.x, p.y, p.z] for p in results.multi_hand_world_landmarks[0].landmark], dtype=np.float32)
            # save to cache
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            np.save(cache_file, lm)
        
        # mean stuff
        lm = (lm - lm.mean(axis=0)) / (lm.std(axis=0) + 1e-8)
        return torch.from_numpy(lm).float().T, torch.tensor(label, dtype=torch.long)
    
    def _get_cache_path(self, img_path):
        subdir = 'train' if 'train' in self.image_dir else 'valid' if 'valid' in self.image_dir else 'other'
        cache_filename = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        return os.path.join(CACHE_DIR, subdir, cache_filename)


def collate_skip_none(batch):
    filtered = [b for b in batch if b is not None]
    if len(filtered) == 0:
        return torch.empty(0, 3, 21), torch.empty(0, dtype=torch.long)
    xs, ys = zip(*filtered)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # continuously increase accuracy stuff
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


def train_epoch(model, loader, opt, crit):
    model.train()
    total_loss = 0.0
    # Welford: {class_idx: {count, mean, M2}}
    class_stats = {}
    
    for i, (x, y) in enumerate(loader, start=1):
        # empty batch
        if x.numel() == 0:
            continue
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        out = model(x)
        # get loss
        loss = crit(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        
        # losses per sample for welford
        with torch.no_grad():
            per_sample_loss = F.cross_entropy(out, y, reduction='none').detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            # update welford
            for cls in np.unique(y_cpu):
                mask = (y_cpu == cls)
                values = per_sample_loss[mask]
                # initialzie
                if int(cls) not in class_stats:
                    class_stats[int(cls)] = {"count": 0, "mean": 0.0, "M2": 0.0}
                st = class_stats[int(cls)]
                # update vals
                for v in values:
                    st_count = st["count"] + 1
                    delta = v - st["mean"]
                    mean = st["mean"] + delta / st_count
                    delta2 = v - mean
                    M2 = st["M2"] + delta * delta2
                    st["count"], st["mean"], st["M2"] = st_count, mean, M2
        
        # display progress after each batch
        print(f"Batch {i}/{len(loader)} - loss={loss.item():.4f}")
    return total_loss / len(loader), class_stats


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        if x.numel() == 0:
            continue
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def main():
    # define datasets
    train_ds = RoboFlowDataset(TRAIN_DIR, TRAIN_CSV)
    val_ds = RoboFlowDataset(VAL_DIR, VAL_CSV)
    
    # parallel processing
    # Keep workers alive across epochs so heavy initializations (like MediaPipe/TFLite) don't repeat each epoch
    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True, num_workers=16, pin_memory=True,
        collate_fn=collate_skip_none, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, num_workers=16, pin_memory=True,
        collate_fn=collate_skip_none, persistent_workers=True
    )

    model = CNN(26).to(DEVICE)  # a-z
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=2)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(100): # more epochs means more training
        loss, class_stats = train_epoch(model, train_loader, opt, crit)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, val_acc={val_acc:.2f}%")
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
        
        # welford stats
        os.makedirs(CACHE_DIR, exist_ok=True)
        # convert welford to actual stats (count, mean, variance)
        export = {}
        letter_classes = [chr(65 + i) for i in range(26)]
        for cls_idx, st in class_stats.items():
            count = st.get("count", 0)
            mean = st.get("mean", 0.0)
            M2 = st.get("M2", 0.0)
            variance = (M2 / (count - 1)) if count > 1 else 0.0
            cls_name = letter_classes[int(cls_idx)] if 0 <= int(cls_idx) < 26 else None
            export[str(cls_idx)] = {
                "class_name": cls_name,
                "count": int(count),
                "mean_loss": float(mean),
                "variance_loss": float(variance)
            }
        stats_path = os.path.join(CACHE_DIR, f"class_loss_stats_epoch_{epoch+1}.json")
        with open(stats_path, 'w') as f:
            json.dump(export, f, indent=2)
        print(f"Saved stats to {stats_path}")


if __name__ == '__main__':
    main()