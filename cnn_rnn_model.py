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
import csv

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
MODEL_PATH = "asl_cnn_rnn_model.pt"

# try different devices to run it on
try:
    import torch_directml
    DEVICE = torch_directml.device()
except:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


class CNNFeatureExtractor(nn.Module):
    """CNN that extracts features from hand landmarks"""
    def __init__(self, feature_size=128):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(3, 64, 3, padding=1), 
            nn.BatchNorm1d(64), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=1), 
            nn.BatchNorm1d(128), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Conv1d(128, 256, 3, padding=1), 
            nn.BatchNorm1d(256), 
            nn.ReLU(), 
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten(),
            nn.Linear(256, feature_size), 
            nn.ReLU(), 
            nn.Dropout(0.4)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 3, 21) - landmarks
        return self.feature_extractor(x)  # Output: (batch_size, feature_size)


class CNNRNNModel(nn.Module):
    """Combined CNN-RNN model for ASL sequence classification"""
    def __init__(self, cnn_feature_size=128, rnn_hidden_size=64, num_classes=26, num_layers=2):
        super().__init__()
        self.cnn = CNNFeatureExtractor(cnn_feature_size)
        self.rnn = nn.LSTM(cnn_feature_size, rnn_hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Linear(rnn_hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, 3, 21)
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Process each frame through CNN
        cnn_features = []
        for i in range(seq_len):
            frame_features = self.cnn(x[:, i, :, :])  # (batch_size, cnn_feature_size)
            cnn_features.append(frame_features)
        
        # Stack features: (batch_size, seq_len, cnn_feature_size)
        cnn_features = torch.stack(cnn_features, dim=1)
        
        # Process sequence through RNN
        rnn_out, (hidden, cell) = self.rnn(cnn_features)
        
        # Use last output for classification
        last_output = rnn_out[:, -1, :]  # (batch_size, rnn_hidden_size)
        last_output = self.dropout(last_output)
        return self.classifier(last_output)


class SequentialRoboFlowDataset(Dataset):
    """Dataset that creates sequences of hand landmarks for RNN training"""
    def __init__(self, image_dir, csv_path, sequence_length=10):
        self.image_dir = image_dir
        self.sequence_length = sequence_length
        self.classes = [chr(65 + i) for i in range(26)]  # A-Z
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []  # (image_path, label_idx)
        
        # Load csv files to map letters to images
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                filename = row.get('filename', '').strip()
                if not filename:
                    continue
                img_path = os.path.join(image_dir, filename)
                if not os.path.exists(img_path):
                    continue
                
                # Get letter from csv
                for letter in self.classes:
                    if row.get(f' {letter}', '').strip() == '1':
                        self.samples.append((img_path, self.class_to_idx[letter]))
                        break
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self):
        """Group images by letter to create sequences"""
        sequences = []
        letter_groups = {}
        
        # Group images by letter
        for img_path, label in self.samples:
            if label not in letter_groups:
                letter_groups[label] = []
            letter_groups[label].append(img_path)
        
        # Create sequences of the same letter
        for label, img_paths in letter_groups.items():
            # Shuffle images for each letter to create varied sequences
            np.random.shuffle(img_paths)
            
            # Create sequences
            for i in range(0, len(img_paths) - self.sequence_length + 1, self.sequence_length // 2):
                sequence = img_paths[i:i + self.sequence_length]
                if len(sequence) == self.sequence_length:
                    sequences.append((sequence, label))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence_paths, label = self.sequences[idx]
        sequence_landmarks = []
        
        for img_path in sequence_paths:
            cache_file = self._get_cache_path(img_path)
            
            # Load or extract landmarks
            if os.path.exists(cache_file):
                lm = np.load(cache_file)
            else:
                # Extract landmarks from image
                image = cv2.imread(img_path)
                if image is None:
                    return None
                
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                
                if not results.multi_hand_world_landmarks:
                    return None
                
                lm = np.array([[p.x, p.y, p.z] for p in results.multi_hand_world_landmarks[0].landmark], dtype=np.float32)
                
                # Save to cache
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                np.save(cache_file, lm)
            
            # Normalize landmarks
            lm = (lm - lm.mean(axis=0)) / (lm.std(axis=0) + 1e-8)
            sequence_landmarks.append(lm)
        
        # Return sequence: (seq_len, 3, 21)
        sequence_tensor = torch.stack([torch.from_numpy(lm).float().T for lm in sequence_landmarks])
        return sequence_tensor, torch.tensor(label, dtype=torch.long)
    
    def _get_cache_path(self, img_path):
        subdir = 'train' if 'train' in self.image_dir else 'valid' if 'valid' in self.image_dir else 'other'
        cache_filename = os.path.splitext(os.path.basename(img_path))[0] + '.npy'
        return os.path.join(CACHE_DIR, subdir, cache_filename)


def collate_skip_none(batch):
    """Collate function that skips None samples"""
    filtered = [b for b in batch if b is not None]
    if len(filtered) == 0:
        return torch.empty(0, 10, 3, 21), torch.empty(0, dtype=torch.long)  # Default seq_len=10
    xs, ys = zip(*filtered)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def train_epoch(model, loader, opt, crit):
    """Train one epoch of the CNN-RNN model"""
    model.train()
    total_loss = 0.0
    class_stats = {}
    
    for i, (x, y) in enumerate(loader, start=1):
        # Skip empty batches
        if x.numel() == 0:
            continue
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        
        # Forward pass
        out = model(x)
        loss = crit(out, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for RNN stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        opt.step()
        total_loss += loss.item()
        
        # Track per-class statistics
        with torch.no_grad():
            per_sample_loss = F.cross_entropy(out, y, reduction='none').detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            
            for cls in np.unique(y_cpu):
                mask = (y_cpu == cls)
                values = per_sample_loss[mask]
                
                if int(cls) not in class_stats:
                    class_stats[int(cls)] = {"count": 0, "mean": 0.0, "M2": 0.0}
                
                st = class_stats[int(cls)]
                for v in values:
                    st_count = st["count"] + 1
                    delta = v - st["mean"]
                    mean = st["mean"] + delta / st_count
                    delta2 = v - mean
                    M2 = st["M2"] + delta * delta2
                    st["count"], st["mean"], st["M2"] = st_count, mean, M2
        
        # Display progress
        if i % 10 == 0:  # Print every 10 batches
            print(f"Batch {i}/{len(loader)} - loss={loss.item():.4f}")
    
    return total_loss / len(loader), class_stats


@torch.no_grad()
def evaluate(model, loader):
    """Evaluate the model"""
    model.eval()
    correct, total = 0, 0
    
    for x, y in loader:
        if x.numel() == 0:
            continue
        
        x, y = x.to(DEVICE), y.to(DEVICE)
        outputs = model(x)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0


def main():
    """Main training function"""
    print(f"Using device: {DEVICE}")
    
    # Create datasets with sequences
    sequence_length = 10  # Number of frames per sequence
    train_ds = SequentialRoboFlowDataset(TRAIN_DIR, TRAIN_CSV, sequence_length)
    val_ds = SequentialRoboFlowDataset(VAL_DIR, VAL_CSV, sequence_length)
    
    print(f"Training sequences: {len(train_ds)}")
    print(f"Validation sequences: {len(val_ds)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=8,  # Smaller batch size for sequences
        shuffle=True, 
        num_workers=2, 
        pin_memory=True, 
        collate_fn=collate_skip_none
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=16, 
        num_workers=2, 
        pin_memory=True, 
        collate_fn=collate_skip_none
    )
    
    # Create model
    model = CNNRNNModel(
        cnn_feature_size=128,
        rnn_hidden_size=64,
        num_classes=26,
        num_layers=2
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    epochs = 15
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train
        train_loss, class_stats = train_epoch(model, train_loader, optimizer, criterion)
        
        # Validate
        val_acc = evaluate(model, val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Val Accuracy: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"New best model saved! Accuracy: {val_acc:.2f}%")
        
        # Save class statistics
        os.makedirs(CACHE_DIR, exist_ok=True)
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
        
        stats_path = os.path.join(CACHE_DIR, f"cnn_rnn_class_stats_epoch_{epoch+1}.json")
        with open(stats_path, 'w') as f:
            json.dump(export, f, indent=2)
    
    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
