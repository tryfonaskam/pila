import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------
# CONFIG
# --------------------
DATASET_DIR = "datasets"
IMAGE_SIZE = (512, 512)

BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-4

SAVE_EVERY = 2  # epochs
CHECKPOINT_DIR = "checkpoints"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

# --------------------
# DATASET
# --------------------
class ImitationDataset(Dataset):
    def __init__(self, dataset_dir):
        self.samples = []

        runs = sorted([
            d for d in os.listdir(dataset_dir)
            if d.startswith("run") and os.path.isdir(os.path.join(dataset_dir, d))
        ])

        if not runs:
            raise RuntimeError("No valid run directories found")

        print(f"[INFO] Found runs: {runs}")

        for run in runs:
            run_path = Path(dataset_dir) / run
            frames_dir = run_path / "frames"
            actions_path = run_path / "actions.csv"

            if not frames_dir.exists() or not actions_path.exists():
                continue

            actions = pd.read_csv(actions_path)
            frame_files = sorted(frames_dir.glob("*.png"))

            assert len(frame_files) == len(actions), \
                f"Frame/action mismatch in {run}"

            for i in range(len(actions)):
                self.samples.append(
                    (frame_files[i], actions.iloc[i].values.astype(np.float32))
                )

        print(f"[INFO] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_path, action = self.samples[idx]

        img = cv2.imread(str(frame_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ---- FORCE 512x512 ----
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32)
        )

# --------------------
# MODEL
# --------------------
class CNNPolicy(nn.Module):
    def __init__(self, conv_out_size, num_actions=4):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# --------------------
# TRAINING
# --------------------
def train():
    dataset = ImitationDataset(DATASET_DIR)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # ---- infer conv output size dynamically ----
    with torch.no_grad():
        dummy_img, _ = dataset[0]
        dummy_img = dummy_img.unsqueeze(0)

        conv_probe = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
        )

        conv_out = conv_probe(dummy_img)
        conv_out_size = conv_out.view(1, -1).shape[1]

    model = CNNPolicy(conv_out_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("[INFO] Starting training")

    for epoch in range(EPOCHS):
        print(f"[INFO] Epoch {epoch+1} started")
        total_loss = 0.0

        for imgs, actions in loader:
            imgs = imgs.to(DEVICE)
            actions = actions.to(DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[EPOCH {epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = f"{CHECKPOINT_DIR}/pila_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    final_path = f"{CHECKPOINT_DIR}/pila_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"[INFO] Final model saved to {final_path}")

# --------------------
if __name__ == "__main__":
    train()
