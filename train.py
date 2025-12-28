import os
import csv
import cv2
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# config start
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DATA_DIR = "dataset"          # contains run01, run02, ...
IMG_SIZE = 256                # image size (square)
FRAME_STACK = 4               # number of frames stacked
BATCH_SIZE = 128              # training batch size
EPOCHS = 5                    # total training epochs
LR = 1e-4                     # learning rate

# Continuous attention
BASE_WEIGHT = 0.4             # minimum weight for zero frames
DYNAMIC_WEIGHT = 0.1          # how much nonzero frames add (0–1)
ATTN_SCALE = 0.1              # sensitivity

SAVE_EVERY = 1                # epochs to save checkpoints

MODEL_DIR = "models"          # model folder directory
MODEL_PATH = os.path.join(MODEL_DIR, "control_model.pth")         # model save path

# config end

writer = SummaryWriter("runs")

def find_runs(root):
    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if d.startswith("run") and os.path.isdir(os.path.join(root, d))
    ]


class ControlDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        runs = find_runs(root)
        if not runs:
            raise RuntimeError("No runXX folders found")

        for run in runs:
            frames_dir = os.path.join(run, "frames")
            csv_path = os.path.join(run, "actions.csv")
            if not os.path.exists(frames_dir) or not os.path.exists(csv_path):
                continue

            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))

            for i in range(FRAME_STACK - 1, len(rows)):
                self.samples.append((frames_dir, rows, i))

        if not self.samples:
            raise RuntimeError("Dataset is empty")

        print(f"Dataset samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_dir, rows, i = self.samples[idx]

        imgs = []
        for j in range(i - FRAME_STACK + 1, i + 1):
            img = cv2.imread(os.path.join(frames_dir, rows[j]["frame"]))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img.astype(np.float32) / 255.0
            imgs.append(np.transpose(img, (2, 0, 1)))

        x = np.concatenate(imgs, axis=0)  # 12×256×256

        y = np.array([
            float(rows[i]["w_s"]),
            float(rows[i]["a_d"]),
            float(rows[i]["q_e"]),
            float(rows[i]["shift_ctrl"]),
            float(rows[i]["mouse_dx"]),
            float(rows[i]["mouse_dy"]),
        ], dtype=np.float32)

        return torch.tensor(x), torch.tensor(y)


class ControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(12, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x.flatten(1))


# nonzero attention computation

def compute_attention(actions):
    mouse_mag = torch.norm(actions[:, 4:6], dim=1)
    key_mag = torch.norm(actions[:, :4], dim=1)

    mag = 1.5 * mouse_mag + 1.0 * key_mag
    attn = torch.tanh(mag * ATTN_SCALE)  # range 0–1

    return attn


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    dataset = ControlDataset(DATA_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      # Windows safe
        pin_memory=True,
        drop_last=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ControlNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
        for imgs, actions in pbar:
            imgs = imgs.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)

            pred = model(imgs)

            base_loss = ((pred - actions) ** 2).mean(dim=1)
            attn = compute_attention(actions)

            loss = (base_loss * (BASE_WEIGHT + DYNAMIC_WEIGHT * attn)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1))


        # tensorboard logging
        writer.add_scalar("train/loss", loss, epoch)

        writer.add_scalar(
            "train/mouse_mag",
            actions[:, 4:6].norm(dim=1).mean().item(),
            epoch
        )

        writer.add_scalar(
            "train/key_mag",
            actions[:, :4].norm(dim=1).mean().item(),
            epoch
        )
        
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(
                CHECKPOINT_DIR,
                f"checkpoint_epoch_{epoch+1}.pth"
            )

            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, ckpt_path)

            print(f"[✔] Saved checkpoint: {ckpt_path}")

        if epoch == EPOCHS - 1:
            torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict()
        }, MODEL_PATH)

    print("Training complete")



if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()