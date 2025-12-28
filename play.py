import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from pynput import keyboard

from capture.screen import ScreenCapture

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "checkpoints/pila_final.pt"
IMAGE_SIZE = (512, 512)

FPS = 240
#DEVICE = "cpu"   # inference stability
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("[INFO] Using device:", DEVICE)

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
# KILL SWITCH
# --------------------
controller = keyboard.Controller()
running = True

def on_press(key):
    global running
    if key == keyboard.Key.esc:
        print("[INFO] ESC pressed — stopping agent")
        running = False
        return False

keyboard.Listener(on_press=on_press).start()

# --------------------
# SCREEN CAPTURE
# --------------------
screen = ScreenCapture()
screen.start()

# --------------------
# INFER FC SIZE
# --------------------
frame = screen.grab()
frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

img = frame.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))
img = torch.tensor(img).unsqueeze(0)

with torch.no_grad():
    conv_probe = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2),
        nn.ReLU(),
        nn.Conv2d(16, 32, 5, stride=2),
        nn.ReLU(),
        nn.Conv2d(32, 64, 5, stride=2),
        nn.ReLU(),
    )
    conv_out = conv_probe(img)
    conv_out_size = conv_out.view(1, -1).shape[1]

# --------------------
# LOAD MODEL
# --------------------
model = CNNPolicy(conv_out_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

print("[INFO] Model loaded")
print("[INFO] Switch to the game window (5 seconds)")
time.sleep(5)

# --------------------
# MAIN LOOP
# --------------------
print("[INFO] Agent running — press ESC to stop")

with torch.no_grad():
    while running:
        frame = screen.grab()
        frame = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

        img = frame.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(DEVICE)

        w, a, s, d = model(img)[0].cpu().numpy()
        controller.release('w')
        controller.release('a')
        controller.release('s')
        controller.release('d')
        if s > 0.4:
            controller.press('s')
        elif w > 0.2:
            controller.press('w')

        if a > d and a > 0.3:
            controller.press('a')
        elif d > a and d > 0.3:
            controller.press('d')

        time.sleep(1 / FPS)

# --------------------
# CLEANUP
# --------------------
controller.release('w')
controller.release('a')
controller.release('s')
controller.release('d')
print("[INFO] Agent stopped")
