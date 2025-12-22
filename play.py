import time
import cv2
import torch
import dxcam
import keyboard
import mouse
import numpy as np
from collections import deque
from torch import nn
time.sleep(5)

# config start
MODEL_PATH = "models/checkpoint_epoch_60.pth"

IMG_SIZE = 256
FRAME_STACK = 4
FPS = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_KEY = "esc"

# Mouse scaling
MOUSE_SCALE_X = 0.0
MOUSE_SCALE_Y = 0.0

# Control thresholds
ROLL_THRESH = 0.008
THROTTLE_THRESH = 0.005
WASD_THRESH = 0.33

# Integrator smoothing
MOUSE_ALPHA = 0.6
THROTTLE_ALPHA = 0.05

# config end


# mdoel
class FlightNet(nn.Module):
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


# load model
print("[INFO] Loading model...")
model = FlightNet().to(DEVICE)

# auto loading of different checkpoint formats

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict):
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"[INFO] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        print(f"[INFO] Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")
    else:
        model.load_state_dict(checkpoint)
        print("[INFO] Loaded raw model state_dict")
else:
    model.load_state_dict(checkpoint)
    print("[INFO] Loaded raw model state_dict")

model.eval()


print(f"[INFO] Loaded epoch {checkpoint.get('epoch', '?')}")
print(f"[INFO] Device: {DEVICE}")

# recording setup
camera = dxcam.create(output_color="BGR")
camera.start(target_fps=FPS)

frame_buffer = deque(maxlen=FRAME_STACK)

# states
mouse_x_state = 0.0
mouse_y_state = 0.0
throttle_state = 0.0


def preprocess(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = frame.astype(np.float32) / 255.0
    return np.transpose(frame, (2, 0, 1))


print("[INFO] AI ACTIVE — press ESC to stop")

# main
while not keyboard.is_pressed(STOP_KEY):
    frame = camera.get_latest_frame()
    if frame is None:
        continue

    frame_buffer.append(preprocess(frame))
    if len(frame_buffer) < FRAME_STACK:
        continue

    stacked = np.concatenate(frame_buffer, axis=0)
    inp = torch.from_numpy(stacked).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(inp)[0].cpu().numpy()

    elevator, rudder, roll_qe, throttle, mouse_dx, mouse_dy = out

    # mouse

    target_dx = mouse_dx + rudder
    target_dy = mouse_dy + elevator

    mouse_x_state = MOUSE_ALPHA * mouse_x_state + (1 - MOUSE_ALPHA) * target_dx
    mouse_y_state = MOUSE_ALPHA * mouse_y_state + (1 - MOUSE_ALPHA) * target_dy

    mouse.move(
        mouse_x_state * MOUSE_SCALE_X,
        mouse_y_state * MOUSE_SCALE_Y,
        absolute=False,
        duration=0
    )


    # wasd

    # W and S
    if elevator > WASD_THRESH:
        keyboard.press("w")
        keyboard.release("s")
    elif elevator < -WASD_THRESH:
        keyboard.press("s")
        keyboard.release("w")
    else:
        keyboard.release("w")
        keyboard.release("s")

    # A and D
    if rudder > WASD_THRESH:
        keyboard.press("d")
        keyboard.release("a")
    elif rudder < -WASD_THRESH:
        keyboard.press("a")
        keyboard.release("d")
    else:
        keyboard.release("a")
        keyboard.release("d")


    # Q and E
    if roll_qe > ROLL_THRESH:
        keyboard.press("e")
        keyboard.release("q")
    elif roll_qe < -ROLL_THRESH:
        keyboard.press("q")
        keyboard.release("e")
    else:
        keyboard.release("q")
        keyboard.release("e")

    # shift and ctrl
    throttle_state = (
        (1 - THROTTLE_ALPHA) * throttle_state +
        THROTTLE_ALPHA * throttle
    )

    if throttle_state > THROTTLE_THRESH:
        keyboard.press("shift")
        keyboard.release("ctrl")
    elif throttle_state < -THROTTLE_THRESH:
        keyboard.press("ctrl")
        keyboard.release("shift")
    else:
        keyboard.release("shift")
        keyboard.release("ctrl")

    time.sleep(1 / FPS)

# stop realease all keys
camera.stop()
keyboard.release("shift")
keyboard.release("ctrl")
keyboard.release("q")
keyboard.release("e")
keyboard.release("w")
keyboard.release("a")
keyboard.release("s")
keyboard.release("d")

print("[INFO] AI stopped.")
