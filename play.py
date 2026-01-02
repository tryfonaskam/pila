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
MODEL_PATH = "checkpoints/checkpoint_epoch_50.pth"

IMG_SIZE = 256
FRAME_STACK = 4
FPS = 120

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_KEY = "esc"

# Mouse scaling
MOUSE_SCALE_X = 0.0
MOUSE_SCALE_Y = 0.0

# Control thresholds
QE_THRESH = 0.008
SHIFT_CTRL_THRESH = 0.005
WASD_THRESH = 0.25
SPACE_THRESH = 0.5
CLICK_THRESH = 0.5
# Integrator smoothing
MOUSE_ALPHA = 0.6
SHIFT_CTRL_ALPHA = 0.05

# config end


# model
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
        self.fc = nn.Linear(128, 9)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x.flatten(1))


# load model
print("[INFO] Loading model...")
model = ControlNet().to(DEVICE)

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
shift_ctrl_state = 0.0


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

    w_s, a_d, q_e, space, shift_ctrl, mouse_dx, mouse_dy, left_click, right_click = out

    # mouse

    target_dx = mouse_dx + a_d
    target_dy = mouse_dy + w_s

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
    if w_s > WASD_THRESH:
        keyboard.press("w")
        keyboard.release("s")
    elif w_s < -WASD_THRESH:
        keyboard.press("s")
        keyboard.release("w")
    else:
        keyboard.release("w")
        keyboard.release("s")

    # A and D
    if a_d > WASD_THRESH + 0.1:
        keyboard.press("d")
        keyboard.release("a")
    elif a_d < -WASD_THRESH:
        keyboard.press("a")
        keyboard.release("d")
    else:
        keyboard.release("a")
        keyboard.release("d")


    # Q and E
    if q_e > QE_THRESH:
        keyboard.press("e")
        keyboard.release("q")
    elif q_e < -QE_THRESH:
        keyboard.press("q")
        keyboard.release("e")
    else:
        keyboard.release("q")
        keyboard.release("e")

    # space and mouse clicks
    if space > SPACE_THRESH:
        keyboard.press("space")
    else:
        keyboard.release("space")
    
    if left_click > CLICK_THRESH:
        mouse.press(button='left')
        mouse.release(button='right')
    elif right_click < -CLICK_THRESH:
        mouse.press(button='right')
        mouse.release(button='left')
    else:
        mouse.release(button='left')
        mouse.release(button='right')
        

    # shift and ctrl
    shift_ctrl_state = (
        (1 - SHIFT_CTRL_ALPHA) * shift_ctrl_state +
        SHIFT_CTRL_ALPHA * shift_ctrl
    )

    if shift_ctrl_state > SHIFT_CTRL_THRESH:
        keyboard.press("shift")
        keyboard.release("ctrl")
    elif shift_ctrl_state < -SHIFT_CTRL_THRESH:
        keyboard.press("ctrl")
        keyboard.release("shift")
    else:
        keyboard.release("shift")
        keyboard.release("ctrl")
    print(out)
    time.sleep(1 / FPS)

# stop release all keys
camera.stop()
keyboard.release("shift")
keyboard.release("ctrl")
keyboard.release("q")
keyboard.release("e")
keyboard.release("w")
keyboard.release("a")
keyboard.release("s")
keyboard.release("d")
keyboard.release("space")
mouse.release(button='left')
mouse.release(button='right')

print("[INFO] AI stopped.")