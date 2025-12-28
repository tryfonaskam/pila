import os
import time
import cv2
import pandas as pd
from pynput import keyboard

from capture.screen import ScreenCapture
from capture.input_state import InputState

FPS = 60
SAVE_DIR = "datasets"

def next_run_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)
    runs = [d for d in os.listdir(SAVE_DIR) if d.startswith("run")]
    return os.path.join(SAVE_DIR, f"run{len(runs)+1:02d}")

def main():
    run_dir = next_run_dir()
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir)

    screen = ScreenCapture()
    inputs = InputState()

    actions = []
    frame_idx = 0
    recording = False

    def on_key(key):
        nonlocal recording
        if key == keyboard.KeyCode.from_char('r'):
            print("[INFO] Recording started")
            screen.start()
            inputs.start()
            recording = True
        elif key == keyboard.KeyCode.from_char('q'):
            print("[INFO] Recording stopped")
            return False

    listener = keyboard.Listener(on_press=on_key)
    listener.start()

    print(">>> PRESS r TO START, q TO STOP <<<")

    try:
        while listener.running:
            if recording:
                frame = screen.grab()
                action = inputs.snapshot()

                path = os.path.join(frames_dir, f"{frame_idx:06d}.png")
                cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                actions.append(action)
                frame_idx += 1

            time.sleep(1 / FPS)

    finally:
        pd.DataFrame(actions).to_csv(
            os.path.join(run_dir, "actions.csv"), index=False
        )
        print(f"[INFO] Saved run to {run_dir}")

if __name__ == "__main__":
    main()
