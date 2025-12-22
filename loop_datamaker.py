import dxcam
import cv2
import numpy as np
import pandas as pd
import keyboard
import mouse
import os
import time

# config start

FPS = 35
FRAME_TIME = 1.0 / FPS

BASE_DIR = "dataset"      # Base folder for all runs
REGION = None             # None = full monitor; or set (left, top, right, bottom)

# Control keys
KEY_PITCH_UP    = 's'
KEY_PITCH_DOWN  = 'w'
KEY_YAW_LEFT    = 'a'
KEY_YAW_RIGHT   = 'd'
KEY_ROLL_LEFT   = 'q'
KEY_ROLL_RIGHT  = 'e'
KEY_THROTTLE_UP = 'shift'
KEY_THROTTLE_DN = 'ctrl'

START_KEY = 'F1'
STOP_KEY  = 'F10'
EXIT_KEY  = 'F2'  # Press ESC to exit the script entirely

# config end

def axis(neg, pos):
    return float(keyboard.is_pressed(pos)) - float(keyboard.is_pressed(neg))

def throttle():
    if keyboard.is_pressed(KEY_THROTTLE_UP):
        return 1.0
    if keyboard.is_pressed(KEY_THROTTLE_DN):
        return -1.0
    return 0.0

# auto-run-manager
def get_next_run_folder(base_dir="dataset"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("run")]
    nums = [int(d[3:]) for d in existing if d[3:].isdigit()]
    next_num = max(nums)+1 if nums else 1
    run_folder = os.path.join(base_dir, f"run{next_num:02d}")
    frames_folder = os.path.join(run_folder, "frames")
    os.makedirs(frames_folder, exist_ok=True)
    return run_folder, frames_folder

# main

print(" Initializing DX capture...")
camera = dxcam.create(output_color="BGR")
camera.start(target_fps=FPS, video_mode=True, region=REGION)
time.sleep(1)

print(f"\n{'='*50}")
print(f" Press {START_KEY} to START recording")
print(f" Press {STOP_KEY} to STOP recording")
print(f" Press {EXIT_KEY} to EXIT the script")
print(f"{'='*50}\n")

try:
    while True:
        # Wait for start key
        keyboard.wait(START_KEY)
        
        # Check if exit was pressed instead
        if keyboard.is_pressed(EXIT_KEY):
            break
        
        # Get new run folder for this recording session
        DATASET_DIR, FRAMES_DIR = get_next_run_folder(BASE_DIR)
        CSV_PATH = os.path.join(DATASET_DIR, "actions.csv")
        print(f" Recording to: {DATASET_DIR}")
        print(" Recording started")

        records = []
        frame_id = 0
        last_mouse_x, last_mouse_y = mouse.get_position()

        # Recording loop
        while not keyboard.is_pressed(STOP_KEY):
            start = time.time()

            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            # Save frame
            frame_name = f"{frame_id:06d}.png"
            cv2.imwrite(os.path.join(FRAMES_DIR, frame_name), frame)

            # Keyboard controls
            elevator = axis(KEY_PITCH_UP, KEY_PITCH_DOWN)
            rudder   = axis(KEY_YAW_LEFT, KEY_YAW_RIGHT)
            roll_qe  = axis(KEY_ROLL_LEFT, KEY_ROLL_RIGHT)
            thr      = throttle()

            # Mouse deltas (relative)
            mx, my = mouse.get_position()
            mouse_dx = mx - last_mouse_x
            mouse_dy = my - last_mouse_y

            # Clamp large jumps (Alt+Tab, multi-monitor)
            mouse_dx = max(-20, min(20, mouse_dx))
            mouse_dy = max(-20, min(20, mouse_dy))

            # Scale to -1..1 for training
            mouse_dx_scaled = mouse_dx / 20.0
            mouse_dy_scaled = mouse_dy / 20.0

            last_mouse_x, last_mouse_y = mx, my

            # Save record
            records.append([
                frame_name,
                elevator,
                rudder,
                roll_qe,
                thr,
                mouse_dx_scaled,
                mouse_dy_scaled
            ])

            frame_id += 1
            elapsed = time.time() - start
            time.sleep(max(0, FRAME_TIME - elapsed))

        # Save data after stopping
        df = pd.DataFrame(records, columns=[
            "frame","elevator","rudder","roll_qe","throttle","mouse_dx","mouse_dy"
        ])
        df.to_csv(CSV_PATH, index=False)
        print(f" Saved {len(df)} samples to {CSV_PATH}")
        print(f"\n Press {START_KEY} to start another recording, or {EXIT_KEY} to exit\n")
        
        # Small delay to prevent immediate re-trigger
        time.sleep(0.5)

finally:
    camera.stop()
    print("\n Recording script terminated")