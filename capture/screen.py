import mss
import numpy as np
import cv2

class ScreenCapture:
    def __init__(self, region=None):
        self.sct = None
        self.region = region

    def start(self):
        if self.sct is None:
            self.sct = mss.mss()
            self.monitor = self.region if self.region else self.sct.monitors[0]

    def grab(self):
        if self.sct is None:
            raise RuntimeError("ScreenCapture not started")

        frame = np.array(self.sct.grab(self.monitor))
        frame = frame[:, :, :3]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
