from pynput import keyboard
import threading

class InputState:
    def __init__(self):
        self.keys = set()
        self.lock = threading.Lock()
        self.listener = None

    def start(self):
        if self.listener is None:
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release
            )
            self.listener.start()

    def on_press(self, key):
        with self.lock:
            self.keys.add(key)

    def on_release(self, key):
        with self.lock:
            self.keys.discard(key)

    def snapshot(self):
        with self.lock:
            keys = self.keys.copy()

        return {
            "w": keyboard.KeyCode.from_char('w') in keys,
            "a": keyboard.KeyCode.from_char('a') in keys,
            "s": keyboard.KeyCode.from_char('s') in keys,
            "d": keyboard.KeyCode.from_char('d') in keys,
        }
