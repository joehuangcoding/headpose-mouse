import time

class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0

    def update_fps(self):
        elapsed_time = time.time() - self.start_time

        # Update FPS every second
        if elapsed_time >= 1:
            self.fps = self.frame_count / elapsed_time
            # print(f"FPS: {self.fps:.2f}")

            # Reset frame counter and start time
            self.frame_count = 0
            self.start_time = time.time()

    def increment_frame_count(self):
        self.frame_count += 1

    def fps(self):
        return self._fps