import time
class ActiveModeController:
    def __init__(self, activation_threshold=10, deactivation_timeout=5, yaw_threshold=2.0, pitch_threshold=2.0):
        self.activation_threshold = activation_threshold
        self.deactivation_timeout = deactivation_timeout
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.active_mode = False
        self.movement_count = 0
        self.last_movement_time = time.time()
        self.last_yaw = 0.0
        self.last_pitch = 0.0

    def check_movement(self, current_yaw, current_pitch):
        yaw_difference = abs(current_yaw - self.last_yaw)
        pitch_difference = abs(current_pitch - self.last_pitch)

        # Check if movement is significant
        if yaw_difference > self.yaw_threshold or pitch_difference > self.pitch_threshold:
            self.movement_count += 1
            self.last_movement_time = time.time()
        else:
            if time.time() - self.last_movement_time > self.deactivation_timeout:
                self.movement_count = 0
        if self.movement_count >= self.activation_threshold:
            self.active_mode = True
            print("Active Mode: ON")
        else:
            self.active_mode = False
            print("Active Mode: OFF")
        self.last_yaw = current_yaw
        self.last_pitch = current_pitch
