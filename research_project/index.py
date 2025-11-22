import sys
# sys.path.append('3DDFA_V2')
import os

# Add 3DDFA_V2 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '3DDFA_V2'))
import cv2
import yaml
import pyautogui
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import calc_pose
import numpy as np
import math
import time
import os
from fps import FPSCounter
from tensorflow.keras.models import load_model
import tensorflow as tf
from active_mode_controller import ActiveModeController
from helper import get_eye_roi, process_eye_images, calculate_centroid, calculate_movement_direction, calculate_distance

pyautogui.FAILSAFE = False

# Load trained eye blink model. The CNN model is trained on MRL Eye Blink dataset
model = load_model('./eye_blink_30_30_gray_2l.h5')

# Load config
cfg = yaml.load(open('./3DDFA_V2/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Initialize FaceBoxes and TDDFA two steps facial landmarks detector
face_boxes = FaceBoxes()
tddfa = TDDFA(gpu_mode=False, **cfg)

cap = cv2.VideoCapture(0)
fps_counter = FPSCounter()

neutral_pitch = 0
neutral_yaw = 0 
speed_scaling_factor = 0.04
pitch = 0
yaw = 0
min_pixel_threshold = 900
blink_counter = 0
blink_counter_total = 0
blink_duration_threshold = 0.2
last_process_time = 0
data_array = []
data_array_right = []
last_blink_time = 0
activeModeController = ActiveModeController(activation_threshold=5, deactivation_timeout=10, yaw_threshold=5.0, pitch_threshold=5.0)
lock_neutral_time = 20
start_time = time.time()
remaining_time = 0
is_in_origin = True
face_centroid = (0, 0)
nose_landmark = [0, 0]
distance = 0
distance_threshold = 8
movement_direction = ''
pitch = 0
yaw = 0
last_mouse_process_time = 0
last_eye_detection_run = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Couldn't capture frame.")
        break
    # height, width = frame.shape[:2]
    # print('height', height, 'width', width)
    frame = cv2.resize(frame, (640,480))
    # frame = cv2.resize(frame, (320,240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fps_counter.increment_frame_count()
    fps_counter.update_fps()
    cv2.putText(frame, f"FPS: {fps_counter.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    boxes = face_boxes(frame)
    param_lst, roi_box_lst = tddfa(frame, boxes)
    current_time = time.time()

    if param_lst is not None:
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
        for param, ver in zip(param_lst, ver_lst):
            x_coordinates = ver[0]
            y_coordinates = ver[1]
            landmarks_list = list(zip(x_coordinates, y_coordinates))
            for landmark in landmarks_list:
                x, y = map(int, landmark)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            nose_landmark = landmarks_list[30]
            face_centroid = calculate_centroid(landmarks_list)
            movement_direction = calculate_movement_direction(nose_landmark, face_centroid, threshold=distance_threshold)
            distance = calculate_distance((nose_landmark[0], nose_landmark[1]), face_centroid)

            # check if active mode is on
            # activeModeController.check_movement(nose_landmark, face_centroid)

            P, pose = calc_pose(param)
            yaw = pose[0]
            pitch = pose[1]
            # detect blinks
            if current_time - last_eye_detection_run >= 0.05:
                last_eye_detection_run = current_time
                left_eye = landmarks_list[36: 42]
                right_eye = landmarks_list[42: 48]
                left_eye_roi = get_eye_roi(frame, gray, left_eye)
                right_eye_roi = get_eye_roi(frame, gray, right_eye)
                if left_eye_roi.size > min_pixel_threshold and right_eye_roi.size > min_pixel_threshold and is_in_origin:
                    left_eye_image = cv2.resize(left_eye_roi, (30, 30))
                    right_eye_image = cv2.resize(right_eye_roi, (30, 30))
                    data_array.append(left_eye_image)
                    data_array_right.append(right_eye_image)
                    # checking if 0.25 seconds have passed
                    if current_time - last_process_time >= blink_duration_threshold:
                        last_process_time = current_time
                        eye_predictions = process_eye_images(model, np.array(data_array + data_array_right))
                        # print(eye_predictions)
                        left_eye_predictions = eye_predictions[:len(eye_predictions)//2]
                        right_eye_predictions = eye_predictions[len(eye_predictions)//2:]
                        left_eye_prediction_mean = np.mean(left_eye_predictions)
                        right_eye_prediction_mean = np.mean(right_eye_predictions)
                        if left_eye_prediction_mean <= 0.5 or right_eye_prediction_mean <= 0.5:
                            blink_counter += 1
                            blink_counter_total += 1
                            last_blink_time = time.time()
                        data_array = []
                        data_array_right = []
            activeModeController.check_movement(yaw, pitch)
            is_in_origin = False
            if movement_direction == 'center':
                is_in_origin = True
                dot_radius = 50
                dot_color = (0, 0, 255) if is_in_origin else (0, 0, 0)
                cv2.circle(frame, (frame.shape[1] // 2, frame.shape[0] // 2), dot_radius, dot_color, -1)
                continue
             # check if the head is moving. Active_mode is set to True if the head is moving
            
            if activeModeController.active_mode:
                if current_time - last_mouse_process_time >= 0.2:
                    last_mouse_process_time = current_time
                    # Calculate the speed based on the angles. The speed is calculated using exponential function
                    speed_yaw = math.exp(abs(yaw - neutral_yaw) * speed_scaling_factor)
                    speed_pitch = math.exp(abs(pitch - neutral_pitch) * speed_scaling_factor)
                    # Calculate the movement based on the angles and speed
                    # Move the cursor
                    move_x = yaw * speed_yaw
                    move_y = pitch * speed_pitch
                    if movement_direction == 'left' or movement_direction == 'right':
                        move_y = 0
                    elif movement_direction == 'up' or movement_direction == 'down':
                        move_x = 0
                    pyautogui.move(move_x, move_y)
        cv2.circle(frame, face_centroid, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"face center x:{face_centroid[0]} y: {face_centroid[1]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Nose x:{nose_landmark[0]} Nose y: {nose_landmark[1]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Blink counts total: {blink_counter_total:.2f} 3 blinks left click', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Active: {activeModeController.active_mode:.2f}. Shake head to set 1', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"distance:{distance}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Movement: {movement_direction}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"yaw: {yaw}", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"pitch: {pitch}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # left click when blinking three times in two seconds and only when active mode is on
    if current_time- last_blink_time < 2 and activeModeController.active_mode:
        if blink_counter == 3:
            pyautogui.click(button='left')
            blink_counter = 0
            print("Left click simulated!")
    else:
        blink_counter = 0

    cv2.imshow('Frame', frame)
    cv2.setWindowProperty('Frame', cv2.WND_PROP_TOPMOST, 1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
