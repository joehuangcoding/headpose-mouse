import cv2
import yaml
import pyautogui
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import calc_pose
import math

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# Load config
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

# Initialize FaceBoxes and TDDFA
face_boxes = FaceBoxes()
tddfa = TDDFA(gpu_mode=False, **cfg)

# Open a connection to the camera (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Smoothing parameters
alpha = 0.2  # Smoothing factor (0 < alpha < 1)
smoothed_x, smoothed_y = pyautogui.position()

# Thresholds for yaw and pitch angles
yaw_threshold = 2  # Adjust this threshold to control when the cursor should move based on yaw angle
pitch_threshold = 2  # Adjust this threshold to control when the cursor should move based on pitch angle

previous_predicted_pitch = 0
previous_predicted_yaw = 0
neutral_pitch = 0
neutral_yaw = 0
speed_scaling_factor = 0.05
move_x = 0
move_y = 0
speed_pitch = 0
speed_yaw = 0
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Couldn't capture frame.")
        break

    # Face detection
    boxes = face_boxes(frame)
    print(f'Detect {len(boxes)} faces')
    print(boxes)

    # Regress 3DMM params
    param_lst, roi_box_lst = tddfa(frame, boxes)

    # Visualization of head pose
    if param_lst is not None:
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
        for param, ver in zip(param_lst, ver_lst):
            P, pose = calc_pose(param)

            predicted_pitch = pose[0]
            predicted_yaw = pose[1]

            # Check if the change in pitch is greater than the threshold
            if abs(predicted_pitch - previous_predicted_pitch) > pitch_threshold:
                # If above the threshold, update stable_pitch
                stable_pitch = predicted_pitch
            else:
                # If below the threshold, use the previous stable pitch
                stable_pitch = previous_predicted_pitch

            # Check if the change in yaw is greater than the threshold
            if abs(predicted_yaw - previous_predicted_yaw) > yaw_threshold:
                # If above the threshold, update stable_yaw
                stable_yaw = predicted_yaw
            else:
                # If below the threshold, use the previous stable yaw
                stable_yaw = previous_predicted_yaw

            # Update the previous_predicted_pitch and previous_predicted_yaw for the next iteration
            previous_predicted_pitch = stable_pitch
            previous_predicted_yaw = stable_yaw

            # Check if the angles are within +5 and -5 degrees
            if abs(stable_pitch - neutral_pitch) <= 10 and -5 <= stable_yaw - neutral_yaw <= 5:
                # Do not move the cursor if within the specified range
                continue

            # Exponentially scale the speed based on the deviation from the neutral position
            speed_pitch = math.exp(abs(stable_pitch - neutral_pitch) * speed_scaling_factor)
            speed_yaw = math.exp(abs(stable_yaw - neutral_yaw) * speed_scaling_factor)

            # Calculate the movement based on the angles and speed
            move_y = stable_yaw * speed_yaw
            move_x = stable_pitch * speed_pitch

            # Move the cursor
            pyautogui.move(move_x, move_y)

    text = f'yaw: {stable_yaw:.1f}, pitch: {stable_pitch:.1f}, roll: {pose[2]:.1f}'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'move_x: {move_x}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'move_y: {move_y}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'speed_pitch: {speed_pitch}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'speed_yaw: {speed_yaw}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
