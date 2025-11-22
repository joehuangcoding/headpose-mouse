import cv2
import numpy as np
import tensorflow as tf
import math
import pyautogui
def get_eye_roi(frame, gray_frame, eye):
    eye_rect = cv2.boundingRect(np.array(eye))
    eye_rect = (
        eye_rect[0] - 15,  # Decrease x-coordinate to make it bigger
        eye_rect[1] - 10,  # Decrease y-coordinate to make it bigger
        eye_rect[2] + 30,  # Increase width
        eye_rect[3] + 30   # Increase height
    )
    cv2.rectangle(frame, (eye_rect[0], eye_rect[1]), (eye_rect[0]+eye_rect[2], eye_rect[1]+eye_rect[3]), (0, 0, 255), 2)
    eye_roi = gray_frame[eye_rect[1]:eye_rect[1]+eye_rect[3], eye_rect[0]:eye_rect[0]+eye_rect[2]]
    return eye_roi

def process_eye_images(model, eye_images):
    eye_images = eye_images.astype(np.float32) / 255.0
    # run the eye images through the model on GPU
    with tf.device('/GPU:0'):
        eye_predictions = model.predict(eye_images, verbose=None)
    return eye_predictions

def calculate_centroid(landmarks_list):
    x = sum([point[0] for point in landmarks_list]) // len(landmarks_list)
    y = sum([point[1] for point in landmarks_list]) // len(landmarks_list)
    return int(x), int(y)

def calculate_distance(point1, point2):
    return ((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)**0.5

def calculate_movement_direction(nose_landmark, face_centroid, threshold=8):
    x, y = nose_landmark[0], nose_landmark[1]
    if x < face_centroid[0] - threshold:
        return "left"
    elif x > face_centroid[0] + threshold:
        return "right"
    elif y < face_centroid[1] - threshold:
        return "up"
    elif y > face_centroid[1] + 2:
        return "down"
    else:
        return "center"