import cv2
import numpy as np

# Constants for calibration (replace these with your calibration data)
lux_values = [10, 50, 100]  # Lux values for calibration
calibration_pixel_means = [30, 120, 200]  # Mean pixel intensity corresponding to lux_values

# Perform linear regression for calibration
calibration_coefficients = np.polyfit(calibration_pixel_means, lux_values, 1)

# Initialize camera
cap = cv2.VideoCapture(0)  # Use 0 for default camera, adjust if needed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate mean pixel intensity
    mean_intensity = np.mean(gray_frame)

    # Estimate lux based on the calibration correlation
    lux_estimate = np.polyval(calibration_coefficients, mean_intensity)

    print(f"Mean Intensity: {mean_intensity} Lux Estimate: {lux_estimate}")

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
