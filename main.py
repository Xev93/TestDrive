import cv2 as cv2
import imutils
import argparse
import numpy as np

def calculate_lines(frames, lines):
    left = []
    right = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,y1), (x2, y2), 1)
        slope = parameters[0]

        y_intercept = parameters[1]

        if slope <0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))

    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)

    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)

    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    y1 = image.shape[0]
    y2 = int(y1 - 150)

    x1 = int((y1 - intercept) / slope)
    x2= int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])



# Load test image from Dashcam
image = cv2.imread("test_images/out9.png", -1)

# Grayscale image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Canny to map edges
eConvert = cv2.Canny(gray, 60, 200)

# Mask to do trapezoidish FOV to remove unnessary data from frame - Currently Calibrated to test frame, need to recalibrate for GTA sample later
# TO-DO: Make dynamic based on Height & Width of source frames. i.e: OldManDan running it on his machine with a different resolution
mask = np.zeros_like(eConvert)
height = eConvert.shape[0]
roi_points = np.array([[0, height], [1150,840], [1300,840], [2560, height]], np.int32)
cv2.fillConvexPoly(mask, roi_points, 255)
masked_image = cv2.bitwise_and(eConvert, mask)

hough = cv2.HoughLinesP(masked_image, 2, np.pi / 180, 100, np.array([]), 100, 50)

lines_visualize = np.zeros_like(image)

for line in hough:
    for x1, y1, x2, y2 in line:
        cv2.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 15)

output = cv2.addWeighted(image, 0.9, lines_visualize, 1, 1)

cv2.imshow("Test", output)
cv2.waitKey(0)
exit()
