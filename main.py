import cv2 as cv
import imutils
import argparse
import numpy as np

# Calculates lines from Hough Transform, merges into one line based on average
def calculate_lines(frame, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
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
    y1 = frame.shape[0]
    y2 = int(y1 - 800)

    x1 = int((y1 - intercept) / slope)
    x2= int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])

#Takes data from calculated lines and adds them to frame. Currently modified to exclude create_lines and calculate_coordinates.
def create_lines(img, lines):

    lines_visualize = np.zeros_like(img)

    if lines is not None:
        for x1, y1, x2, y2 in lines:
                cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 15)

    return lines_visualize


# Load test image from Dashcam
#image = cv.imread("test_images/out2.png")

#Load sample video - Eventually convert this to desktop capture for GTA
vid = cv.VideoCapture("test_images/sample.mp4")

while (vid.isOpened()):

    ret, frame = vid.read()

    # Grayscale image
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #fGray = cv.bilateralFilter(gray, 7, 75, 75)

    pImage = cv.Canny(gray, 150, 250, 4)
    #pImage = fGray

# Mask to do trapezoidish FOV to remove unnessary data from frame - Currently Calibrated to test frame, need to recalibrate for GTA sample later
# TO-DO: Make dynamic based on Height & Width of source frames. i.e: OldManDan running it on his machine with a different resolution
    mask = np.zeros_like(pImage)
    height = pImage.shape[0]
    #Edit ROI_POINTS per perspective or mask will be bad.
    roi_points = np.array([[0, 1150], [0, 900], [1000,550], [1600,550], [2560, 900], [2560, 1150]], np.int32)
    cv.fillConvexPoly(mask, roi_points, (255,255,255))
    masked_image = cv.bitwise_and(pImage, mask)

    # hough = cv.HoughLinesP(masked_image, 2, np.pi / 180, 50, np.array([]), 100, 100)


    # lines = calculate_lines(frame, hough)

    # lines_visualize = create_lines(frame, lines)

    # output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)

    cv.imshow("Test", masked_image)

    if cv.waitKey(10) == ord('q'):
        break

vid.release()
cv.destroyAllWindows()