import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import radians, sin, cos
import os


def cannyImplement(image):

    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bluredImage = cv2.GaussianBlur(grayImage, (1,1), 0)
    cannyImage = cv2.Canny(bluredImage, 85, 100)

    return cannyImage


def steeringWheelDetect(firstImage, image):

    height = image.shape[0]
    polygon = np.array([[(400, 300), (850, 300), (900, height), (400,height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    maskedImage = cv2.bitwise_and(image, mask)

    return maskedImage


def detectedCircle(firstImage, circles):

    if circles is not None:
        circles = np.round(circles[0:]).astype("int")
        for (x,y,r) in circles[0,:]:
            cv2.circle(firstImage, (x, y), (190), (0, 0, 255), 4)
            cv2.circle(firstImage, (x, y-190), (10), (0, 0, 255), -1)

    return firstImage



cap = cv2.VideoCapture('MercedesSteelingWheel.mp4')

while (cap.isOpened()):

    _, frame = cap.read()
    cannyImage = cannyImplement(frame)
    maskedImage = steeringWheelDetect(frame, cannyImage)

    circles = cv2.HoughCircles(maskedImage, cv2.HOUGH_GRADIENT, 1, 600, param1=200, param2=110, minRadius=0, maxRadius=0)

    detectedWheel = detectedCircle(frame, circles)
    comboImage = cv2.addWeighted(frame, 1, detectedWheel, 0.2, 1)
    imageWay = detectingWayAngle(comboImage)


    cv2.imshow('Photo', comboImage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
