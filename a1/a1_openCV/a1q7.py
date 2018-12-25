import cv2 as cv
import numpy as np

# read image
img = cv.imread("portrait.jpg", 0)
# gaussian blur
blur = cv.GaussianBlur(img,(3, 3),0)

# sobel-x and sobel-y

# 
