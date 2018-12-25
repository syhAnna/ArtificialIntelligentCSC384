import cv2 as cv

img = cv.imread('portrait.jpg', 0)
out = cv.Canny(img, 180, 480)

cv.imwrite('canny.jpg', out)
