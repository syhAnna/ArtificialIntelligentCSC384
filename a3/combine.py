import cv2 as cv


s_img = cv.imread("bookCover.jpg")
l_img = cv.imread("im1.jpg")
x_offset=y_offset=50
l_img[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

cv.imwrite('q2_3combine.jpg', l_img)
