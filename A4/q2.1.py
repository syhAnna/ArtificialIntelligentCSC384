import cv2 as cv
import numpy as np

# _ALL CALIB.txt from first 3 images
calib = {'f': 721.537700, 'px': 609.559300, 'py': 172.854000, 'baseline': 0.5327119288}


def calculate_depth(imgs, calib):
    disparity = cv.imread(imgs)
    depth = np.true_divide(calib['f'] * calib['baseline'], disparity)
    # consider the meter of the depth (1000) in the later calculation
    print(depth)
    return depth


if __name__ == '__main__':
    cv.imwrite('004945q1.jpg', calculate_depth('004945_left_disparity.png', calib))
    cv.imwrite('004964q1.jpg', calculate_depth('004964_left_disparity.png', calib))
    cv.imwrite('005002q1.jpg', calculate_depth('005002_left_disparity.png', calib))
