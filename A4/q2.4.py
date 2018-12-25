import cv2 as cv
import numpy as np


img1_dict = {'num_detection': 6, 'detection_boxes': [[0.50004214, 0.6555561 , 0.8872184 , 0.8707886 ],
       [0.44908354, 0.584361  , 0.5227398 , 0.6364023 ],
       [0.01041733, 0.5026116 , 0.09855365, 0.5190967 ],
       [0.46349224, 0.42532513, 0.5317025 , 0.47155157],
       [0.4670957 , 0.35683563, 0.53998923, 0.3719673 ],
       [0.49316233, 0.23185101, 0.54320043, 0.26347476]],
        'detection_scores': [0.9491954 , 0.903302  , 0.7825848 , 0.72892267, 0.66242313, 0.57823133],
        'detection_classes': [ 3,  3, 10,  3,  1,  3]}
img2_dict = {'num_detection': 6, 'detection_boxes': [[0.47048816, 0.44235417, 0.5088938 , 0.474377  ],
       [0.46128154, 0.5218015 , 0.55010664, 0.58383596],
       [0.4213005 , 0.7209805 , 0.5365165 , 0.80251265],
       [0.47370738, 0.49366155, 0.520934  , 0.5202959 ],
       [0.5039439 , 0.15278855, 0.5627457 , 0.21511334],
       [0.28877726, 0.29553753, 0.3853735 , 0.3158021 ]],
        'detection_scores': [0.88989836, 0.7733079 , 0.7360068 , 0.72747695, 0.65513057, 0.555762  ],
        'detection_classes': [ 3,  3,  3,  3,  3, 10]}
img3_dict = {'num_detection': 4, 'detection_boxes': [[0.46915835, 0.39268324, 0.59815925, 0.51287454],
       [0.49698436, 0.16895944, 0.70170236, 0.31567362],
       [0.41786578, 0.8332899 , 0.6043134 , 0.986494  ],
       [0.0924626 , 0.814217  , 0.18619645, 0.8461287 ]],
        'detection_scores': [0.88686144, 0.8019987 , 0.72281444, 0.50532067],
        'detection_classes': [ 3,  3,  3, 10]}
calib = {'f': 721.537700, 'px': 609.559300, 'py': 172.854000, 'baseline': 0.5327119288}


def calculate_depth(img):
    disparity = cv.imread(img)
    depth = np.true_divide(calib['f'] * calib['baseline'], disparity)
    return depth


def center(ZDepth, img, img_dict):
    img = cv.imread(img)
    height, width = img.shape[0], img.shape[1]
    box_depth = []

    for i in range(img_dict['num_detection']):
        depth = []
        box = img_dict['detection_boxes'][i]
        pt1 = (int(box[1] * width), int(box[0] * height))
        pt2 = (int(box[3] * width), int(box[2] * height))

        for row in range(pt1[1], pt2[1]):
            for column in range(pt1[0], pt2[0]):
                z = ZDepth[row, column, 0]
                x = np.true_divide((column - calib['px']) * z, calib['f'])
                y = np.true_divide((row - calib['py']) * z, calib['f'])
                depth.append((z, x, y))
        depth.sort()
        box_depth.append(depth[len(depth) // 2])  # (z, x, y)

    print(box_depth)
    return box_depth


def segmentation(img, ZDepth, box_depth, img_dict):
    img = cv.imread(img)
    height, width = img.shape[0], img.shape[1]
    segmentation = np.zeros((height, width, 3))
    num_detection = img_dict['num_detection']

    for i in range(num_detection):
        box = img_dict['detection_boxes'][i]
        pt1 = (int(box[1] * width), int(box[0] * height))
        pt2 = (int(box[3] * width), int(box[2] * height))

        for row in range(pt1[1], pt2[1]):
            for column in range(pt1[0], pt2[0]):
                z = ZDepth[row, column, 0]
                x = np.true_divide((column - calib['px']) * z, calib['f'])
                y = np.true_divide((row - calib['py']) * z, calib['f'])
                if (z - box_depth[i][0]) ** 2 + (x - box_depth[i][1]) ** 2 + (y - box_depth[i][2]) ** 2 <= 9:
                    segmentation[row, column] = 255 - 30 * i

    return segmentation


if __name__ == '__main__':
    ZDepth1 = calculate_depth('004945_left_disparity.png')
    ZDepth2 = calculate_depth('004964_left_disparity.png' )
    ZDepth3 = calculate_depth('005002_left_disparity.png' )

    box_depth1 = center(ZDepth1, '004945_left_disparity.png', img1_dict)
    box_depth2 = center(ZDepth2, '004964_left_disparity.png', img2_dict)
    box_depth3 = center(ZDepth3, '005002_left_disparity.png', img3_dict)

    cv.imwrite('004945seg.jpg', segmentation('004945.jpg', ZDepth1, box_depth1, img1_dict))
    cv.imwrite('004964seg.jpg', segmentation('004964.jpg', ZDepth2, box_depth2, img2_dict))
    cv.imwrite('005002seg.jpg', segmentation('005002.jpg', ZDepth3, box_depth3, img3_dict))

