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


def num_object(img_dict):
    count_person, count_bicycle, count_car, count_traffic_light = 0, 0, 0, 0
    type_dict = {}

    for i in range(img_dict['num_detection']):
        classes = img_dict['detection_classes'][i]

        if classes == 1:
            # person
            count_person += 1
            if 1 in type_dict:
                type_dict[1].append(i)
            else:
                type_dict[1] = [i]
        elif classes == 2:
            # bicycle
            count_bicycle += 1
            if 2 in type_dict:
                type_dict[2].append(i)
            else:
                type_dict[2] = [i]
        elif classes == 3:
            # car
            count_car += 1
            if 3 in type_dict:
                type_dict[3].append(i)
            else:
                type_dict[3] = [i]
        elif classes == 10:
            # traffic_light
            count_traffic_light += 1
            if 10 in type_dict:
                type_dict[4].append(i)
            else:
                type_dict[4] = [i]

    str1 = 'There is(are) {} person in the scene; {} bicycle(s) in the scene; {} car(s) in the scene.'.format(
        count_person, count_bicycle, count_car)
    print(str1)

    if count_traffic_light != 0:
        str2 = 'There is(are) {} traffic light nearby.'.format(count_traffic_light)
        print(str2)

    return type_dict


def calculate_depth(img):   # input is left_disparity
    disparity = cv.imread(img)
    depth = np.true_divide(calib['f'] * calib['baseline'], disparity)
    return depth


def center(ZDepth, img, img_dict):
    """
    Return list of (z, x, y) of the center point of the object inside the box.
    """
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

    # print(box_depth)
    return box_depth


def find_closest(img_dict, box_depth, type_dict):
    """
    Find the closest object among all. box_depth is list of (z, x, y)
    """
    classes = type_dict.keys()

    for types in classes:
        types_index = type_dict[types]  # list of index
        min_index_distance = (0, np.inf)
        for i in types_index:
            pt = box_depth[i]
            distance = (pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2) ** (1 / 2)
            if distance < min_index_distance[1]:
                min_index_distance = (i, distance)
        X = box_depth[min_index_distance[0]][1]
        print_helper(min_index_distance[0], X, min_index_distance[1], img_dict)


def print_helper(index, X, min_distance, img_dict):
    if X >= 0:
        txt = 'to your right'
    else:
        txt = 'to your left'

    types = img_dict['detection_classes'][index]
    if types == 1:
        # person
        label = 'person'
    elif types == 2:
        # bicycle
        label = 'bicycle'
    elif types == 3:
        # car
        label = 'car'
    elif types == 10:
        # traffic light
        label = 'traffic light'

    str = 'There is a {} {} meters {}.\n It is {} meters away from you\n'.format(label, abs(X), txt, min_distance)
    print(str)


if __name__ == '__main__':
    # image1
    print('============ image1 ============')
    ZDepth1 = calculate_depth('004945_left_disparity.png')
    box_depth1 = center(ZDepth1, '004945_left_disparity.png', img1_dict)

    type_dict1 = num_object(img1_dict)
    find_closest(img1_dict, box_depth1, type_dict1)

    # image2
    print('============ image2 ============')
    ZDepth2 = calculate_depth('004964_left_disparity.png')
    box_depth2 = center(ZDepth2, '004964_left_disparity.png', img2_dict)

    type_dict2 = num_object(img2_dict)
    find_closest(img2_dict, box_depth2, type_dict2)

    # image3
    print('============ image3 ============')
    ZDepth3 = calculate_depth('005002_left_disparity.png')
    box_depth3 = center(ZDepth3, '005002_left_disparity.png', img3_dict)

    type_dict3 = num_object(img3_dict)
    find_closest(img3_dict, box_depth3, type_dict3)
