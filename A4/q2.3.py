import cv2 as cv


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


def visualize(img, img_dict):
    img = cv.imread(img)
    height, width = img.shape[0], img.shape[1]
    for i in range(img_dict['num_detection']):
        box = img_dict['detection_boxes'][i]
        pt1 = (int(box[1] * width), int(box[0] * height))
        pt2 = (int(box[3] * width), int(box[2] * height))
        classes = img_dict['detection_classes'][i]

        if classes == 1:
            # person, blue (b, g, r)
            cv.rectangle(img, pt1, pt2, color=(255, 0, 0), thickness=3)
            cv.putText(img, 'Person', org=pt1, color=(255, 255, 255), fontScale=1, fontFace=1, thickness=2)
        elif classes == 2:
            # bicycle, green
            cv.rectangle(img, pt1, pt2, color=(0, 255, 0), thickness=3)
            cv.putText ( img, 'Bicycle', org=pt1, color=(255, 255, 255), fontScale=1, fontFace=1, thickness=2)
        elif classes == 3:
            # car, red
            cv.rectangle(img, pt1, pt2, color=(0, 0, 255), thickness=3)
            cv.putText(img, 'Car', org=pt1, color=(255, 255, 255), fontScale=1, fontFace=1, thickness=2)
        elif classes == 10:
            # traffic_light, cyan
            cv.rectangle(img, pt1, pt2, color=(255, 255, 0), thickness=3)
            cv.putText(img, 'Traffic Light', org=pt1, color=(255, 255, 255), fontScale=1, fontFace=1, thickness=2)
    return img


if __name__ == '__main__':
    cv.imwrite('004945q2_3.jpg', visualize('004945.jpg', img1_dict))
    cv.imwrite('004964q2_3.jpg', visualize('004964.jpg', img2_dict))
    cv.imwrite('005002q2_3.jpg', visualize('005002.jpg', img3_dict))
