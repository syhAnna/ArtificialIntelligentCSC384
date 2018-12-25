import numpy as np

threshold = 0.65


def eliminate_dict(output_dict):
    scores = output_dict['detection_scores']

    index = []
    num_detection = 0
    for i in range(output_dict['num_detections']):
        score = scores[i]
        if score > threshold:
            index.append(i)
            num_detection += 1

    detection_boxes = []
    detection_scores = []
    detection_classes = []
    eliminated_dict = {}

    for i in range(len(index)):
        detection_boxes.append(output_dict['detection_boxes'][index[i]])
        detection_scores.append(output_dict['detection_scores'][index[i]])
        detection_classes.append(output_dict['detection_classes'][index[i]])

    eliminated_dict['num_detection'] = num_detection
    eliminated_dict['detection_boxes'] = np.array(detection_boxes)
    eliminated_dict['detection_scores'] = np.array(detection_scores)
    eliminated_dict['detection_classes'] = np.array(detection_classes)

    return eliminated_dict