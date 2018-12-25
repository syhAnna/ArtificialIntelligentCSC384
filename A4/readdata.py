def read_data(filename):
    img_dict = {}
    file = open(filename, 'r')

    num_detection = int(file.readline())
    img_dict['num_detection'] = num_detection
    img_dict['detection_boxes'] = []
    img_dict['detection_scores'] = []
    img_dict['detection_classes'] = []

    # read box
    for i in range(num_detection):
        line_box = file.readline()
        box = line_box.split (',')
        for j in range(4):
            box[j] = float(box[j])
        img_dict['detection_boxes'].append(box)

    # read score
    for i in range(num_detection):
        line_score = file.readline()
        img_dict['detection_scores'].append(float(line_score))

    # read type
    for i in range(num_detection):
        line_type = file.readline()
        img_dict['detection_classes'].append(int(line_type))

    return img_dict


if __name__ == '__main__':
    img_dict1 = read_data('datafile_dict1.txt')
    print(img_dict1)
    print('\n')
    img_dict2 = read_data('datafile_dict2.txt')
    print(img_dict2)
    print ( '\n' )
    img_dict3 = read_data('datafile_dict3.txt')
    print(img_dict3)
