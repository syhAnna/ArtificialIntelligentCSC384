from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random


# loading the data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# store class name for future use
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# show a bar graph of number of instance in each class
class_dict = {}
for i in range(len(train_labels)):
    if train_labels[i] in class_dict:
        class_dict[train_labels[i]] += 1
    else:
        class_dict[train_labels[i]] = 1

# show a bar graph of the number of instance in each class
class_lst = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
num_instance = []
for i in range(len(class_lst)):
    num_instance.append(class_dict[class_lst[i]])

width = 1/1.5

plt.bar(class_lst, num_instance, width, color="blue")

# split the dataset into training set (70) and validation set (30)
# 1. according to the training label, split the training data into the 10 classes
split_class = {}  # dict of {class_num: [list of img]}
for i in range(len(train_labels)):
    if train_labels[i] in split_class:
        split_class[train_labels[i]].append(train_images[i])
    else:
        split_class[train_labels[i]] = [train_images[i]]

# 2. for each class, first random the image data in the list, then split to 7:3
img_class = split_class.values()  # list of lists of data
training_set, validation_set = [], []
for item_class in img_class:
    random.shuffle(item_class)
    training_set.extend(item_class[:4200])
    validation_set.extend(item_class[4200:])

# build up the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20, batch_size=16, validation_split=0.3)
