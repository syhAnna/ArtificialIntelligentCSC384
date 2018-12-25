# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0


y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
bar_graph_dict = {}

# ----- plotting bar graph
# print(train_labels)
# for label in train_labels:
#     if label in bar_graph_dict.keys():
#         bar_graph_dict[label] += 1
#     else:
#         bar_graph_dict[label] = 1
#
# bar_graph_list = [bar_graph_dict[label] for label in bar_graph_dict.keys()]
# x = range(len(bar_graph_list))
# width = 1/1.5
# plt.bar(x, bar_graph_list, width, color="yellow")
# plt.show()
# ----- plotting bar graph


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


history = model.fit(train_images, train_labels, epochs=15, batch_size=16, validation_split=0.3)

# -----plot training and validation loss and acc------
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# ------plot training and validation loss and acc------

# ------ changing batch size ------
n_batch = [8, 16, 32, 64]
for i in n_batch:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=20, batch_size=i, validation_split=0.3)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('batch_size')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('batch_size')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    model.reset_states()
# ------ changing batch size ------
train_loss, train_acc = model.evaluate(train_images, train_labels)
test_loss, test_acc = model.evaluate(test_images, test_labels)
#
print('Test accuracy:', test_acc)



# changing batch size
n_batch = [8, 16, 32, 64]
for i in n_batch:
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=20, batch_size=i, validation_split=0.3)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('batch_size')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('batch_size')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    model.reset_states()

train_loss, train_acc = model.evaluate(train_images, train_labels)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)