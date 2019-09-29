# coding: utf-8
# 神经网络搭建与训练
# by z0gSh1u @ https://github.com/z0gSh1u

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils, plot_model
import keras

from dataset_loader import load_dataset

# SignNet definition
class SignNet:
  @staticmethod
  def build(input_shape, classes=6):
    model = Sequential()
    # Block 1
    model.add(
      Conv2D(16,
        kernel_size=4,
        padding='same',
        input_shape=input_shape,
        strides=(1, 1)
      )
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(8, 8), strides=(8, 8), padding='same'))
    # Block 2
    model.add(
      Conv2D(48,
        kernel_size=2,
        padding='same',
        input_shape=input_shape,
        strides=(1, 1)
      )
    )
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
    # Block 3
    model.add(Flatten())
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model

# Hyper Params
EPOCH = 200
BATCH_SIZE = 64
LR = 0.001
OPTIMIZER = Adam(lr=LR)
LOSSFUNC = 'categorical_crossentropy'
VALIDATION_SPLIT = 0.1
# Other Params
IMG_ROWS, IMG_COLS = 64, 64
CLASSES = 6
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)

# Load dataset
x_train, y_train, x_test, y_test, _ = load_dataset()
# Normalization
x_train = x_train / 255; x_test = x_test / 255
# One-hot
y_train = np_utils.to_categorical(y_train, CLASSES); y_test = np_utils.to_categorical(y_test, CLASSES)

# Build model
model = SignNet.build(INPUT_SHAPE, CLASSES)
model.compile(optimizer=OPTIMIZER, loss=LOSSFUNC, metrics=['accuracy'])
model.summary()

# === For TensorBoard usage, uncomment them if you need
# cb_tf = keras.callbacks.TensorBoard(write_images=1, histogram_freq=1)
# cbks = [cb_tf]

# === For model plotting usage, uncomment them if you need
# plot_model(model, show_shapes=True)

history = model.fit(
  x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=VALIDATION_SPLIT, shuffle=True#, callbacks=cbks
)
score = model.evaluate(x_test, y_test)

# === Save trained model, uncomment them if you need
# model.save('SignNet.h5')

print('Test score: ', score[0])
print('Test accuracy: ', score[1])
# === Plotting, uncomment them if you need
# print('---- History ----')
# print('History keys: ', history.history.keys())
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('(SignNet) Model Accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'])
# plt.savefig('Accuracy.png')
# plt.show()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('(SignNet) Model Loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'])
# plt.savefig('Loss.png')
# plt.show()