# %%
import math
import sys
import numpy as np
import re
import string
import random
from random import randint
import cv2
import os

from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# %%
# CONSTANTS
IM_PATH = "/home/fizzer/CNN_images"
NUM_LABELS = 4
CONFIDENCE_THRESHOLD = 0.01
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 1e-4

# %%
# Setup ref and read files
one_hot_ref = {
    'L': np.array([1.,0.,0.,0.]),
    'F': np.array([0.,1.,0.,0.]),
    'R': np.array([0.,0.,1.,0.]),
    'S': np.array([0.,0.,0.,1.])
}
# Read the images from the files 
dir_contents = np.array(os.listdir(IM_PATH))
np.random.shuffle(dir_contents)
im_set = []
for im_path in dir_contents:
    spl = im_path.split('_')
    x_vel = float(spl[0])
    z_vel = float(spl[1])
    #TODO: confirm labels
    if (x_vel > 0):
        label = one_hot_ref['F']
    elif (z_vel > 0):
        label = one_hot_ref['L']
    elif (z_vel < 0):
        label = one_hot_ref['R']
    else:
        label = one_hot_ref['S']
    # TODO: CHANGE THIS IF WE WANT COLOR
    # im_set.append([label, cv2.imread(f"{IM_PATH}/{im_path}")])
    im_set.append([label, cv2.cvtColor(cv2.imread(f"{IM_PATH}/{im_path}"), cv2.COLOR_RGB2GRAY)])
# im_set = np.array(im_set, dtype=object)

# %%
# Create x_dataset and y_dataset
x_dataset = np.array([img[1][360:] for img in im_set[:]])
y_dataset = np.array([img[0] for img in im_set[:]])
x_dataset = x_dataset/255.0
del(im_set)
# TODO: may need this x_dataset = x_dataset.reshape(len(x_dataset), len(x_dataset[0]), len(x_dataset[0][0]),-1)

# %%
print(x_dataset.shape)

# %%
# TODO: The following was copy pasted - validate
def reset_weights(model):
    for ix, layer in enumerate(model.layers):
        if (hasattr(model.layers[ix], 'kernel_initializer') and 
            hasattr(model.layers[ix], 'bias_initializer')):
            weight_initializer = model.layers[ix].kernel_initializer
            bias_initializer = model.layers[ix].bias_initializer

            old_weights, old_biases = model.layers[ix].get_weights()

            model.layers[ix].set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=len(old_biases))])

# %%
# Set up CNN
conv_model = models.Sequential()
conv_model.add(layers.Conv2D(3, (5, 5), activation='relu',
                        input_shape=(360, 1280, 1)))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(24, (5, 5), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(36, (5, 5), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(48, (5, 5), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Conv2D(64, (5, 5), activation='relu'))
conv_model.add(layers.MaxPooling2D((2, 2)))
conv_model.add(layers.Flatten())
conv_model.add(layers.Dropout(0.5))
conv_model.add(layers.Dense(512, activation='relu'))
conv_model.add(layers.Dense(50, activation='relu'))
conv_model.add(layers.Dense(10, activation='relu'))
conv_model.add(layers.Dense(4, activation='softmax'))
conv_model.compile(loss='mse', optimizer=optimizers.RMSprop(learning_rate=LEARNING_RATE), metrics=['acc'])

# %%
# reset_weights(conv_model)

# %%
# Train CNN
begin = 0
block_length = len(x_dataset) // 5
for i in range(block_length, len(x_dataset), block_length):
    history_conv = conv_model.fit(x_dataset[begin:block_length], y_dataset[begin:block_length], 
                                validation_split=VALIDATION_SPLIT, 
                                epochs=20, 
                                batch_size=8)
    x_dataset = x_dataset[block_length:]
    y_dataset = y_dataset[block_length:]

# %%



