#! /usr/bin/env python3

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

def main(args):
    IM_PATH = '~/CNN_images/'
    NUM_LABELS = 4
    CONFIDENCE_THRESHOLD = 0.01
    VALIDATION_SPLIT = 0.2
    LEARNING_RATE = 1e-4
    one_hot_ref = {
        'L': np.array([1.,0.,0.,0.]),
        'F': np.array([0.,1.,0.,0.]),
        'R': np.array([0.,0.,1.,0.]),
        'S': np.array([0.,0.,0.,1.])
    }
    # Read the images from the files 
    dir_contents = np.array(os.listdir(IM_PATH))
    np.random.shuffle(dir_contents)
    im_set = np.array()
    for im_path in dir_contents:
        spl = im_path.split('_')
        x_vel = int(spl[0])
        z_vel = int(spl[1])
        #TODO: confirm labels
        if (x_vel > 0):
            label = one_hot_ref['F']
        elif (z_vel > 0):
            label = one_hot_ref['L']
        elif (z_vel < 0):
            label = one_hot_ref['R']
        else:
            label = one_hot_ref['S']
        im_set = np.append(im_set, (label, cv2.imread(f"{IM_PATH}{im_path}")))
        x_dataset = np.array([img[1] for img in im_set[:]])
        y_dataset = np.array([img[0] for img in im_set])
        x_dataset = x_dataset/255.0
        # TODO: may need this x_dataset = x_dataset.reshape(len(x_dataset), len(x_dataset[0]), len(x_dataset[0][0]),-1)

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
        
        conv_model = models.Sequential()
        conv_model.add(layers.Conv2D(32, (3, 3), activation='relu',
                             input_shape=(120, 100, 1)))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Flatten())
        conv_model.add(layers.Dropout(0.5))
        conv_model.add(layers.Dense(512, activation='relu'))
        conv_model.add(layers.Dense(36, activation='softmax'))
        conv_model.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
                   metrics=['acc'])

        # reset_weights(conv_model)

        history_conv = conv_model.fit(X_dataset, Y_dataset, 
                              validation_split=VALIDATION_SPLIT, 
                              epochs=80, 
                              batch_size=16)


if __name__ == '__main__':
    main(sys.argv)