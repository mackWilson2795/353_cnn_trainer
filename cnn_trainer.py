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
        elif (z_vel < 0):
            label = one_hot_ref['L']
        elif (z_vel > 0):
            label = one_hot_ref['R']
        else:
            label = one_hot_ref['S']
        im_set = np.append(im_set, cv2.imread(f"{IM_PATH}{im_path}"))


if __name__ == '__main__':
    main(sys.argv)