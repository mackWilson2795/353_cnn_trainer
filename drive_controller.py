from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
import cv2
import numpy as np
from geometry_msgs.msg import Twist

class driver_controller:
    CP_PATH = "/home/fizzer/cnn_trainer/model_cps/"
    SAVE_PATH = "/home/fizzer/cnn_trainer/model_save/"
    MODEL_X = 180
    MODEL_Y = 320
    LEARNING_RATE = 1e-4
    IMG_DOWNSCALE_RATIO = 0.25

    def __init__(self, save_path = SAVE_PATH) -> None:
        one_hot_ref = {
            np.array([1.,0.,0.,0.]): 'L',
            np.array([0.,1.,0.,0.]): 'F',
            np.array([0.,0.,1.,0.]): 'R',
            np.array([0.,0.,0.,1.]): 'S'
        }
        conv_model = models.load_model(save_path)

    def drive(self, img):
        img  = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (0,0),
                            fx=self.IMG_DOWNSCALE_RATIO, fy=self.IMG_DOWNSCALE_RATIO)
        prediction = self.conv_model.predict(img)
        move = Twist()
        if self.one_hot_ref.get(prediction) == 'L':
            move.linear.x = 0.0
            move.angular.z = 1.0
        elif self.one_hot_ref.get(prediction) == 'F':
            move.linear.x = 0.5
            move.angular.z = 0.0
        elif self.one_hot_ref.get(prediction) == 'R':
            move.linear.x = 0.0
            move.angular.z -1.0
        else:
            move.linear.x = 0.0
            move.angular.z = 0.0
        return move
        
    def __load_weights(self):
        

    def __save_network(self):
        # TODO: do
        1
    def __create_model(self):
        conv_model = models.Sequential()
        conv_model.add(layers.Conv2D(3, (5, 5), activation='relu',
                                input_shape=(self.MODEL_X, self.MODEL_Y, 1)))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(24, (5, 5), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(36, (5, 5), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(48, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        conv_model.add(layers.MaxPooling2D((2, 2)))
        conv_model.add(layers.Flatten())
        conv_model.add(layers.Dropout(0.5))
        conv_model.add(layers.Dense(512, activation='relu'))
        conv_model.add(layers.Dense(50, activation='relu'))
        conv_model.add(layers.Dense(10, activation='relu'))
        conv_model.add(layers.Dense(4, activation='softmax'))
        conv_model.compile(loss='mse', optimizer=optimizers.RMSprop(learning_rate=self.LEARNING_RATE), metrics=['acc'])
        return conv_model
