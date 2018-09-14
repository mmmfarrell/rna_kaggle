"""
This model was adapted from
https://github.com/joycex99/tiny-yolo-keras/blob/master/utils.py
"""

from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization 
from keras.layers.advanced_activations import LeakyReLU



def make_tiny_yolo_model(config, print_model_summary=False):
    """
    Create a Keras model for tiny YOLO

    Tiny YOLOv2 has 9 convolutional layers.
    """

    IMAGE_H = config['IMAGE_H']
    IMAGE_W = config['IMAGE_W']
    CLASS = config['CLASS']
    BOX = config['BOX']
    GRID_H = config['GRID_H']
    GRID_W = config['GRID_W']

    model = Sequential()

    # Layer 1
    model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False,
        input_shape=(IMAGE_H, IMAGE_W, 3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2 - 5
    for i in range(0,4):
        model.add(Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', 
            use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 6
    model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same'))

    # Layer 7 - 8
    for _ in range(0,2):
        model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same',
            use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

    # Layer 9
    model.add(Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1),
        kernel_initializer='he_normal'))
    model.add(Activation('linear'))
    model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS)))

    return model


