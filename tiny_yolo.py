from keras.models import Sequential, Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.merge import concatenate
import keras.backend as K



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

    # input_image = Input(shape=(IMAGE_H, IMAGE_W, 3))
    # # true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))

    # # Conv 1
    # x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', 
            # use_bias=False)(input_image)
    # x = BatchNormalization(name='norm_1')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    # # Conv 2
    # x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_2', 
            # use_bias=False)(x)
    # x = BatchNormalization(name='norm_2')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    # # Conv 3
    # x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_3', 
            # use_bias=False)(x)
    # x = BatchNormalization(name='norm_3')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    # # Conv 4
    # x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_4', 
            # use_bias=False)(x)
    # x = BatchNormalization(name='norm_4')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    # # Conv 5
    # x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_5', 
            # use_bias=False)(x)
    # x = BatchNormalization(name='norm_5')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    # # Conv 6
    # x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', 
            # use_bias=False)(x)
    # x = BatchNormalization(name='norm_6')(x)
    # x = LeakyReLU(alpha=0.1)(x)
    # x = MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same')(x)

    # # Conv 7
    # x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_7', 
            # use_bias=False)(x)
    # x = BatchNormalization(name='norm_7')(x)
    # x = LeakyReLU(alpha=0.1)(x)

    # # Conv 8
    # x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_8', 
            # use_bias=False)(x)
    # x = BatchNormalization(name='norm_8')(x)
    # x = LeakyReLU(alpha=0.1)(x)

    # # Conv 9
    # x = Conv2D(BOX * (4 + 1 + CLASS), (3,3), strides=(1,1), padding='same', 
            # name='conv_9', use_bias=False)(x)
    # # x = BatchNormalization(name='norm_9')(x)
    # # x = LeakyReLU(alpha=0.1)(x)

    # # output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)
    # output = x

    # model = Model(input_image, output)

    # if print_model_summary:
        # model.summary()

    # return model


    model = Sequential()

    # Layer 1
    model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, input_shape=(416,416,3)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2 - 5
    for i in range(0,4):
        model.add(Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', use_bias=False))
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
        model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))

    # Layer 9
    model.add(Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))
    model.add(Activation('linear'))
    model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS)))

    return model


