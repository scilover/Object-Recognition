from tensorflow.python.keras.layers import *
from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras import Model
from tensorflow.keras import utils
import tensorflow as tf


class Yolo:

    def __init__(self, input_shape, grid_size):
        self.grid_size = grid_size
        self.input_shape = input_shape

    # def pad(self):
    #
    #     dh, dw = self.grid_size
    #     pad_width = int((224 - dw) / 2)
    #     pad_height = int((224 - dh) / 2)
    #
    #     return ZeroPadding2D((pad_height, pad_width))


    def model(self):

        m, n = self.input_shape
        dh, dw = self.grid_size
        h, w = int(m/dh), int(n/dw)
        inputs = Input(shape=(*self.input_shape, 3), name='inputs')
        x = densenet.preprocess_input(inputs)
        # x = Dropout(rate=0.1)(x)
        x = densenet.DenseNet(blocks=[6,12,24,16],
                            include_top=False,
                            weights=None,
                            input_tensor=None,
                            input_shape=(*self.input_shape,3),
                            pooling=None)(x)
        x = tf.image.resize(x, (h, w))
        spp1 = MaxPooling2D(3, strides=1)(x)
        spp1 = tf.image.resize(spp1, x.shape[1:3])

        spp2 = MaxPooling2D(5, strides=1)(x)
        spp2 = tf.image.resize(spp2, x.shape[1:3])

        spp3 = MaxPooling2D(7, strides=1)(x)
        spp3 = tf.image.resize(spp3, x.shape[1:3])

        spp4 = MaxPooling2D(8, strides=1)(x)
        spp4 = tf.image.resize(spp4, x.shape[1:3])
        # spp5 = MaxPooling2D(9, strides=1)(x)
        # spp5 = tf.image.resize(spp5, x.shape[1:3])
        x = Concatenate(axis=-1)([x, spp1, spp2, spp3, spp4])
        exist = Dense(1, activation='sigmoid', name='exist')(x)
        category = Dense(80, activation='softmax', name='category')(x)
        location = Dense(2, activation='sigmoid', name='location')(x)
        shape = Dense(2, activation='relu', name='shape')(x)
        # shape = Dense(2, name='shape')(x)
        # shape = tf.keras.layers.LeakyReLU(alpha=0.1)(shape)
        x = Concatenate(axis=-1)([exist, category, location, shape])

        return Model(inputs, x)


if __name__ == '__main__':

    model = Yolo((320, 320), grid_size=(16,16)).model()
    # img_tensor = tools.img_to_tensor(r'G:\Data\yoloV4自己的数据集\JPEGImages\010000.jpg')
    # img_tensor = img_tensor[tf.newaxis, ...]

    model.summary()
    utils.plot_model(model, show_shapes=True, show_layer_names=True)

