
# take code for VAE from Keras https://keras.io/examples/generative/vae/
from tensorflow import keras

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, \
                                    Add, SpatialDropout2D, Layer, InputLayer
from tensorflow.keras.models import Model

import tensorflow as tf

activation = "selu"
padding = "same"


# Next function initially taken from
# https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
class ResBlock(Layer):
    def __init__(self, downsample: bool, filters: int, kernel_size: int = 3):
        super(ResBlock, self).__init__()
        self.conv1 = Conv2D(kernel_size=kernel_size,
                            strides= (1 if not downsample else 2),
                            filters=filters,
                            padding=padding,
                            activation=activation)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(kernel_size=kernel_size,
                            strides=1,
                            filters=filters,
                            padding=padding)

        if downsample:
            self.conv3 = Conv2D(kernel_size=1,
                                strides=2,
                                filters=filters,
                                padding=padding)
        else:
            self.conv3 = Conv2D(kernel_size=1,
                                strides=1,
                                filters=filters,
                                padding=padding)


        self.bn2 = BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        y = self.conv3(inputs)
        z = Add()([x, y])
        z = keras.activations.selu(z)
        z = self.bn2(z)
        return z

# Next function initially taken from
# https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
class ResBlockTranspose(Layer):
    def __init__(self, downsample: bool, filters: int, kernel_size: int = 3):
        super(ResBlockTranspose, self).__init__()
        self.conv1 = Conv2DTranspose(kernel_size=kernel_size,
                            strides=(1 if not downsample else 2),
                            filters=filters,
                            padding=padding,
                            activation=activation)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2DTranspose(kernel_size=kernel_size,
                            strides=1,
                            filters=filters,
                            padding=padding)

        if downsample:
            self.conv3 = Conv2DTranspose(kernel_size=1,
                                strides=2,
                                filters=filters,
                                padding=padding)
        else:
            self.conv3 = Conv2DTranspose(kernel_size=1,
                                strides=1,
                                filters=filters,
                                padding=padding)


        self.bn2 = BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        y = self.conv3(inputs)
        z = Add()([x, y])
        z = keras.activations.selu(z)
        z = self.bn2(z)
        return z

class DiffusionResNet(Model):
    def __init__(self, sample_shape, depth=4, downsample=True):
        super().__init__(name = f"DiffusionResNet_{depth}")
        filters = sample_shape[-1]
        self.skips = [Conv2D(filters, 3, (1,1), padding="same", activation=activation)]
        self.res_blocks = []
        self.dropout1 = SpatialDropout2D(0.4)
        for i in range(depth):
            filters *= 2
            self.res_blocks.append(ResBlock(downsample, filters, 3))
            if i < depth-1:
                self.skips.append(Conv2D(filters, 3, (1,1), padding="same", activation=activation))
            else:
                self.skips.append(None)
        self.res_blocks_tr = []
        for _ in range(depth):
            filters /= 2
            self.res_blocks_tr.append(ResBlockTranspose(downsample, filters, 3))
        self.dropout2 = SpatialDropout2D(0.4)
        self.bn = BatchNormalization()

    def call(self, inputs):
        x = self.res_blocks[0](inputs)
        res_outs = [self.dropout1(x)]
        for i in range(1,len(self.res_blocks)):
            res_outs.append(self.res_blocks[i](res_outs[-1]))
        x = self.res_blocks_tr[0](res_outs[-1])
        for i in range(1,len(self.res_blocks_tr)):
            y = self.skips[-i-1](res_outs[-i-1])
            x = Add()([x, y])
            x = keras.activations.selu(x)
            x = self.res_blocks_tr[i](x)
        x = Add()([self.skips[0](inputs),x])
        x = keras.activations.selu(x)
        x = self.dropout2(x)
        x = self.bn(x)
        return x


