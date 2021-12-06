import time
from os import environ

import numpy as np
import tensorflow.keras as keras

from smartredis import Client
from smartredis.error import RedisReplyError
from smartsim.ml import form_name
from smartsim.ml.data import BatchDownloader, ContinuousBatchDownloader


class StaticDataGenerator(BatchDownloader, keras.utils.Sequence):
    def __init__(
        self,
        **kwargs
    ):
        BatchDownloader.__init__(self, **kwargs)
        

    def __getitem__(self, index):
        if len(self) < 1:
            raise ValueError(
                "Not enough samples in generator for one batch. Please run init_samples() or initialize generator with init_samples=True"
            )
        # Generate indices of the batch
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indices)

        if y is not None:
            return x, y
        else:
            return x


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


    def __data_generation(self, indices):
        # Initialization
        x = self.samples[indices]

        if self.need_targets:
            y = self.targets[indices]
            if self.num_classes is not None:
                y = keras.utils.to_categorical(y, num_classes=self.num_classes)
        elif self.autoencoding:
            y = x
        else:
            y = None

        return x, y


class DataGenerator(ContinuousBatchDownloader, StaticDataGenerator):
    def __init__(
        self,
        **kwargs
    ):
        StaticDataGenerator.__init__(self, **kwargs)


    def __data_generation(self, indices):
        return StaticDataGenerator.__data_generation(self, indices)


    def on_epoch_end(self):
        self.update_data()
        super().on_epoch_end()