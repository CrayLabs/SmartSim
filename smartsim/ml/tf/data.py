import numpy as np
import tensorflow.keras as keras

from smartsim.ml import DynamicDataDownloader, StaticDataDownloader


class StaticDataGenerator(StaticDataDownloader, keras.utils.Sequence):
    """A class to download a dataset from the DB.

    Details about parameters and features of this class can be found
    in the documentation of ``StaticDataDownloader``, of which it is just
    a TensorFlow-specialized sub-class.
    """

    def __init__(self, **kwargs):
        StaticDataDownloader.__init__(self, **kwargs)

    def __getitem__(self, index):
        if len(self) < 1:
            msg = "Not enough samples in generator for one batch. "
            msg += "Please run init_samples() or initialize generator with init_samples=True"
            raise ValueError(msg)
        # Generate indices of the batch
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indices)

        if y is not None:
            return x, y
        else:
            return x

    def on_epoch_end(self):
        """Callback called at the end of each training epoch

        If `self.shuffle` is set to `True`, data is shuffled.
        """
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


class DynamicDataGenerator(DynamicDataDownloader, StaticDataGenerator):
    """A class to download batches from the DB.

    Details about parameters and features of this class can be found
    in the documentation of ``DynamicDataDownloader``, of which it is just
    a TensorFlow-specialized sub-class.
    """

    def __init__(self, **kwargs):
        StaticDataGenerator.__init__(self, **kwargs)

    def on_epoch_end(self):
        """Callback called at the end of each training epoch

        Update data (the DB is queried for new batches) and
        if `self.shuffle` is set to `True`, data is also shuffled.
        """
        self.update_data()
        super().on_epoch_end()
