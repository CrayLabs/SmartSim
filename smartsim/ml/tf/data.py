# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import typing as t
from tensorflow import keras

from smartsim.ml import DataDownloader


class _TFDataGenerationCommon(DataDownloader, keras.utils.Sequence):
    def __getitem__(
        self, index: int
    ) -> t.Tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        if len(self) < 1:
            raise ValueError(
                "Not enough samples in generator for one batch. Please "
                "run init_samples() or initialize generator with init_samples=True"
            )
        # Generate indices of the batch
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]

        # Generate data
        xval, yval = self._data_generation(indices)

        if yval is not None:
            return xval, yval

        return xval

    def on_epoch_end(self) -> None:
        """Callback called at the end of each training epoch

        If `self.shuffle` is set to `True`, data is shuffled.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_generation(self, indices: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        # Initialization
        if self.samples is None:
            raise ValueError("No samples loaded for data generation")

        xval = self.samples[indices]

        if self.need_targets:
            yval = self.targets[indices]
            if self.num_classes is not None:
                yval = keras.utils.to_categorical(yval, num_classes=self.num_classes)
        elif self.autoencoding:
            yval = xval
        else:
            return xval

        return xval, yval


class StaticDataGenerator(_TFDataGenerationCommon):
    """A class to download a dataset from the DB.

    Details about parameters and features of this class can be found
    in the documentation of ``DataDownloader``, of which it is just
    a TensorFlow-specialized sub-class with dynamic=False.
    """

    def __init__(self, **kwargs: t.Any) -> None:
        dynamic = kwargs.pop("dynamic", False)
        kwargs["dynamic"] = False
        super().__init__(**kwargs)
        if dynamic:
            msg = (
                "Static data generator cannot be started with dynamic=True, "
                "setting it to False"
            )
            self.log(msg)


class DynamicDataGenerator(_TFDataGenerationCommon):
    """A class to download batches from the DB.

    Details about parameters and features of this class can be found
    in the documentation of ``DataDownloader``, of which it is just
    a TensorFlow-specialized sub-class with dynamic=True.
    """

    def __init__(self, **kwargs: t.Any) -> None:
        dynamic = kwargs.pop("dynamic", True)
        kwargs["dynamic"] = True
        super().__init__(**kwargs)
        if not dynamic:
            msg = (
                "Dynamic data generator cannot be started with dynamic=False,"
                " setting it to True"
            )
            self.log(msg)

    def on_epoch_end(self) -> None:
        """Callback called at the end of each training epoch

        Update data (the DB is queried for new batches) and
        if `self.shuffle` is set to `True`, data is also shuffled.
        """
        self.update_data()
