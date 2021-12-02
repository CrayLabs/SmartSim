import time
from os import environ

import numpy as np
import tensorflow.keras as keras

from smartredis import Client
from smartredis.error import RedisReplyError
from smartsim.ml import form_name


class StaticDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        batch_size=32,
        shuffle=True,
        uploader_info="auto",
        uploader_name="training_data",
        sample_prefix="samples",
        target_prefix="targets",
        sub_indices=None,
        num_classes=None,
        producer_prefixes=None,
        smartredis_cluster=True,
        smartredis_address=None,
        replica_rank=0,
        num_replicas=1,
        verbose=False,
        init_samples=True,
    ):
        self.client = Client(smartredis_address, smartredis_cluster)
        self.replica_rank = replica_rank
        self.num_replicas = num_replicas
        self.smartredis_address = smartredis_address
        self.smartredis_cluster = smartredis_cluster
        self.uploader_info = uploader_info
        self.uploader_name = uploader_name
        self.verbose = verbose
        self.samples = None
        self.targets = None
        self.num_samples = 0
        self.indices = np.arange(0)
        self.shuffle = shuffle
        self.batch_size = batch_size
        if uploader_info == "manual":
            self.sample_prefix = sample_prefix
            self.target_prefix = target_prefix
            self.sub_indices = sub_indices
            if producer_prefixes:
                self.producer_prefixes = list(producer_prefixes)
            else:
                producer_prefixes = [""]
            self.num_classes = num_classes
        elif uploader_info == "auto":
            if not uploader_name:
                raise ValueError(
                    "uploader_name can not be empty if uploader_info is 'auto'"
                )
            self.get_uploader_info(uploader_name)
        else:
            raise ValueError(
                f"uploader_info must be one of 'auto' or 'manual', but was {uploader_info}"
            )

        if init_samples:
            self.init_samples(None)


    def log(self, message):
        if self.verbose:
            print(message)

    def list_all_sources(self):
        uploaders = environ["SSKEYIN"].split(",")
        sources = []
        for uploader in uploaders:
            if any(
                [
                    uploader.startswith(producer_prefix)
                    for producer_prefix in self.producer_prefixes
                ]
            ):
                if self.sub_indices:
                    sources.extend(
                        [[uploader, sub_index] for sub_index in self.sub_indices]
                    )
                else:
                    sources.append([uploader, None])

        per_replica = len(sources) // self.num_replicas
        if per_replica > 0:
            if self.replica_rank < self.num_replicas - 1:
                sources = sources[
                    self.replica_rank
                    * per_replica : (self.replica_rank + 1)
                    * per_replica
                ]
            else:
                sources = sources[self.replica_rank * per_replica :]
        else:
            self.log(
                "Number of loader replicas is higher than number of sources, automatic split cannot be performed, "
                "all replicas will have the same dataset. If this is not intended, then implement a distribution strategy "
                "and modify `sources`."
            )

        return sources

    def init_samples(self, sources=None):
        self.autoencoding = self.sample_prefix == self.target_prefix

        if sources is not None:
            self.sources = sources

        if self.sources is None:
            self.sources = self.list_all_sources()

        self.log("Generator initialization complete")

        self.update_data()

    @property
    def need_targets(self):
        return self.target_prefix and not self.autoencoding

    def get_uploader_info(self, uploader_name):
        dataset_name = form_name(uploader_name, "info")
        self.log(f"Uploader dataset name: {dataset_name}")
        while not self.client.dataset_exists(dataset_name):
            time.sleep(10)

        uploader_info = self.client.get_dataset(dataset_name)
        self.sample_prefix = uploader_info.get_meta_strings("sample_prefix")[0]
        self.log(f"Uploader sample prefix: {self.sample_prefix}")

        try:
            self.target_prefix = uploader_info.get_meta_strings("target_prefix")[0]
        except:
            self.target_prefix = None
        self.log(f"Uploader target prefix: {self.target_prefix}")

        try:
            self.producer_prefixes = uploader_info.get_meta_strings("producer_prefix")
        except:
            self.producer_prefixes = [""]
        self.log(f"Uploader producer prefix: {self.producer_prefixes}")

        try:
            self.num_classes = uploader_info.get_meta_scalars("num_classes")[0]
        except:
            self.num_classes = None
        self.log(f"Uploader num classes: {self.num_classes}")

        try:
            self.sub_indices = uploader_info.get_meta_strings("sub_indices")
        except:
            self.sub_indices = None
        self.log(f"Uploader sub-indices: {self.sub_indices}")

    def __len__(self):
        length = int(np.floor(self.num_samples / self.batch_size))
        return length

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

    def __iter__(self):
    
        if not self.sources:
            pass
        else:
            # Generate data
            if len(self) < 1:
                raise ValueError(
                    "Not enough samples in generator for one batch. Please run init_samples() or initialize generator with init_samples=True"
                )

            for index in range(len(self)):
                indices = self.indices[
                    index * self.batch_size : (index + 1) * self.batch_size
                ]

                x, y = self.__data_generation(indices)

                if y is not None:
                    yield x, y
                else:
                    yield x


    def data_exists(self, batch_name, target_name):
        if self.need_targets:
            return self.client.tensor_exists(batch_name) and self.client.tensor_exists(
                target_name
            )
        else:
            return self.client.tensor_exists(batch_name)

    def add_samples(self, batch_name, target_name):
        if self.samples is None:
            self.samples = self.client.get_tensor(batch_name)
            if self.need_targets:
                self.targets = self.client.get_tensor(target_name)
        else:
            self.samples = np.concatenate(
                (self.samples, self.client.get_tensor(batch_name))
            )
            if self.need_targets:
                self.targets = np.concatenate(
                    (self.targets, self.client.get_tensor(target_name))
                )

        self.num_samples = self.samples.shape[0]
        self.indices = np.arange(self.num_samples)
        self.log("Success!")
        self.log(f"New dataset size: {self.num_samples}, batches: {len(self)}")

    def _update_samples_and_targets(self):
        for source in self.sources:
            entity = source[0]
            sub_index = source[1]
            self.client.set_data_source(entity)
            batch_name = form_name(self.sample_prefix, sub_index)
            if self.need_targets:
                target_name = form_name(self.target_prefix, sub_index)
            else:
                target_name = None

            self.log(f"Retrieving {batch_name} from {entity}")
            while not self.data_exists(batch_name, target_name):
                time.sleep(10)

            self.add_samples(batch_name, target_name)

    def update_data(self):
        self._update_samples_and_targets()

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


class DataGenerator(StaticDataGenerator):
    def __init__(
        self,
        batch_size=32,
        shuffle=True,
        uploader_info="auto",
        uploader_name="training_data",
        sample_prefix="samples",
        target_prefix="targets",
        sub_indices=None,
        num_classes=None,
        producer_prefixes=None,
        smartredis_cluster=True,
        smartredis_address=None,
        replica_rank=0,
        num_replicas=1,
        verbose=False,
        init_samples=True,
    ):
        super().__init__(
            batch_size,
            shuffle,
            uploader_info,
            uploader_name,
            sample_prefix,
            target_prefix,
            sub_indices,
            num_classes,
            producer_prefixes,
            smartredis_cluster,
            smartredis_address,
            replica_rank,
            num_replicas,
            verbose,
            init_samples,
        )


    def list_all_sources(self):
        sources = super().list_all_sources()
        # Append the batch index to each source
        for source in sources:
            source.append(0)
        return sources


    def init_samples(self, sources=None):
        self.autoencoding = self.sample_prefix == self.target_prefix
        if sources is None:
            self.sources = self.list_all_sources()

        while len(self) < 1:
            self._update_samples_and_targets()
            if len(self) < 1:
                time.sleep(10)
        self.log("Generator initialization complete")


    def _update_samples_and_targets(self):
        for source in self.sources:
            entity = source[0]
            sub_index = source[1]
            index = source[2]
            self.client.set_data_source(entity)
            batch_name = form_name(self.sample_prefix, index, sub_index)
            if self.need_targets:
                target_name = form_name(self.target_prefix, index, sub_index)
            else:
                target_name = None

            self.log(f"Retrieving {batch_name} from {entity}")
            # Poll next batch based on index, if available: retrieve it, update index and loop
            while self.data_exists(batch_name, target_name):
                self.add_samples(batch_name, target_name)
                source[2] += 1
                index = source[2]
                batch_name = form_name(self.sample_prefix, index, sub_index)
                if self.need_targets:
                    target_name = form_name(self.target_prefix, index, sub_index)

                self.log(f"Retrieving {batch_name}...")


    def on_epoch_end(self):
        self.update_data()
        super().on_epoch_end()