from os import environ

import numpy as np

from smartredis import Client, Dataset
from smartredis.error import RedisReplyError

import time


def form_name(*args):
    return "_".join(str(arg) for arg in args if arg is not None)


class TrainingDataUploader:
    def __init__(
        self,
        name="training_data",
        sample_prefix="samples",
        target_prefix="targets",
        num_classes=None,
        producer_prefixes=None,
        smartredis_cluster=True,
        smartredis_address=None,
        sub_indices=None,
    ):
        if not name:
            raise ValueError("Name can not be empty")
        if not sample_prefix:
            raise ValueError("Sample prefix can not be empty")

        self.name = name
        self.sample_prefix = sample_prefix
        self.target_prefix = target_prefix
        self.producer_prefixes = producer_prefixes
        self.num_classes = num_classes
        if isinstance(sub_indices, int):
            self.sub_indices = [str(sub_idx) for sub_idx in range(sub_indices)]
        elif isinstance(sub_indices, list):
            self.sub_indices = [str(sub_idx) for sub_idx in sub_indices]
        elif sub_indices is None:
            self.sub_indices = None
        else:
            raise ValueError("sub_indices must be either list or int")

        self.client = Client(address=smartredis_address, cluster=smartredis_cluster)
        self.batch_idx = 0

    def publish_info(self):
        info_ds = Dataset(form_name(self.name, "info"))
        info_ds.add_meta_string("sample_prefix", self.sample_prefix)
        if self.target_prefix:
            info_ds.add_meta_string("target_prefix", self.target_prefix)
        if self.producer_prefixes:
            info_ds.add_meta_string("producer_prefixes", self.producer_prefixes)
        if self.num_classes:
            info_ds.add_meta_scalar("num_classes", self.num_classes)
        if self.sub_indices:
            for sub_index in self.sub_indices:
                info_ds.add_meta_string("sub_indices", sub_index)
        self.client.put_dataset(info_ds)

    def put_batch(self, samples, targets=None, sub_index=None):

        batch_key = form_name(self.sample_prefix, self.batch_idx, sub_index)
        self.client.put_tensor(batch_key, samples)
        print(f"Put batch {batch_key}")

        if (
            targets is not None
            and self.target_prefix
            and (self.target_prefix != self.sample_prefix)
        ):
            labels_key = form_name(self.target_prefix, self.batch_idx, sub_index)
            self.client.put_tensor(labels_key, targets)

        self.batch_idx += 1


class BatchDownloader:
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
    ):
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
        elif self.uploader_info == "auto":
            pass
        else:
            raise ValueError(
                f"uploader_info must be one of 'auto' or 'manual', but was {self.uploader_info}"
            )

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


    def init_sources(self, client):
        self.client = client
        if self.uploader_info == "auto":
            if not self.uploader_name:
                raise ValueError(
                    "uploader_name can not be empty if uploader_info is 'auto'"
                )
            self.get_uploader_info(self.uploader_name)
        elif self.uploader_info == "manual":
            pass
        else:
            raise ValueError(
                f"uploader_info must be one of 'auto' or 'manual', but was {self.uploader_info}"
            )

        self.sources = self.list_all_sources()


    def init_samples(self, sources=None):
        self.autoencoding = self.sample_prefix == self.target_prefix

        if sources is not None:
            self.sources = sources

        if self.sources is None:
            self.sources = self.list_all_sources()

        self.log("Generator initialization complete")

        self.update_data()

    def __iter__(self):
        
        if not self.sources:
            pass
        else:
            self.update_data()
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


    def __data_generation(self, indices):
        # Initialization
        x = self.samples[indices]

        if self.need_targets:
            y = self.targets[indices]

        elif self.autoencoding:
            y = x
        else:
            y = None

        return x, y