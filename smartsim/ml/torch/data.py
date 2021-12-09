import time
from os import environ

import numpy as np
import torch

from smartsim.ml.data import BatchDownloader, ContinuousBatchDownloader


class StaticDataGenerator(BatchDownloader, torch.utils.data.IterableDataset):
    """A class to download batches from the DB.

    By default, the StaticDataGenerator has to be created in a process
    launched through SmartSim, with sample producers listed as incoming
    entities.

    All details about the batches must be defined in
    the constructor; two mechanisms are available, `manual` and
    `auto`.

     - When specifying `auto`, the user must also specify
      `uploader_name`. BatchDownloader will get all needed information
      from the database (this expects a Dataset like the one created
      by TrainingDataUploader to be available and stored as `uploader_name`
      on the DB).

     - When specifying `manual`, the user must also specify details
       of batch naming. Specifically, for each incoming entity with
       a name starting with an element of `producer_prefixes`,
       BatchDownloader will query the DB
       for all batches named <sample_prefix>_<sub_index> for all indices
       in `sub_indexes` if supplied, and, if
       `target_prefix` is supplied, it will also query for all targets
       named <target_prefix>.<sub_index>. If `producer_prefixes` is
       None, then all incoming entities will be treated as producers,
       and for each one, the corresponding batches will be downloaded.

    The flag `init_samples` defines whether sources (the list of batches
    to be fetched) and samples (the actual data) should automatically
    be set up in the costructor.

    Note that if the ``StaticDataGenerator`` has to be used through a ``DataLoader``,
    `init_samples` must be set to `False`, as sources and samples will be initialized
    by the ``DataLoader`` workers.

    If the user needs to modify the list of sources, then `init_samples=False`
    has to be set. In that case, to set up a `StaticDataGenerator`, the user has to call
    `init_sources()` (which initializes the list of sources and the SmartRedis client)
    and `init_samples()`.  After `init_sources()` is called,
    a list of data sources is populated, representing the batches which
    will be downloaded.

    Each source is represented as a tuple `(producer_name, sub_index)`.
    Before `init_samples()` is called, the user can modify the list.
    Once `init_samples()` is called, all data is downloaded and batches
    can be obtained with iter().

    After initialization, samples and targets will not be updated. The data can
    be shuffled by calling `update_data()`, if `shuffle` is set to ``True`` at
    initialization.

    :param batch_size: Size of batches obtained with __iter__
    :type batch_size: int
    :param shuffle: whether order of samples has to be shuffled when calling `update_data`
    :type shuffle: bool
    :param uploader_info: Set to `auto` uploader information has to be downloaded from DB,
                          or to `manual` if it is provided by the user
    :type uploader_info: str
    :param uploader_name: Name of uploader info dataset, only used if `uploader_info` is `auto`
    :type uploader_name: str
    :param sample_prefix: prefix of keys representing batches
    :type sample_prefix: str
    :param target_prefix: prefix of keys representing targets
    :type target_prefix: str
    :param sub_indices: Sub indices of the batches. This is useful in case each producer
                        has multiple ranks and each rank produces batches. Each
                        rank will then need to use a different sub-index, which is an element
                        of the `sub_indices`. If an integer is specified for `sub_indices`,
                        then it is converted to `range(sub_indices)`.
    :type sub_indices: int or list
    :param num_classes: Number of classes of targets, if categorical
    :type num_classes: int
    :param producer_prefixes: Prefixes of processes which will be producing batches.
                              This can be useful in case the consumer processes also
                              have other incoming entities.
    :type producer_prefixes: str
    :param smartredis_cluster: Whether the Orchestrator will be run as a cluster
    :type smartredis_cluster: bool
    :param smartredis_address: Address of Redis client as <ip_address>:<port>
    :type smartredis_address: str
    :param replica_rank: When BatchDownloader is used distributedly, indicates
                         the rank of this object
    :type replica_rank: int
    :param num_replicas: When BatchDownlaoder is used distributedly, indicates
                         the total number of ranks
    :type num_replicas: int
    :param verbose: Whether log messages should be printed
    :type verbose: bool
    :param init_samples: whether samples should be initialized in the constructor
    :type init_samples: bool
    """

    def __init__(self, **kwargs):
        BatchDownloader.__init__(self, **kwargs)

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
            self.samples = torch.tensor(self.client.get_tensor(batch_name))
            if self.need_targets:
                self.targets = torch.tensor(self.client.get_tensor(target_name))
        else:
            self.samples = torch.cat(
                (self.samples, torch.tensor(self.client.get_tensor(batch_name)))
            )
            if self.need_targets:
                self.targets = torch.cat(
                    (self.targets, torch.tensor(self.client.get_tensor(target_name)))
                )

        self.num_samples = self.samples.shape[0]
        self.indices = np.arange(self.num_samples)
        self.log("Success!")
        self.log(f"New dataset size: {self.num_samples}")

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
        if self.shuffle:
            np.random.shuffle(self.indices)

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

class DataGenerator(ContinuousBatchDownloader, StaticDataGenerator):
    """A class to download batches from the DB.

    By default, the DataGenerator has to be created in a process
    launched through SmartSim, with sample producers listed as incoming
    entities.

    All details about the batches must be defined in
    the constructor; two mechanisms are available, `manual` and
    `auto`.

     - When specifying `auto`, the user must also specify
      `uploader_name`. DataGenerator will get all needed information
      from the database (this expects a Dataset like the one created
      by TrainingDataUploader to be available and stored as `uploader_name`
      on the DB).

     - When specifying `manual`, the user must also specify details
       of batch naming. Specifically, for each incoming entity with
       a name starting with an element of `producer_prefixes`,
       BatchDownloader will query the DB
       for all batches named <sample_prefix>_<sub_index>_<iteration> for all indices
       in `sub_indices` if supplied, and, if
       `target_prefix` is supplied, it will also query for all targets
       named <target_prefix>.<sub_index>.<iteration>. If `producer_prefixes` is
       None, then all incoming entities will be treated as producers,
       and for each one, the corresponding batches will be downloaded.

    The flag `init_samples` defines whether sources (the list of batches
    to be fetched) and samples (the actual data) should automatically
    be set up in the costructor.

    Note that if the ``DataGenerator`` has to be used through a ``DataLoader``,
    `init_samples` must be set to `False`, as sources and samples will be initialized
    by the ``DataLoader`` workers.

    If the user needs to modify the list of sources, then `init_samples=False`
    has to be set. In that case, to set up a `BatchDownlaoder`, the user has to call
    `init_sources()` (which initializes the list of sources and the SmartRedis client)
    and `init_samples()`.  After `init_sources()` is called,
    a list of data sources is populated, representing the batches which
    will be downloaded. See `init_sources()`

    Each source is represented as a tuple `(producer_name, sub_index, iteration)`.
    Before `init_samples()` is called, the user can modify the list.
    Once `init_samples()` is called, all data is downloaded and batches
    can be obtained with iter().

    After initialization, samples and targets can be updated calling `update_data()`,
    which shuffles the available samples, if `shuffle` is set to ``True`` at initialization.

    :param batch_size: Size of batches obtained with __iter__
    :type batch_size: int
    :param shuffle: whether order of samples has to be shuffled when calling `update_data`
    :type shuffle: bool
    :param uploader_info: Set to `auto` uploader information has to be downloaded from DB,
                          or to `manual` if it is provided by the user
    :type uploader_info: str
    :param uploader_name: Name of uploader info dataset, only used if `uploader_info` is `auto`
    :type uploader_name: str
    :param sample_prefix: prefix of keys representing batches
    :type sample_prefix: str
    :param target_prefix: prefix of keys representing targets
    :type target_prefix: str
    :param sub_indices: Sub indices of the batches. This is useful in case each producer
                        has multiple ranks and each rank produces batches. Each
                        rank will then need to use a different sub-index, which is an element
                        of the `sub_indices`. If an integer is specified for `sub_indices`,
                        then it is converted to `range(sub_indices)`.
    :type sub_indices: int or list
    :param num_classes: Number of classes of targets, if categorical
    :type num_classes: int
    :param producer_prefixes: Prefixes of processes which will be producing batches.
                              This can be useful in case the consumer processes also
                              have other incoming entities.
    :type producer_prefixes: str
    :param smartredis_cluster: Whether the Orchestrator will be run as a cluster
    :type smartredis_cluster: bool
    :param smartredis_address: Address of Redis client as <ip_address>:<port>
    :type smartredis_address: str
    :param replica_rank: When BatchDownloader is used in a distributed setting, indicates
                         the rank of this replica
    :type replica_rank: int
    :param num_replicas: When ContinuousBatchDownlaoder is used in a distributed setting,
                         indicates the total number of replicas (ranks)
    :type num_replicas: int
    :param verbose: Whether log messages should be printed
    :type verbose: bool
    :param init_samples: whether samples should be initialized in the constructor
    :type init_samples: bool
    """

    def __init__(self, **kwargs):
        StaticDataGenerator.__init__(self, **kwargs)

    def __iter__(self):
        if self.sources:
            self.update_data()
        return super().__iter__()

    def _add_samples(self, batch_name, target_name):
        StaticDataGenerator.add_samples(self, batch_name, target_name)


    def __iter__(self):
        if self.sources:
            self.update_data()
        return super().__iter__()


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: StaticDataGenerator, **kwargs):
        super().__init__(
            dataset,
            worker_init_fn=self.worker_init_fn,
            persistent_workers=True,
            **kwargs,
        )

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # the dataset copy in this worker process
        dataset.init_sources()
        overall_sources = dataset.sources

        worker_id = worker_info.id

        # configure the dataset to only process the split workload
        per_worker = int((len(overall_sources)) // worker_info.num_workers)

        if per_worker > 0:
            if worker_id < worker_info.num_workers - 1:
                sources = overall_sources[
                    worker_id * per_worker : (worker_id + 1) * per_worker
                ]
            else:
                sources = overall_sources[worker_id * per_worker :]
        else:
            if worker_id < len(overall_sources):
                sources = overall_sources[worker_id]
            else:
                sources = []

        print(f"{worker_id}: {sources}")

        dataset.init_samples(sources)
