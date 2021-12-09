import time
from os import environ

import numpy as np

from smartredis import Client, Dataset
from smartredis.error import RedisReplyError


def form_name(*args):
    return "_".join(str(arg) for arg in args if arg is not None)


class TrainingDataUploader:
    """A class to simplify uploading batches of samples to train a model.

    This class can be used to upload samples following a simple convention
    for naming. Once created, the function `publish_info` can be used
    to put all details about the data set on the Orchestrator. A training
    process can thus access them and get all relevant information to download
    the batches which are uploaded.

    Each time a new batch is available, it is sufficient to call `put_batch`,
    and the data will be stored following the naming convention specified
    by the attributes of this class.

    :param name: Name of the dataset as stored on the Orchestrator
    :type name: str
    :param sample_prefix: Prefix of samples batches
    :type sample_prefix: str
    :param target_prefix: Prefix of target batches (if needed)
    :type target_prefix: str
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
    :param sub_indices: Sub indices of the batches. This is useful in case each process
                        consists in multiple ranks and each rank produces batches. Each
                        rank will then need to use a different sub-index, which is an element
                        of the `sub_indices`. If an integer is specified for `sub_indices`,
                        then it is converted to `range(sub_indices)`.
    :type sub_indices: int or list
    :param verbose: If output should be logged to screen.
    :type verbose: bool

    """

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
        verbose=False,
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
        self.verbose = verbose

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
        if self.verbose:
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
    """A class to download batches from the DB.

    By default, the BatchDownloader has to be created in a process
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

    If the user needs to modify the list of sources, then `init_samples=False`
    has to be set. In that case, to set up a `BatchDownlaoder`, the user has to call
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
        **kwargs,
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
            if isinstance(sub_indices, int):
                self.sub_indices = [str(sub_idx) for sub_idx in range(sub_indices)]
            elif isinstance(sub_indices, list):
                self.sub_indices = [str(sub_idx) for sub_idx in sub_indices]
            elif sub_indices is None:
                self.sub_indices = None
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

        if init_samples:
            self.init_sources()
            self.init_samples()

    def log(self, message):
        """Log message if self.verbose is set to ``True``

        :param message: The message to log
        :type message: str
        """
        if self.verbose:
            print(message)

    def _list_all_sources(self):
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

    def init_sources(self):
        """Initalize list of data sources based on incoming entitites and self.sub_indices.


        Each source is represented as a tuple `(producer_name, sub_index)`.
        Before `init_samples()` is called, the user can modify the list.
        Once `init_samples()` is called, all data is downloaded and batches
        can be obtained with iter(). The list of all sources is stored as `self.sources`.

        :raises ValueError: If self.uploader_info is set to `auto` but no `uploader_name` is specified.
        :raises ValueError: If self.uploader_info is not set to `auto` or `manual`.
        """
        self.client = Client(self.smartredis_address, self.smartredis_cluster)
        if self.uploader_info == "auto":
            if not self.uploader_name:
                raise ValueError(
                    "uploader_name can not be empty if uploader_info is 'auto'"
                )
            self._get_uploader_info(self.uploader_name)
        elif self.uploader_info == "manual":
            pass
        else:
            raise ValueError(
                f"uploader_info must be one of 'auto' or 'manual', but was {self.uploader_info}"
            )

        self.sources = self._list_all_sources()

    @property
    def need_targets(self):
        """Compute if targets have to be downloaded.

        :return: Whether targets (or labels) should be downloaded
        :rtype: bool
        """
        return self.target_prefix and not self.autoencoding

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

    def init_samples(self, sources=None):
        """Initialize samples (and targets, if needed).

        This function will not return until samples have been downloaded
        from all sources.

        :param sources: List of sources as defined in `init_sources`, defaults to None,
                        in which case sources will be initialized, unless `self.sources`
                        is already set
        :type sources: list[tuple], optional
        """
        self.autoencoding = self.sample_prefix == self.target_prefix

        if sources is not None:
            self.sources = sources

        if self.sources is None:
            self.sources = self._list_all_sources()

        self.log("Generator initialization complete")

        self._update_samples_and_targets()
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_exists(self, batch_name, target_name):

        if self.need_targets:
            return self.client.tensor_exists(batch_name) and self.client.tensor_exists(
                target_name
            )
        else:
            return self.client.tensor_exists(batch_name)

    def _add_samples(self, batch_name, target_name):
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

    def _get_uploader_info(self, uploader_name):
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
            while not self._data_exists(batch_name, target_name):
                time.sleep(10)

            self._add_samples(batch_name, target_name)

    def update_data(self):
        """Update data.

        Since the list of samples is not changed
        at runtime, this function
        only shuffles samples if `self.shuffle` is ``True``.
        """
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

    def __len__(self):
        length = int(np.floor(self.num_samples / self.batch_size))
        return length


class ContinuousBatchDownloader(BatchDownloader):
    """A class to download batches from the DB.

    By default, the ContinuousBatchDownloader has to be created in a process
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
       for all batches named <sample_prefix>_<sub_index>_<iteration> for all indices
       in `sub_indices` if supplied, and, if
       `target_prefix` is supplied, it will also query for all targets
       named <target_prefix>.<sub_index>.<iteration>. If `producer_prefixes` is
       None, then all incoming entities will be treated as producers,
       and for each one, the corresponding batches will be downloaded.

    The flag `init_samples` defines whether sources (the list of batches
    to be fetched) and samples (the actual data) should automatically
    be set up in the costructor.

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
        super().__init__(**kwargs)

    def _list_all_sources(self):
        sources = super()._list_all_sources()
        # Append the batch index to each source
        for source in sources:
            source.append(0)
        return sources

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
            while self._data_exists(batch_name, target_name):
                self._add_samples(batch_name, target_name)
                source[2] += 1
                index = source[2]
                batch_name = form_name(self.sample_prefix, index, sub_index)
                if self.need_targets:
                    target_name = form_name(self.target_prefix, index, sub_index)

                self.log(f"Retrieving {batch_name}...")

    def update_data(self):
        """Update data.

        Fetch new batches (if available) from the DB. Also shuffle
        list of samples if `self.shuffle` is set to ``True``.
        """
        self._update_samples_and_targets()
        if self.shuffle:
            np.random.shuffle(self.indices)

    def init_samples(self, sources=None):
        """Initialize samples (and targets, if needed).

        This function will not return until at least one batch worth of data
        has been downloaded.

        :param sources: List of sources as defined in `init_sources`, defaults to None,
                        in which case sources will be initialized, unless `self.sources`
                        is already set
        :type sources: list[tuple], optional
        """
        self.autoencoding = self.sample_prefix == self.target_prefix

        if sources is not None:
            self.sources = sources

        if self.sources is None:
            self.sources = self._list_all_sources()

        if self.sources:

            while len(self) < 1:
                self._update_samples_and_targets()
                if len(self) < 1:
                    time.sleep(10)
            self.log("Generator initialization complete")
        else:
            self.log(
                "Generator has no associated sources, this can happen if the number of "
                "loader workers is larger than the number of available sources."
            )
