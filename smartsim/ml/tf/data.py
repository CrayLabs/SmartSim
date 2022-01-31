import numpy as np
import tensorflow.keras as keras

from smartsim.ml import BatchDownloader, ContinuousBatchDownloader


class StaticDataGenerator(BatchDownloader, keras.utils.Sequence):
    """A class to download batches from the DB, based on Keras ``Sequence``s.

    By default, the StaticDataGenerator has to be created in a process
    launched through SmartSim, with sample producers listed as incoming
    entities.

    All details about the batches must be defined in
    the constructor; two mechanisms are available, `manual` and
    `auto`.

     - When specifying `auto`, the user must also specify
      `uploader_name`. StaticDataGenerator will get all needed information
      from the database (this expects a Dataset like the one created
      by TrainingDataUploader to be available and stored as `uploader_name`
      on the DB).

     - When specifying `manual`, the user must also specify details
       of batch naming. Specifically, for each incoming entity with
       a name starting with an element of `producer_prefixes`,
       StaticDataGenerator will query the DB
       for all batches named <sample_prefix>_<sub_index> for all index
       in `sub_indexes` if supplied, and, if
       `target_prefix` is supplied, it will also query for all targets
       named <target_prefix>.<sub_index>. If `producer_prefixes` is
       None, then all incoming entities will be treated as producers,
       and for each one, the corresponding batches will be downloaded.

    The flag `init_samples` defines whether sources (the list of batches
    to be fetched) and samples (the actual data) should automatically
    be set up in the costructor.

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
    initialization. The function `on_epoch_end()` is a Keras callback function,
    which is called at the end of each epoch and calls `update_data()`.

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
    :param replica_rank: When StaticDataGenerator is used distributedly, indicates
                         the rank of this object
    :type replica_rank: int
    :param num_replicas: When StaticDataGenerator is used distributedly, indicates
                         the total number of ranks
    :type num_replicas: int
    :param verbose: Whether log messages should be printed
    :type verbose: bool
    :param init_samples: whether samples should be initialized in the constructor
    :type init_samples: bool
    """

    def __init__(self, **kwargs):
        BatchDownloader.__init__(self, **kwargs)

    def __getitem__(self, index):
        if len(self) < 1:
            msg = "Not enough samples in generator for one batch. "
            msg += "Please run init_samples() or initialize generator with init_samples=True"
            raise ValueError(
                msg
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


class DataGenerator(ContinuousBatchDownloader, StaticDataGenerator):
    """A class to download batches from the DB.

    By default, the DataGenerator has to be created in a process
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
    This is done automatically at the end of each epoch, when `on_epoch_end()` is called
    by Keras, as a callback function.

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


    def on_epoch_end(self):
        """Callback called at the end of each training epoch

        Update data (the DB is queried for new batches) and
        if `self.shuffle` is set to `True`, data is also shuffled.
        """
        self.update_data()
        super().on_epoch_end()
