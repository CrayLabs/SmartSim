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

import time
from os import environ

import numpy as np
from smartredis import Client, Dataset
from smartredis.error import RedisReplyError

from ..error import SmartSimError
from ..log import get_logger

logger = get_logger(__name__)


def form_name(*args):
    return "_".join(str(arg) for arg in args if arg is not None)


class DataInfo:
    def __init__(
        self, list_name, sample_name="samples", target_name="targets", num_classes=None
    ):
        self.list_name = list_name
        self.sample_name = sample_name
        self.target_name = target_name
        self.num_classes = num_classes
        self._ds_name = form_name(self.list_name, "info")

    def publish(self, client: Client):
        info_ds = Dataset(self._ds_name)
        info_ds.add_meta_string("sample_name", self.sample_name)
        if self.target_name:
            info_ds.add_meta_string("target_name", self.target_name)
        if self.num_classes:
            info_ds.add_meta_scalar("num_classes", self.num_classes)
        client.put_dataset(info_ds)

    def download(self, client: Client):
        try:
            info_ds = client.get_dataset(self._ds_name)
        except RedisReplyError:
            # If the info was not published, proceed with default parameters
            return
        self.sample_name = info_ds.get_meta_strings("sample_name")[0]
        field_names = info_ds.get_metadata_field_names()
        if "target_name" in field_names:
            self.target_name = info_ds.get_meta_strings("target_name")[0]
        if "num_classes" in field_names:
            self.num_classes = info_ds.get_meta_scalars("num_classes")[0]


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

    :param list_name: Name of the dataset as stored on the Orchestrator
    :type list_name: str
    :param sample_name: Name of samples tensor in uploaded Datasets
    :type sample_name: str
    :param target_name: Name of targets tensor (if needed) in uploaded Datasets
    :type target_name: str
    :param num_classes: Number of classes of targets, if categorical
    :type num_classes: int
    :param cluster: Whether the SmartSim Orchestrator is being run as a cluster
    :type cluster: bool
    :param address: Address of Redis DB as <ip_address>:<port>
    :type address: str
    :param rank: Rank of DataUploader in multi-process application (e.g. MPI rank).
    :type rank: int
    :param verbose: If output should be logged to screen.
    :type verbose: bool

    """

    def __init__(
        self,
        list_name="training_data",
        sample_name="samples",
        target_name="targets",
        num_classes=None,
        cluster=True,
        address=None,
        rank=None,
        verbose=False,
    ):
        if not list_name:
            raise ValueError("Name can not be empty")
        if not sample_name:
            raise ValueError("Sample name can not be empty")

        self.list_name = list_name
        self.sample_name = sample_name
        self.target_name = target_name
        self.num_classes = num_classes

        self.client = Client(address=address, cluster=cluster)
        self.verbose = verbose
        self.batch_idx = 0
        self.rank = rank
        self.info = DataInfo(list_name, sample_name, target_name, num_classes)

    def publish_info(self):
        self.info.publish(self.client)

    def put_batch(self, samples, targets=None):
        batch_ds = Dataset(form_name("training_samples", self.rank, self.batch_idx))
        batch_ds.add_tensor(self.sample_name, samples)
        if (
            targets is not None
            and self.target_name
            and (self.target_name != self.sample_name)
        ):
            batch_ds.add_tensor(self.target_name, targets)
            if self.verbose:
                logger.info(
                    f"Putting dataset {batch_ds._name} with samples and targets"
                )
        else:
            if self.verbose:
                logger.info(f"Putting dataset {batch_ds._name} with samples")

        self.client.put_dataset(batch_ds)
        self.client.append_to_list(self.list_name, batch_ds)
        if self.verbose:
            logger.info(f"Added dataset to list {self.list_name}")

        self.batch_idx += 1


class StaticDataDownloader:
    """A class to download a dataset from the DB.

    By default, the StaticDataDownloader has to be created in a process
    launched through SmartSim, with sample producers listed as incoming
    entities.

    All details about the batches must be defined in
    the constructor; two mechanisms are available, `manual` and
    `auto`.

     - When specifying `auto`, the user must also specify
      `uploader_name`. StaticDataDownloader will get all needed information
      from the database (this expects a Dataset like the one created
      by TrainingDataUploader to be available and stored as `uploader_name`
      on the DB).

     - When specifying `manual`, the user must also specify details
       of batch naming. Specifically, for each incoming entity with
       a name starting with an element of `producer_prefixes`,
       StaticDataDownloader will query the DB
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
    :param sample_name: prefix of keys representing batches
    :type sample_name: str
    :param target_name: prefix of keys representing targets
    :type target_name: str
    :param num_classes: Number of classes of targets, if categorical
    :type num_classes: int
    :param producer_prefixes: Prefixes of names of which will be producing batches.
                              These can be e.g. prefixes of SmartSim entity names in
                              an ensemble.
    :type producer_prefixes: str
    :param cluster: Whether the Orchestrator will be run as a cluster
    :type cluster: bool
    :param address: Address of Redis client as <ip_address>:<port>
    :type address: str
    :param replica_rank: When StaticDataDownloader is used distributedly, indicates
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
        data_info: DataInfo = None,
        list_name="training_data",
        cluster=True,
        address=None,
        replica_rank=0,
        num_replicas=1,
        verbose=False,
        init_samples=True,
        **kwargs,
    ):
        self.replica_rank = replica_rank
        self.num_replicas = num_replicas
        self.address = address
        self.cluster = cluster
        self.verbose = verbose
        self.samples = None
        self.targets = None
        self.num_samples = 0
        self.indices = np.arange(0)
        self.next_index = self.replica_rank
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.client = None
        if not data_info:
            if not list_name:
                raise ValueError(
                    "Neither data_info nor list_name were passed to the data loader."
                )
            data_info = DataInfo(list_name=self.list_name)
            self.client = Client(self.address, self.cluster)
            data_info.download(self.client)
        self.sample_name = data_info.sample_name
        self.target_name = data_info.target_name
        self.autoencoding = self.sample_name == self.target_name
        self.num_classes = data_info.num_classes
        self.list_name = data_info.list_name

        if self.client is None:
            self.client = Client(self.address, self.cluster)

        if init_samples:
            self.init_samples()

    def log(self, message):
        if self.verbose:
            logger.info(message)

    @property
    def need_targets(self):
        """Compute if targets have to be downloaded.

        :return: Whether targets (or labels) should be downloaded
        :rtype: bool
        """
        return self.target_name and not self.autoencoding

    def __len__(self):
        length = int(np.floor(self.num_samples / self.batch_size))
        return length

    def __iter__(self):

        self.update_data()
        # Generate data
        if len(self) < 1:
            msg = "Not enough samples in generator for one batch. "
            msg += "Please run init_samples() or initialize generator with init_samples=True"
            raise ValueError(msg)

        for index in range(len(self)):
            indices = self.indices[
                index * self.batch_size : (index + 1) * self.batch_size
            ]

            x, y = self.__data_generation(indices)

            if y is not None:
                yield x, y
            else:
                yield x

    def init_samples(self):
        """Initialize samples (and targets, if needed).

        This function will not return until samples have been downloaded: it will
        make a new attempt to download samples every ten seconds

        :param sources: List of sources as defined in `init_sources`, defaults to None,
                        in which case sources will be initialized, unless `self.sources`
                        is already set
        :type sources: list[tuple], optional
        """
        while True:
            self._update_samples_and_targets()
            if len(self):
                break
            else:
                logger.info("DataLoader not download samples, will try again in 10 seconds")
                time.sleep(10)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_exists(self, batch_name, target_name):

        if self.need_targets:
            return self.client.tensor_exists(batch_name) and self.client.tensor_exists(
                target_name
            )
        else:
            return self.client.tensor_exists(batch_name)

    def _add_samples(self, indices):
        if self.num_replicas == 1:
            datasets: list[Dataset] = self.client.get_dataset_list_range(self.list_name, start_index=indices[0], end_index = indices[-1])
        else:
            datasets: list[Dataset] = []
            for idx in indices:
                datasets += self.client.get_dataset_list_range(self.list_name, start_index=indices[idx], end_index=indices[idx])
        
        if self.samples is None:        
            self.samples = datasets[0].get_tensor(self.sample_name)
            if self.need_targets:
                self.targets = datasets[0].get_tensor(self.target_name)

            if len(datasets) > 1:
                datasets = datasets[1:]

        for dataset in datasets:
            self.samples = np.concatenate(
                (self.samples, dataset.get_tensor(self.sample_name))
            )
            if self.need_targets:
                self.targets = np.concatenate(
                    (self.targets, dataset.get_tensor(self.target_name))
                )

        self.num_samples = self.samples.shape[0]
        self.indices = np.arange(self.num_samples)
        self.log(f"New dataset size: {self.num_samples}, batches: {len(self)}")

    def _update_samples_and_targets(self):
        list_length = self.client.get_list_length(self.list_name)
        # As we emply a simple round-robin strategy, we need the list to be
        # as long as the next multiple of total number of replicas plus the
        # rank of this data loader (plus one because of 0-based indexing)

        # Strictly greater, because next_index is 0-based
        if list_length > self.next_index:
            indices = range(self.next_index, list_length, self.num_replicas)
            self._add_samples(indices)
            self.next_index = indices[-1] + self.replica_rank

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

    def __len__(self):
        length = int(np.floor(self.num_samples / self.batch_size))
        return length