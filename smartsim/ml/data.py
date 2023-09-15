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
import typing as t
from os import environ

import numpy as np
from smartredis import Client, Dataset
from smartredis.error import RedisReplyError

from ..error import SSInternalError
from ..log import get_logger


logger = get_logger(__name__)


def form_name(*args: t.Any) -> str:
    return "_".join(str(arg) for arg in args if arg is not None)


class DataInfo:
    """A class holding all relevant information to download datasets from aggregation
    lists

    This class can be passed as argument to SmartSim's ML data loaders, as it wraps the
    information about the aggregation list holding the training datasets.

    Each training dataset will store batches of samples and (optionally) labels or
    targets in the form of tensors. The tensors must always have the same names, which
    can be accessed in ``DataInfo.sample_name`` and ``DataInfo.target_name``.

    :param list_name: Name of the aggregation list used for sample datasets
    :type list_name: str
    :param sample_name: Name of tensor holding training samples in stored datasets.
    :type sample_name: str
    :param target_name: Name of tensor holding targets or labels in stored datasets.
    :type target_name: str
    :num_classes: Number of classes (for categorical data).
    :type num_classes: int | None
    """

    def __init__(
        self,
        list_name: str,
        sample_name: str = "samples",
        target_name: str = "targets",
        num_classes: t.Optional[int] = None,
    ) -> None:
        self.list_name = list_name
        self.sample_name = sample_name
        self.target_name = target_name
        self.num_classes = num_classes
        self._ds_name = form_name(self.list_name, "info")

    def publish(self, client: Client) -> None:
        """Upload DataInfo information to Orchestrator

        The information is put on the DB as a DataSet, with strings
        stored as metastrings and integers stored as metascalars.

        :param client: Client to connect to Database
        :type client: SmartRedis.Client
        """
        info_ds = Dataset(self._ds_name)
        info_ds.add_meta_string("sample_name", self.sample_name)
        if self.target_name:
            info_ds.add_meta_string("target_name", self.target_name)
        if self.num_classes:
            info_ds.add_meta_scalar("num_classes", self.num_classes)
        client.put_dataset(info_ds)

    def download(self, client: Client) -> None:
        """Download DataInfo information from Orchestrator

        The information retrieved from the DB is used to populate
        this object's members. If the information is not available
        on the DB, the object members are not modified.

        :param client: Client to connect to Database
        :type client: SmartRedis.Client
        """
        try:
            info_ds = client.get_dataset(self._ds_name)
        except RedisReplyError:
            # If the info was not published, proceed with default parameters
            logger.warning(
                "Could not retrieve data for DataInfo object, the following "
                "values will be kept."
            )
            logger.warning(str(self))
            return
        self.sample_name = info_ds.get_meta_strings("sample_name")[0]
        field_names = info_ds.get_metadata_field_names()
        if "target_name" in field_names:
            self.target_name = info_ds.get_meta_strings("target_name")[0]
        if "num_classes" in field_names:
            self.num_classes = info_ds.get_meta_scalars("num_classes")[0]

    def __repr__(self) -> str:
        strings = ["DataInfo object"]
        strings += [f"Aggregation list name: {self.list_name}"]
        strings += [f"Sample tensor name: {self.sample_name}"]
        if self.target_name:
            strings += [f"Target tensor name: {self.target_name}"]
        if self.num_classes:
            strings += [f"Number of classes: {self.num_classes}"]
        return "\n".join(strings)


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
        list_name: str = "training_data",
        sample_name: str = "samples",
        target_name: str = "targets",
        num_classes: t.Optional[int] = None,
        cluster: bool = True,
        address: t.Optional[str] = None,
        rank: t.Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        if not list_name:
            raise ValueError("Name can not be empty")
        if not sample_name:
            raise ValueError("Sample name can not be empty")

        self.client = Client(address=address, cluster=cluster)
        self.verbose = verbose
        self.batch_idx = 0
        self.rank = rank
        self._info = DataInfo(list_name, sample_name, target_name, num_classes)

    @property
    def list_name(self) -> str:
        return self._info.list_name

    @property
    def sample_name(self) -> str:
        return self._info.sample_name

    @property
    def target_name(self) -> str:
        return self._info.target_name

    @property
    def num_classes(self) -> t.Optional[int]:
        return self._info.num_classes

    def publish_info(self) -> None:
        self._info.publish(self.client)

    def put_batch(
        self,
        samples: np.ndarray,  # type: ignore[type-arg]
        targets: t.Optional[np.ndarray] = None,  # type: ignore[type-arg]
    ) -> None:
        batch_ds_name = form_name("training_samples", self.rank, self.batch_idx)
        batch_ds = Dataset(batch_ds_name)
        batch_ds.add_tensor(self.sample_name, samples)
        if (
            targets is not None
            and self.target_name
            and (self.target_name != self.sample_name)
        ):
            batch_ds.add_tensor(self.target_name, targets)
            if self.verbose:
                logger.info(f"Putting dataset {batch_ds_name} with samples and targets")
        else:
            if self.verbose:
                logger.info(f"Putting dataset {batch_ds_name} with samples")

        self.client.put_dataset(batch_ds)
        self.client.append_to_list(self.list_name, batch_ds)
        if self.verbose:
            logger.info(f"Added dataset to list {self.list_name}")
            logger.info(f"List length {self.client.get_list_length(self.list_name)}")

        self.batch_idx += 1


class DataDownloader:
    """A class to download a dataset from the DB.

    By default, the DataDownloader has to be created in a process
    launched through SmartSim, with sample producers listed as incoming
    entities.

    Information about the uploaded datasets can be defined in two ways:

     - By supplying a DataInfo object as value of ``data_info_or_list_name``

     - By supplying a string as value of ``data_info_or_list_name``.
       in this case, an attempt is made to download information from the
       DB, where a Dataset called ``<data_info_or_list_name>_info`` should be
       available and have the information normally stored by DataInfo.publish()

    The flag `init_samples` defines whether samples should automatically
    be set up in the costructor.

    If the user needs to modify the `DataDownloader` object before starting
    the training, then `init_samples=False` has to be set.
    In that case, to set up a `DataDownloader`, the user has to call
    `init_samples()`.

    Calling `update_data()`
     - check if new batches are available and download them,
       if `dynamic` is set to `True`
     - shuffle the dataset if `shuffle` is set to ``True``.

    :param batch_size: Size of batches obtained with __iter__
    :type batch_size: int
    :param dynamic: Whether new batches should be donwnloaded when ``update_data``
        is called.
    :type dtnamic: bool
    :param shuffle: whether order of samples has to be shuffled when
        calling `update_data`
    :type shuffle: bool
    :param data_info_or_list_name: DataInfo object with details about dataset to
        download, if a string is passed, it is used to download DataInfo data
        from DB, assuming it was stored with ``list_name=data_info_or_list_name``
    :type data_info_or_list_name: DataInfo | str
    :param list_name: Name of aggregation list used to upload data
    :type list_name: str
    :param cluster: Whether the Orchestrator will be run as a cluster
    :type cluster: bool
    :param address: Address of Redis client as <ip_address>:<port>
    :type address: str
    :param replica_rank: When StaticDataDownloader is used distributedly,
        indicates the rank of this object
    :type replica_rank: int
    :param num_replicas: When BatchDownlaoder is used distributedly, indicates
                         the total number of ranks
    :type num_replicas: int
    :param verbose: Whether log messages should be printed
    :type verbose: bool
    :param init_samples: whether samples should be initialized in the constructor
    :type init_samples: bool
    :param max_fetch_trials: maximum number of attempts to initialize data
    :type max_fetch_trials: int
    """

    def __init__(
        self,
        data_info_or_list_name: t.Union[str, DataInfo],
        batch_size: int = 32,
        dynamic: bool = True,
        shuffle: bool = True,
        cluster: bool = True,
        address: t.Optional[str] = None,
        replica_rank: int = 0,
        num_replicas: int = 1,
        verbose: bool = False,
        init_samples: bool = True,
        max_fetch_trials: int = -1,
    ) -> None:
        self.address = address
        self.cluster = cluster
        self.verbose = verbose
        self.samples = None
        self.targets = None
        self.num_samples = 0
        self.indices = np.arange(0)
        self.shuffle = shuffle
        self.dynamic = dynamic
        self.batch_size = batch_size
        if isinstance(data_info_or_list_name, DataInfo):
            self._info = data_info_or_list_name
        elif isinstance(data_info_or_list_name, str):
            self._info = DataInfo(list_name=data_info_or_list_name)
            client = Client(self.address, self.cluster)
            self._info.download(client)
        else:
            raise TypeError("data_info_or_list_name must be either DataInfo or str")
        self._client: t.Optional[Client] = None
        sskeyin = environ.get("SSKEYIN", "")
        self.uploader_keys = sskeyin.split(",")

        self.set_replica_parameters(replica_rank, num_replicas)

        if init_samples:
            self.init_samples(max_fetch_trials)

    @property
    def client(self) -> Client:
        if self._client is None:
            raise ValueError("Client not initialized")
        return self._client

    def log(self, message: str) -> None:
        if self.verbose:
            logger.info(message)

    def set_replica_parameters(self, replica_rank: int, num_replicas: int) -> None:
        self.replica_rank = replica_rank
        self.num_replicas = num_replicas
        self.next_indices = [self.replica_rank] * max(1, len(self.uploader_keys))

    @property
    def autoencoding(self) -> bool:
        return self.sample_name == self.target_name

    @property
    def list_name(self) -> str:
        return self._info.list_name

    @property
    def sample_name(self) -> str:
        return self._info.sample_name

    @property
    def target_name(self) -> str:
        return self._info.target_name

    @property
    def num_classes(self) -> t.Optional[int]:
        return self._info.num_classes

    @property
    def need_targets(self) -> bool:
        """Compute if targets have to be downloaded.

        :return: Whether targets (or labels) should be downloaded
        :rtype: bool
        """
        return bool(self.target_name) and not self.autoencoding

    def __len__(self) -> int:
        length = int(np.floor(self.num_samples / self.batch_size))
        return length

    def _calc_indices(self, index: int) -> np.ndarray:  # type: ignore[type-arg]
        return self.indices[index * self.batch_size : (index + 1) * self.batch_size]

    def __iter__(
        self,
    ) -> t.Iterator[t.Tuple[np.ndarray, np.ndarray]]:  # type: ignore[type-arg]
        self.update_data()
        # Generate data
        if len(self) < 1:
            raise ValueError(
                "Not enough samples in generator for one batch. Please run "
                "init_samples() or initialize generator with init_samples=True"
            )

        yield from (
            self._data_generation(self._calc_indices(idx)) for idx in range(len(self))
        )

    def init_samples(self, init_trials: int = -1) -> None:
        """Initialize samples (and targets, if needed).

        A new attempt to download samples will be made every ten seconds,
        for ``init_trials`` times.
        :param init_trials: maximum number of attempts to fetch data
        :type init_trials: int
        """
        self._client = Client(self.address, self.cluster)

        num_trials = 0
        max_trials = init_trials or -1
        while not self and num_trials != max_trials:
            self._update_samples_and_targets()
            self.log(
                "DataLoader could not download samples, will try again in 10 seconds"
            )
            time.sleep(10)
            num_trials += 1

        if not self:
            raise SSInternalError(
                "Could not download samples in given number of trials"
            )
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_exists(self, batch_name: str, target_name: str) -> bool:
        if self.need_targets:
            return all(
                self.client.tensor_exists(datum) for datum in [batch_name, target_name]
            )

        return bool(self.client.tensor_exists(batch_name))

    def _add_samples(self, indices: t.List[int]) -> None:
        datasets: t.List[Dataset] = []

        if self.num_replicas == 1:
            datasets = self.client.get_dataset_list_range(
                self.list_name, start_index=indices[0], end_index=indices[-1]
            )
        else:
            for idx in indices:
                datasets += self.client.get_dataset_list_range(
                    self.list_name, start_index=idx, end_index=idx
                )

        if self.samples is None:
            self.samples = datasets[0].get_tensor(self.sample_name)
            if self.need_targets:
                self.targets = datasets[0].get_tensor(self.target_name)

            if len(datasets) > 1:
                datasets = datasets[1:]

        if self.samples is not None:
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

    def _update_samples_and_targets(self) -> None:
        self.log(f"Rank {self.replica_rank} out of {self.num_replicas} replicas")

        for uploader_idx, uploader_key in enumerate(self.uploader_keys):
            if uploader_key:
                self.client.use_list_ensemble_prefix(True)
                self.client.set_data_source(uploader_key)

            list_length = self.client.get_list_length(self.list_name)

            # Strictly greater, because next_index is 0-based
            if list_length > self.next_indices[uploader_idx]:
                start = self.next_indices[uploader_idx]
                indices = list(range(start, list_length, self.num_replicas))
                self._add_samples(indices)
                self.next_indices[uploader_idx] = indices[-1] + self.num_replicas

    def update_data(self) -> None:
        if self.dynamic:
            self._update_samples_and_targets()
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _data_generation(
        self, indices: np.ndarray  # type: ignore[type-arg]
    ) -> t.Tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
        # Initialization
        if self.samples is None:
            raise ValueError("Samples have not been initialized")

        xval = self.samples[indices]

        if self.need_targets:
            yval = self.targets[indices]
        elif self.autoencoding:
            yval = xval
        else:
            return xval

        return xval, yval
