import numpy as np
import torch
from smartsim.ml.data import BatchDownloader, ContinuousBatchDownloader

class StaticDataGenerator(BatchDownloader,torch.utils.data.IterableDataset):
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
    
    def __init__(
        self,
        **kwargs
    ):
        BatchDownloader.__init__(self, **kwargs)


    def _add_samples(self, batch_name, target_name):
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
    def __init__(
        self,
        **kwargs
    ):
        StaticDataGenerator.__init__(
            self,
            **kwargs
        )


    def __iter__(self):
        if self.sources:
            self.update_data()
        return super().__iter__()


    def _add_samples(self, batch_name, target_name):
        StaticDataGenerator.add_samples(self, batch_name, target_name)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
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
