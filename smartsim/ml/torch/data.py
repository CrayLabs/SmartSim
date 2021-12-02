import time
from os import environ

import numpy as np
import torch
import torch.nn.functional as F

from smartredis import Client
from smartredis.error import RedisReplyError
from smartsim.ml import form_name
from smartsim.ml.data import BatchDownloader, ContinuousBatchDownloader

class StaticDataGenerator(BatchDownloader,torch.utils.data.IterableDataset):
    def __init__(
        self,
        **kwargs
    ):
        BatchDownloader.__init__(self, **kwargs)


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


class DataGenerator(ContinuousBatchDownloader, StaticDataGenerator):
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


    def add_samples(self, batch_name, target_name):
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
