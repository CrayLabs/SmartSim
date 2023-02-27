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
import torch

from smartredis import Client, Dataset
from smartsim.ml.data import StaticDataDownloader


class StaticDataGenerator(StaticDataDownloader, torch.utils.data.IterableDataset):
    """A class to download a dataset from the DB.

    Details about parameters and features of this class can be found
    in the documentation of ``StaticDataDownloader``, of which it is just
    a PyTorch-specialized sub-class.

    Note that if the ``StaticDataGenerator`` has to be used through a ``DataLoader``,
    `init_samples` must be set to `False`, as sources and samples will be initialized
    by the ``DataLoader`` workers.
    """

    def __init__(self, **kwargs):
        StaticDataDownloader.__init__(self, **kwargs)

    def _add_samples(self, indices):
        if self.client is None:
            client = Client(self.address, self.cluster)
        else:
            client = self.client

        if self.num_replicas == 1:
            datasets: list[Dataset] = client.get_dataset_list_range(self.list_name, start_index=indices[0], end_index = indices[-1])
        else:
            datasets: list[Dataset] = []
            for idx in indices:
                datasets += client.get_dataset_list_range(self.list_name, start_index=idx, end_index=idx)
        
        if self.samples is None:        
            self.samples = torch.tensor(datasets[0].get_tensor(self.sample_name))
            if self.need_targets:
                self.targets = torch.tensor(datasets[0].get_tensor(self.target_name))

            if len(datasets) > 1:
                datasets = datasets[1:]

        for dataset in datasets:
            self.samples =torch.cat(
                (self.samples, torch.tensor(dataset.get_tensor(self.sample_name)))
            )
            if self.need_targets:
                self.targets = torch.cat(
                    (self.targets, torch.tensor(dataset.get_tensor(self.target_name)))
                )

        self.num_samples = self.samples.shape[0]
        self.indices = np.arange(self.num_samples)
        self.log(f"New dataset size: {self.num_samples}, batches: {len(self)}")

    def update_data(self):
        self._update_samples_and_targets()
        if self.shuffle:
            np.random.shuffle(self.indices)


class DynamicDataGenerator(StaticDataGenerator):
    """A class to download batches from the DB.

    Details about parameters and features of this class can be found
    in the documentation of ``DynamicDataDownloader``, of which it is just
    a PyTorch-specialized sub-class.

    Note that if the ``DynamicDataGenerator`` has to be used through a ``DataLoader``,
    `init_samples` must be set to `False`, as sources and samples will be initialized
    by the ``DataLoader`` workers.
    """

    def __init__(self, **kwargs):
        StaticDataGenerator.__init__(self, **kwargs)

    def __iter__(self):
        self.update_data()
        return super().__iter__()

    def _add_samples(self, indices):
        StaticDataGenerator._add_samples(self, indices)

    def __iter__(self):
        self.update_data()
        return super().__iter__()


class DataLoader(torch.utils.data.DataLoader):  # pragma: no cover
    """DataLoader to be used as a wrapper of StaticDataGenerator or DynamicDataGenerator

    This is just a sub-class of ``torch.utils.data.DataLoader`` which
    sets up sources of a data generator correctly. DataLoader parameters
    such as `num_workers` can be passed at initialization. `batch_size` should always
    be set to None.
    """

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
        dataset: StaticDataGenerator = worker_info.dataset  # the dataset copy in this worker process

        worker_id = worker_info.id
        num_workers = worker_info.num_workers

        dataset.num_replicas *= num_workers
        dataset.replica_rank = dataset.replica_rank*num_workers + worker_id
        dataset.log(f"{worker_id}/{num_workers}")

        dataset.init_samples()
