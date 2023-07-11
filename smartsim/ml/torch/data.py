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
import typing as t
from smartredis import Client, Dataset

from smartsim.ml.data import DataDownloader


class _TorchDataGenerationCommon(DataDownloader, torch.utils.data.IterableDataset):
    def __init__(self, **kwargs: t.Any) -> None:
        init_samples = kwargs.pop("init_samples", False)
        kwargs["init_samples"] = False
        super().__init__(**kwargs)
        if init_samples:
            self.log(
                "PyTorch Data Generator has to be created with "
                "init_samples=False. Setting it to False automatically."
            )

    def _add_samples(self, indices: t.List[int]) -> None:
        if self.client is None:
            client = Client(self.address, self.cluster)
        else:
            client = self.client

        datasets: t.List[Dataset] = []
        if self.num_replicas == 1:
            datasets = client.get_dataset_list_range(
                self.list_name, start_index=indices[0], end_index=indices[-1]
            )
        else:
            for idx in indices:
                datasets += client.get_dataset_list_range(
                    self.list_name, start_index=idx, end_index=idx
                )

        if self.samples is None:
            self.samples = torch.tensor(datasets[0].get_tensor(self.sample_name))
            if self.need_targets:
                self.targets = torch.tensor(datasets[0].get_tensor(self.target_name))

            if len(datasets) > 1:
                datasets = datasets[1:]

        for dataset in datasets:
            self.samples = torch.cat(
                (self.samples, torch.tensor(dataset.get_tensor(self.sample_name)))
            )
            if self.need_targets:
                self.targets = torch.cat(
                    (self.targets, torch.tensor(dataset.get_tensor(self.target_name)))
                )

        if self.samples is not None:
            self.num_samples = self.samples.shape[0]
        self.indices = np.arange(self.num_samples)
        self.log(f"New dataset size: {self.num_samples}, batches: {len(self)}")


class StaticDataGenerator(_TorchDataGenerationCommon):
    """A class to download a dataset from the DB.

    Details about parameters and features of this class can be found
    in the documentation of ``DataDownloader``, of which it is just
    a PyTorch-specialized sub-class with dynamic=False and init_samples=False.

    When used in the DataLoader defined in this class, samples are initialized
    automatically before training. Other data loaders using this generator
    should implement the same behavior.

    """

    def __init__(self, **kwargs: t.Any) -> None:
        dynamic = kwargs.pop("dynamic", False)
        kwargs["dynamic"] = False
        super().__init__(**kwargs)
        if dynamic:
            self.log(
                "Static data generator cannot be started "
                "with dynamic=True, setting it to False"
            )


class DynamicDataGenerator(_TorchDataGenerationCommon):
    """A class to download batches from the DB.

    Details about parameters and features of this class can be found
    in the documentation of ``DataDownloader``, of which it is just
    a PyTorch-specialized sub-class with dynamic=True and init_samples=False.

    When used in the DataLoader defined in this class, samples are initialized
    automatically before training. Other data loaders using this generator
    should implement the same behavior.
    """

    def __init__(self, **kwargs: t.Any) -> None:
        dynamic = kwargs.pop("dynamic", True)
        kwargs["dynamic"] = True
        super().__init__(**kwargs)
        if not dynamic:
            self.log(
                "Dynamic data generator cannot be started with dynamic=False, "
                "setting it to True"
            )


def _worker_init_fn(worker_id: int) -> None:
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process

    worker_id = worker_info.id
    num_workers = worker_info.num_workers

    dataset.set_replica_parameters(
        replica_rank=dataset.replica_rank * num_workers + worker_id,
        num_replicas=dataset.num_replicas * num_workers,
    )
    dataset.log(
        f"Worker {worker_id+1}/{num_workers}: dataset replica "
        f"{dataset.replica_rank+1}/{dataset.num_replicas}"
    )

    dataset.init_samples()


class DataLoader(torch.utils.data.DataLoader):  # pragma: no cover
    """DataLoader to be used as a wrapper of StaticDataGenerator or DynamicDataGenerator

    This is just a sub-class of ``torch.utils.data.DataLoader`` which
    sets up sources of a data generator correctly. DataLoader parameters
    such as `num_workers` can be passed at initialization. `batch_size` should always
    be set to None.
    """

    def __init__(self, dataset: _TorchDataGenerationCommon, **kwargs: t.Any) -> None:
        super().__init__(
            dataset,
            worker_init_fn=_worker_init_fn,
            persistent_workers=True,
            **kwargs,
        )
