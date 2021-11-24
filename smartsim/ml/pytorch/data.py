from os import environ
from smartredis import Client, Dataset
from smartredis.error import RedisReplyError

import numpy as np
import time

import torch
import torch.nn.functional as F

from smartsim.ml import form_name

class StaticDataGenerator(torch.utils.data.IterableDataset):
    def __init__(self,
                 uploader_info="auto",
                 uploader_name="training_data",
                 sample_prefix="samples",
                 target_prefix="targets",
                 sub_indices=None,
                 num_classes=None,
                 producer_prefixes=None,
                 smartredis_cluster=True,
                 smartredis_address=None,
                 init_samples=True,
                 device="cpu"
                ):
        self.client = Client(smartredis_address, smartredis_cluster)
        if uploader_info == "manual":
            self.sample_prefix = sample_prefix
            self.target_prefix = target_prefix
            self.sub_indices = sub_indices
            if producer_prefixes:
                self.producer_prefixes = list(producer_prefixes)
            else:
                producer_prefixes = [""]
            self.num_classes = num_classes
        elif uploader_info == "auto":
            if not uploader_name:
                raise ValueError("uploader_name can not be empty if uploader_info is 'auto'")
            self.get_uploader_info(uploader_name)
        else:
            raise ValueError(f"uploader_info must be one of 'auto' or 'manual', but was {uploader_info}")

        self.autoencoding = (self.sample_prefix == self.target_prefix)

        self.samples = None
        if self.need_targets:
            self.targets = None
        self.num_samples = 0

        if init_samples:
            self.init_samples(None)

        self.device = device


    def list_all_sources(self):
        uploaders = environ["SSKEYIN"].split(',')
        sources = []
        for uploader in uploaders:
            if any([uploader.startswith(producer_prefix) for producer_prefix in self.producer_prefixes]):
                if self.sub_indices:
                    sources.extend([[uploader, sub_index] for sub_index in self.sub_indices])
                else:
                    sources.append([uploader, None])
        return sources

    
    def init_samples(self, sources=None):
        if sources is None:
            self.sources = self.list_all_sources()

        self.update_data()
        print("Generator initialization complete")


    @property
    def need_targets(self):
        return self.target_prefix and not self.autoencoding


    def get_uploader_info(self, uploader_name):
        dataset_name = form_name(uploader_name, "info")
        print(f"Uploader dataset name: {dataset_name}")
        while not self.client.dataset_exists(dataset_name):
            time.sleep(10)
        
        uploader_info = self.client.get_dataset(dataset_name)
        self.sample_prefix = uploader_info.get_meta_strings("sample_prefix")[0]
        print(f"Uploader sample prefix: {self.sample_prefix}")

        try:
            self.target_prefix = uploader_info.get_meta_strings("target_prefix")[0]
        except:
            self.target_prefix = None
        print(f"Uploader target prefix: {self.target_prefix}")

        try:
            self.producer_prefixes = uploader_info.get_meta_strings("producer_prefix")
        except:
            self.producer_prefixes = [""]
        print(f"Uploader producer prefix: {self.producer_prefixes}")

        try:
            self.num_classes = uploader_info.get_meta_scalars("num_classes")[0]
        except:
            self.num_classes = None
        print(f"Uploader num classes: {self.num_classes}")

        try:
            self.sub_indices = uploader_info.get_meta_strings("sub_indices")
        except:
            self.sub_indices = None
        print(f"Uploader sub-indices: {self.sub_indices}")


    def __len__(self):
        return self.num_samples 


    def __getitem__(self, index):
        # Generate data
        x, y = self.__data_generation(index)

        if y is not None:
            return x, y
        else:
            return x


    def data_exists(self, batch_name, target_name):
        if self.need_targets:
                return (self.client.tensor_exists(batch_name) and self.client.tensor_exists(target_name))
        else:
            return self.client.tensor_exists(batch_name)


    def add_samples(self, batch_name, target_name):
        if self.samples is None:
            self.samples = torch.tensor(self.client.get_tensor(batch_name), device=self.device)
            if self.need_targets:
                self.targets = torch.tensor(self.client.get_tensor(target_name), device=self.device)
        else:
            self.samples = torch.cat((self.samples, torch.tensor(self.client.get_tensor(batch_name), device=self.device)))
            if self.need_targets:
                self.targets = torch.cat((self.targets, torch.tensor(self.client.get_tensor(target_name), device=self.device)))

        self.num_samples = self.samples.shape[0]
        print("Success!")
        print(f"New dataset size: {self.num_samples}")


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
            
            print(f"Retrieving {batch_name} from {entity}")
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
            if self.num_classes is not None:
                y = F.one_hot(y, num_classes=self.num_classes)
        elif self.autoencoding:
            y = x
        else:
            y = None

        return x, y


class DataGenerator(StaticDataGenerator):
    def __init__(self,
                 uploader_info="auto",
                 uploader_name="training_data",
                 sample_prefix="samples",
                 target_prefix="targets",
                 sub_indices=None,
                 num_classes=None,
                 producer_prefixes=None,
                 smartredis_cluster=True,
                 smartredis_address=None,
                 init_samples=True,
                 device="cpu",
                 ):
        super().__init__(uploader_info,
                        uploader_name,
                        sample_prefix,
                        target_prefix,
                        sub_indices,
                        num_classes,
                        producer_prefixes,
                        smartredis_cluster,
                        smartredis_address,
                        init_samples,
                        device
                        )


    def list_all_sources(self):
        sources = super().list_all_sources()
        # Append the batch index to each source
        for source in sources:
            source.append(0)
        return sources


    def init_samples(self, sources=None):
        if sources is None:
            self.sources = self.list_all_sources()

        while len(self) < 1:
            self._update_samples_and_targets()
            if len(self) < 1:
                time.sleep(10)
        print("Generator initialization complete")


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

            print(f"Retrieving {batch_name} from {entity}")
            # Poll next batch based on index, if available: retrieve it, update index and loop
            while self.data_exists(batch_name, target_name):
                self.add_samples(batch_name, target_name)
                source[2] += 1
                index = source[2]
                batch_name = form_name(self.sample_prefix, index, sub_index)
                if self.need_targets:
                    target_name = form_name(self.target_prefix, index, sub_index)

                print(f"Retrieving {batch_name}...")
