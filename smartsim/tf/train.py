from os import environ
from smartredis import Client, Dataset
from smartredis.error import RedisReplyError
from smartsim.utils import get_logger

import numpy as np
import tensorflow.keras as keras
import time

logger = get_logger(__name__)

def form_name(*args):
    return "_".join(str(arg) for arg in args if arg is not None)


class TrainingDataUploader():
    def __init__(self, 
                 name="training_data",
                 sample_prefix="samples",
                 target_prefix="targets",
                 num_classes=None,
                 producer_prefixes=None,
                 smartredis_cluster=True,
                 smartredis_address=None,
                 sub_indices=None):
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
        print(f"Put batch {batch_key}")

        if targets is not None and self.target_prefix and (self.target_prefix != self.sample_prefix):
            labels_key = form_name(self.target_prefix, self.batch_idx, sub_index)
            self.client.put_tensor(labels_key, targets)
        
        self.batch_idx += 1


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
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
                 init_samples=True,
                 ):

        self.batch_size = batch_size
        self.shuffle = shuffle
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
        self.indices = None
        self.num_samples = 0

        if init_samples:
            self.init_samples(None)


    def list_all_sources(self):
        uploaders = environ["SSKEYIN"].split(',')
        sources = []
        for uploader in uploaders:
            if any([uploader.startswith(producer_prefix) for producer_prefix in self.producer_prefixes]):
                if self.sub_indices:
                    sources.extend([[uploader, sub_index, 0] for sub_index in self.sub_indices])
                else:
                    sources.append([uploader, None, 0])

        return sources


    def init_samples(self, sources=None):
        self.next_index = list()
        if sources is None:
            self.sources = self.list_all_sources()

        while len(self) < 1:
            self.on_epoch_end()
            if len(self) < 1:
                time.sleep(10)
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
        length = int(np.floor(self.num_samples / self.batch_size))
        return length


    def __getitem__(self, index):
        if len(self) < 1:
            raise ValueError("Not enough samples in generator for one batch. Please run init_samples() or initialize generator with init_samples=True")
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        x, y = self.__data_generation(indices)

        if y is not None:
            return x, y
        else:
            return x


    def get_samples(self):
        for source in self.sources:
            entity = source[0]
            sub_index = source[1]
            index = source[2]
            self.client.set_data_source(entity)
            batch_name = form_name(self.sample_prefix, index, sub_index)
            print(f"Retrieving {batch_name} from {entity}...")
            while self.client.tensor_exists(batch_name):
                if self.samples is None:
                    self.samples = self.client.get_tensor(batch_name)
                else:
                    self.samples = np.concatenate((self.samples,self.client.get_tensor(batch_name)))
                self.num_samples = self.samples.shape[0]
                self.indices = np.arange(self.num_samples)
                print("Success!")
                print(f"New dataset size: {self.num_samples}")
                if self.shuffle:
                    np.random.shuffle(self.indices)
                source[2] += 1
                index = source[2]
                batch_name = form_name(self.sample_prefix, index, sub_index)
                
                print(f"Retrieving {batch_name}...")


    def get_samples_and_targets(self):
        for source in self.sources:
            entity = source[0]
            sub_index = source[1]
            index = source[2]
            self.client.set_data_source(entity)
            batch_name = form_name(self.sample_prefix, index, sub_index)
            target_name = form_name(self.target_prefix, index, sub_index)
            
            print(f"Retrieving {batch_name} and {target_name} from {entity}")
            # Poll next batch based on index, if available: retrieve it, update index and loop
            while self.client.tensor_exists(batch_name) and self.client.tensor_exists(target_name):
                if self.samples is None:
                    self.samples = self.client.get_tensor(batch_name)
                    self.targets = self.client.get_tensor(target_name)
                else:
                    self.samples = np.concatenate((self.samples,self.client.get_tensor(batch_name)))
                    self.targets = np.concatenate((self.targets, self.client.get_tensor(target_name)))
                self.num_samples = self.samples.shape[0]
                self.indices = np.arange(self.num_samples)
                print("Success!")
                print(f"New dataset size: {self.num_samples}")
                if self.shuffle:
                    np.random.shuffle(self.indices)
                source[2] += 1
                index = source[2]
                batch_name = form_name(self.sample_prefix, index, sub_index)
                target_name = form_name(self.target_prefix, index, sub_index)
                print(f"Retrieving {batch_name} and {target_name}...")


    def on_epoch_end(self):
        if self.need_targets:
            self.get_samples_and_targets()
        else:
            self.get_samples()
        
        self.batch_id = 0


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

