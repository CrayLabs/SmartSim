from os import environ
from smartredis import Client, Dataset
from smartredis.error import RedisReplyError

import numpy as np
import tensorflow.keras as keras
import time

def form_name(*args):
    return "_".join(str(args))

class TrainingDataUploader():
    def __init__(self, 
                 name="training_data",
                 sample_prefix="samples",
                 target_prefix="targets",
                 num_classes=None,
                 producer_prefix="",
                 smartredis_cluster=True,
                 smartredis_address=None):
        if not name:
            raise ValueError("Name can not be empty.")
        if not sample_prefix:
            raise ValueError("Sample prefix can not be empty")

        self.name = name
        self.sample_prefix = sample_prefix
        self.target_prefix = target_prefix
        self.producer_prefix = producer_prefix
        self.num_classes = num_classes

        self.client = Client(address=smartredis_address, cluster=smartredis_cluster)
        self.batch_idx = 0

    def publish_info(self):
        info_ds = Dataset(form_name(self.name, "info"))
        info_ds.add_meta_string("sample_prefix", self.sample_prefix)
        if self.target_prefix:
            info_ds.add_meta_string("label_prefix", self.target_prefix)
        if self.producer_prefix:
            info_ds.add_meta_string("producer_prefix", self.producer_prefix)
        if self.num_classes:
            info_ds.add_meta_scalar("num_classes", self.num_classes)
        self.client.put_dataset(info_ds)

    def put_batch(self, samples, targets=None):

        batch_key = form_name(self.batch_prefix, str(self.batch_idx))
        self.client.put_tensor(batch_key, samples)

        if targets and self.target_prefix and (self.target_prefix != self.sample_prefix):
            labels_key = form_name(self.target_prefix, str(self.batch_idx))
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
                 num_classes=None,
                 producer_prefix=None,
                 smartredis_cluster=True,
                 smartredis_address=None,
                 ):


        self.batch_size = batch_size
        self.shuffle = shuffle
        self.client = Client(smartredis_address, smartredis_cluster)
        if uploader_info == "manual":
            self.sample_prefix = sample_prefix
            self.target_prefix = target_prefix
            self.producer_prefix = producer_prefix
            self.num_classes = num_classes
        elif uploader_info == "auto":
            if not uploader_name:
                raise ValueError("uploader_name can not be empty if uploader_info is 'auto'")
            self.get_uploader_info(uploader_name)
        else:
            raise ValueError(f"uploader_info must be one of 'auto' or 'manual', but was {uploader_info}")

        self.autoencoding = (self.sample_prefix == self.target_prefix)

        self.next_index = {}
        for entity_name in environ["SSKEYIN"].split(','):
            if entity_name.startswith(self.producer_prefix):
                self.next_index[entity_name] = 0
        self.samples = None
        if self.need_targets:
            self.targets = None
        
        self.indices = None
        while self.samples is None:
            self.on_epoch_end()
            if not self.samples:
                time.sleep(10)


    @property
    def need_targets(self):
        return self.target_prefix and not self.autoencoding


    def get_uploader_info(self, uploader_name):
        dataset_name = form_name(uploader_name, "info")
        while not self.client.dataset_exists(dataset_name):
            time.sleep(10)
        
        uploader_info = self.client.get_dataset(dataset_name)
        self.sample_prefix = uploader_info.get_meta_strings("sample_prefix")
        try:
            self.target_prefix = uploader_info.get_meta_strings("target_prefix")
        except RedisReplyError:
            self.target_prefix = None
        try:
            self.producer_prefix = uploader_info.get_meta_strings("producer_prefix")
        except RedisReplyError:
            self.producer_prefix = ""
        try:
            self.num_classes = uploader_info.get_meta_scalars("num_classes")
        except RedisReplyError:
            self.num_classes = None


    def __len__(self):
        length = int(np.floor(self.num_samples / self.batch_size))
        return length


    def __getitem__(self, index):
        # Generate indices of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        x, y = self.__data_generation(indices)

        if y:
            return x, y
        else:
            return x


    def get_samples(self, entity):
        batch_name = form_name(self.sample_prefix, self.next_index[entity])
        print(f"Retrieving {batch_name} from {entity}...")
        while self.client.tensor_exists(batch_name):
            if self.samples is None:
                self.samples = self.client.get_tensor(batch_name)
                self.num_samples = self.samples.shape[1:]
            else:
                self.samples = np.concatenate((self.samples,self.client.get_tensor(batch_name)))
            print("Success!")
            self.next_index[entity] += 1
            batch_name = form_name(self.sample_prefix, self.next_index[entity])
            print(f"Retrieving {batch_name}...")


    def get_samples_and_targets(self, entity):
        batch_name = form_name(self.sample_prefix, self.next_index[entity])
        target_name = form_name(self.target_prefix, self.next_index[entity])
        print(f"Retrieving {batch_name} from {entity}...")
        while self.client.tensor_exists(batch_name) and self.client.tensor_exists(target_name):
            if self.samples is None:
                self.samples = self.client.get_tensor(batch_name)
                self.labels = self.client.get_tensor(target_name)
                self.num_samples = self.samples.shape[1:]
            else:
                self.samples = np.concatenate((self.samples,self.client.get_tensor(batch_name)))
                self.labels = np.concatenate((self.labels, self.client.get_tensor(target_name)))
            print("Success!")
            self.next_index[entity] += 1
            batch_name = form_name(self.sample_prefix, self.next_index[entity])
            target_name = form_name(self.target_prefix, self.next_index[entity])
            print(f"Retrieving {batch_name} and {target_name}...")


    def on_epoch_end(self):

        for entity in self.next_index:
            self.client.set_data_source(entity)
            if self.need_targets:
                self.get_samples_and_targets(entity)
            else:
                self.get_samples(entity)
                
        self.indices = np.arange(self.num_samples)
        print(f"New dataset size: {self.num_samples}")
        if self.shuffle:
            np.random.shuffle(self.indices)
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

