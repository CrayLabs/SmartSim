from os import environ
from smartredis import Client, Dataset
from smartredis.error import RedisReplyError


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