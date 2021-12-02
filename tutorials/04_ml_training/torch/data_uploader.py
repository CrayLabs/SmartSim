from smartsim.ml import TrainingDataUploader
from os import environ
from time import sleep
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

batches_per_loop = 10

data_uploader = TrainingDataUploader(num_classes=mpi_size, smartredis_cluster=False, producer_prefixes="uploader", sub_indices=mpi_size)
if environ["SSKEYOUT"] == "uploader_0":
    data_uploader.publish_info()


for _ in range(15):
    new_batch = np.random.normal(loc=float(mpi_rank), scale=5.0, size=(32*batches_per_loop, 3, 224, 224)).astype(float)
    new_labels = np.ones(shape=(32*batches_per_loop,)).astype(int) * mpi_rank

    data_uploader.put_batch(new_batch, new_labels, sub_index=mpi_rank)
    print(f"{mpi_rank}: New data pushed to DB")
    sleep(120)
