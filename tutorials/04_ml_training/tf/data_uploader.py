from smartsim.ml import TrainingDataUploader
from os import environ
from time import sleep
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

batches_per_loop = 10

data_uploader = TrainingDataUploader(num_classes=mpi_size,
                                     smartredis_cluster=False, 
                                     producer_prefixes="uploader",
                                     num_ranks=mpi_size,
                                     rank=mpi_rank)

if environ["SSKEYOUT"] == "uploader_0" and mpi_rank==0:
    data_uploader.publish_info()


# Start "simulation", produce data every two minutes, for thirty minutes
for _ in range(15):
    new_batch = np.random.normal(loc=float(mpi_rank), scale=5.0, size=(32*batches_per_loop, 224, 224, 3)).astype(float)
    new_labels = np.ones(shape=(32*batches_per_loop,)).astype(int) * mpi_rank

    data_uploader.put_batch(new_batch, new_labels)
    print(f"{mpi_rank}: New data pushed to DB")
    sleep(120)
