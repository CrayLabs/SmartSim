from argparse import ArgumentParser
from os import environ

import numpy as np

from smartsim.ml import TrainingDataUploader

# simulate multi-rank without requesting mpi4py as dep
mpi_size = 2


def create_data_uploader(rank):
    return TrainingDataUploader(
        name="test_data",
        sample_prefix="test_samples",
        target_prefix="test_targets",
        num_classes=mpi_size,
        producer_prefixes="test_uploader",
        cluster=False,
        address=None,
        num_ranks=mpi_size,
        rank=rank,
        verbose=True,
    )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--format", default="tf", type=str)
    args = parser.parse_args()
    format = args.format

    batch_size = 4
    data_uploaders = [create_data_uploader(rank) for rank in range(mpi_size)]

    print(environ["SSKEYOUT"])
    if environ["SSKEYOUT"] == "test_uploader_0":
        data_uploaders[0].publish_info()

    batches_per_loop = 1
    shape = (
        (batch_size * batches_per_loop, 32, 32, 1)
        if format == "tf"
        else (batch_size * batches_per_loop, 1, 32, 32)
    )

    # Start "simulation", produce data every two minutes, for thirty minutes
    for _ in range(2):
        for mpi_rank in range(mpi_size):
            new_batch = np.random.normal(
                loc=float(mpi_rank), scale=5.0, size=shape
            ).astype(float)
            new_labels = (
                np.ones(shape=(batch_size * batches_per_loop,)).astype(int) * mpi_rank
            )

            data_uploaders[mpi_rank].put_batch(new_batch, new_labels)
            print(f"{mpi_rank}: New data pushed to DB")
