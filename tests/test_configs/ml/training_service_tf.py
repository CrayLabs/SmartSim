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

import tensorflow.keras as keras

from smartsim.ml.tf import DynamicDataGenerator


def check_dataloader(dl, rank):
    assert dl.uploader_name == "test_data"
    assert dl.sample_prefix == "test_samples"
    assert dl.target_prefix == "test_targets"
    assert dl.uploader_info == "auto"
    assert dl.num_classes == 2
    assert dl.producer_prefixes == ["test_uploader"]
    assert dl.sub_indices == ["0", "1"]
    assert dl.verbose == True
    assert dl.replica_rank == rank
    assert dl.num_replicas == 2
    assert dl.address == None
    assert dl.cluster == False
    assert dl.shuffle == True
    assert dl.batch_size == 4
    assert len(dl.sources) == 2
    for i in range(2):
        assert dl.sources[i][0:2] == [f"test_uploader_{rank}", str(i)]
        assert type(dl.sources[i][2]) == int


# Pretend we are running distributed without requiring Horovod
hvd_size = 2


def create_data_generator(rank):
    return DynamicDataGenerator(
        cluster=False,
        uploader_name="test_data",
        verbose=True,
        num_replicas=2,
        replica_rank=rank,
        batch_size=4,
    )


training_generators = [create_data_generator(rank) for rank in range(hvd_size)]

[
    check_dataloader(training_generator, rank)
    for (rank, training_generator) in enumerate(training_generators)
]

model = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            filters=4, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 1)
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(training_generators[0].num_classes, activation="softmax"),
    ]
)

model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

print("Starting training")

for epoch in range(2):
    print(f"Epoch {epoch}")
    for rank in range(hvd_size):
        model.fit(
            training_generators[rank],
            steps_per_epoch=None,
            epochs=epoch + 1,
            initial_epoch=epoch,
            batch_size=training_generators[rank].batch_size,
            verbose=2,
        )

assert all([len(training_generators[rank]) == 4 for rank in range(hvd_size)])
