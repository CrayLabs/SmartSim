from gc import callbacks
import argparse
import logging
import pathlib
import random
import subprocess
import sys
import shutil
import time
import typing as t
import uuid

import dragon
import dragon.infrastructure
import dragon.channels as dch
import dragon.infrastructure.facts as df
import dragon.infrastructure.parameters as dp
import dragon.managed_memory as dm
import dragon.utils as du
from dragon.data.distdictionary.dragon_dict import DragonDict

import multiprocessing as mp
import tensorflow as tf
import torch
import io
import os
import json


torch_model_type = "PyTorch"
torch_model_name = "pytorch-demo-model"

ts = time.time_ns()
pid = os.getpid()
logger = logging.getLogger()

output_dir = pathlib.Path("mli")
if not output_dir.exists():
    output_dir.mkdir()

logging.basicConfig(
    level="DEBUG", filename=str(output_dir / f"log.{ts}.{pid}.txt"), filemode="w"
)

WorkerFn = t.Callable[[mp.Queue, bytes, DragonDict], None]
ResultFn = t.Callable[[bytes, int], None]


class TorchNet(torch.nn.Module):
    def __init__(self) -> None:
        super(TorchNet, self).__init__()
        self._input = torch.nn.Linear(2, 16)
        self._fc0 = torch.nn.Linear(16, 8)
        self._fc1 = torch.nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v0 = self._input(x)
        v1 = self._fc0(v0)
        v2 = self._fc1(v1)
        return v2


class ModelRegistration:
    def __init__(self, name: str, path: pathlib.Path, model_type: str) -> None:
        self.name = name
        self.path = path
        self.model_type = model_type

    @property
    def key(self) -> str:
        return f"{self.model_type}:{self.name}"

    @property
    def model(self) -> bytes:
        # with open(self.path, "rb") as fp:
        return self.path.read_bytes()

    def __str__(self) -> str:
        return f"{self.name} @ {self.path}"


def torch_worker(
    queue: mp.Queue,
    pool_id: bytes,
    feature_store: DragonDict,
) -> None:
    """A worker that handles requests for inference using a PyTorch model"""
    logger.info("PyTorch worker started")

    # NOTE: we need to be able to isolate this or ensure the worker environment
    # matches the user needs.
    # a docker container or pyenv may work....

    # pool_id = du.B64.str_to_bytes(dp.this_process.default_pd)
    mem_pool = dm.MemoryPool.attach(pool_id)

    running = True

    while running:
        if queue.empty():
            logger.info("Torch worker queue is empty. Sleeping...")
            time.sleep(10)
            continue

        if msg := queue.get_nowait():
            if msg == "stop":
                running = False
                continue

            logger.info(f"Received incoming message: {msg}")

            ts = msg["ts"]
            callback_id = int(
                msg["callback_cuid"]
            )  # getting this from msg to facilitate interleaving requests...
            # but _do i need to_? i may be able to avoid cuid in msg if each callback is 1:1
            # NOTE: should we allow users to get multiple messages in one callback channel (e.g. if i have multiple
            # inferences occurring, do they need two callback channels or can they differentiate)
            model_name = msg["model_name"]
            model_key = msg["model_key"]
            tensor = msg["x"]
            request_id = msg["request_id"]
            # pool_id = msg["pool_id"]

            logger.info(f"Performing inference w/{model_name}. input: {tensor}")

            y = "not-yet-computed"
            inference_key = ""

            try:
                model_bytes = feature_store[model_key]
                io_bytes = io.BytesIO(model_bytes)
                model = torch.load(io_bytes, weights_only=False)
                model.eval()

                y = model(tensor)

                inference_key = f"{model_key}.{str(uuid.uuid4())}"
                feature_store[inference_key] = y

                result = str(
                    {
                        "ts": ts,
                        "request_id": request_id,
                        "model_key": model_key,
                        "value_key": inference_key,
                        # "y": y,
                    }
                ).encode("utf-8")

            # except KeyError:
            #     # model wasn't in the feature_store
            except:
                logger.error("An error occurred during inference", exc_info=True)
                continue

            try:
                callback_channel = dch.Channel(mem_pool, callback_id)
                writer = dch.ChannelSendH(callback_channel)
                writer.open()

                writer.send_bytes(result)  # inference_key.encode("utf-8"))
                logger.info(
                    f"Completed inference for `{msg}`. Response `{inference_key}` sent to callback channel"
                )

                writer.close()
            except:
                logger.error(
                    f"An error occurring while respoding through callback channel for `{msg}`"
                )
                continue


# def worker_manager(queue: mp.Queue) -> None:
#     logger.info("starting worker_manager")

#     while True:
#         rest = random.random() * 5
#         time.sleep(rest)

#         if queue.empty():
#             continue

#         if msg := queue.get_nowait():
#             logger.info(f"received {msg}")


def mock_simulation(queue: mp.Queue, pool_id: bytes) -> None:
    """Mock a simulation that makes requests for inference. Publishes a
    series of messages to a queue that will trigger an ML worker"""
    logger.info("Starting mock_simulation")

    # pool_id = du.B64.str_to_bytes(dp.this_process.default_pd)
    # pool_id = du.B64.str_to_bytes("mock_simulation_mem_pool")
    mem_pool = dm.MemoryPool.attach(pool_id)

    # todo: ask Dragon team about CUID definitions... is it unique per process and ok
    # to reuse or am i "getting lucky" by using the same id everywhere in this demo?
    simulation_callback_cuid = df.BASE_USER_MANAGED_CUID
    logger.debug(f"{simulation_callback_cuid=}")

    callback_channel = dch.Channel(mem_pool, simulation_callback_cuid)
    callback_channel = dch.ChannelRecvH(callback_channel)
    callback_channel.open()

    # create a mock message consumer that will handle results of the workers
    handler = mp.Process(
        target=mock_result_handler, args=(pool_id, simulation_callback_cuid)
    )
    handler.start()

    try:
        # while True:
        for i in range(6):
            time.sleep(5)
            msg = {
                "ts": time.time_ns(),
                "callback_cuid": simulation_callback_cuid,
                "model_name": torch_model_name,
                "model_key": f"{torch_model_type}:{torch_model_name}",
                "x": torch.rand((1, 2)),
                # request_id does the same thing as ts for now. pick one...
                "request_id": str(uuid.uuid4()),
                "pool_id": pool_id,
            }
            queue.put_nowait(str(msg))
            logger.info(f"Message `{msg['ts']}` placed into producer queue")
    finally:
        callback_channel.close()


def mock_result_handler(pool_id: bytes, callback_cuid: int) -> None:
    """Listen to a callback channel for confirmation that a worker has
    completed a requested inference operation"""
    # create the channel for receiving callback messages from the workers
    # NOTE: the pool id must be known by the result handler at the time this is
    # set up because i can't look at a message that i can't configure a channel
    # to receive...
    # pool_id = du.B64.str_to_bytes(pool_id)
    mem_pool = dm.MemoryPool.attach(pool_id)
    # pool_id = du.B64.str_to_bytes("mock_simulation_mem_pool")
    mem_pool = dm.MemoryPool.attach(pool_id)

    callback_channel = dch.Channel(mem_pool, callback_cuid)
    reader = dch.ChannelRecvH(callback_channel)
    reader.open()

    # while True:
    for i in range(15):
        time.sleep(2)
        try:
            last_msg: bytes = reader.recv_bytes(timeout=None, blocking=False)
            if last_msg:
                logger.info(f"Received result message from worker: {last_msg}")
            else:
                logger.info(f"No messages received by the handler")

            msg = json.decode(last_msg.decode("utf-8"))

        except dch.ChannelEmpty:
            # logger.error("Error checking control channel.", exc_info=True)
            ...
        except:
            logger.warning(
                "An exception occurred when reading callback channel.", exc_info=True
            )
            # swallow timeouts


class WorkStream:
    def __init__(
        self,
        input_queue: mp.Queue,
        worker_fn: WorkerFn,
        # mem_pool: dm.MemoryPool,
        pool_id: bytes,
        callback_cuid: int,
        worker_type: str,
        feature_store: DragonDict,
    ) -> None:
        self.worker_type = worker_type
        """Friendly name to reference the worker type"""
        self.input_queue = input_queue
        """A queue used to communicate work requests to MLI"""

        self.queue = mp.Queue()
        """A queue used to communicate with the worker"""
        self.worker_fn = worker_fn
        """A function that will process events sent through the queue"""
        self.worker: t.Optional[mp.Process] = None
        """Reference to the running worker"""
        self.pool_id: bytes = pool_id
        """The shared memory pool for placing data"""
        self.callback_cuid = callback_cuid
        """The channel ID the worker will use to return results to a task originator"""
        self.feature_store = feature_store

        # IDEA: maybe we also try to remove indirection and have
        # client messages get forwarded directly to the worker somehow?
        # - could be that the incoming queue is sent to worker and the
        #   worker starts listening to it instead of the MLI manager?
        # self.callback_channel: t.Optional[dch.Channel] = None
        # """A channel used to communicate directly with the worker"""

    def start(self) -> None:
        """Start the associated worker"""
        try:
            self.worker = mp.Process(
                target=self.worker_fn,
                args=(
                    self.queue,
                    self.pool_id,
                    self.feature_store,
                ),
            )
            self.worker.start()
        except:
            logging.error("An error occurred while starting a worker", exc_info=True)


def initialize_workers(
    pool_id: bytes,
    pt_queue: mp.Queue,
    pt_cuid: int,
    feature_store: DragonDict,
) -> t.List[WorkStream]:
    """
    Create a static set of worker resources required to support MLI features
    """
    # create a PyTorch worker
    pt_worker = WorkStream(
        pt_queue, torch_worker, pool_id, pt_cuid, torch_model_type, feature_store
    )
    pt_worker.start()

    return [pt_worker]


def work_it(workers: t.List[WorkStream], mem_pool: dm.MemoryPool, control_cuid: int):
    """Run an infinite event loop until terminated or a control message is received
    that asks for a shutdown.

    Monitors the work input queues and transforms/forwards messages to the correct
    worker for the work"""
    running = True
    # MOCKS out the simulation doing whatever it does and sending messages to
    # workers through queues

    # pool_id = du.B64.str_to_bytes(dp.this_process.default_pd)
    # mem_pool = dm.MemoryPool.attach(pool_id)

    # todo: chris -> the channel should be created by the "final resting place, not here"
    # make sure we only pass in the cuid instead...
    # create an input stream that allows work_it to be terminated on demand
    control_channel = dch.Channel(mem_pool, control_cuid)
    control_reader = dch.ChannelRecvH(control_channel)
    control_reader.open()

    logger.info(f"MLI control channel CUID is: {control_cuid}")

    while running:
        # todo: consider starting individual threads for queue monitoring, too.
        for worker in workers:
            if worker.input_queue.empty():
                # logger.info(
                #     f"No messages in input_queue for {worker.worker_type} worker"
                # )
                continue

            logger.info("worker input_queue is not empty.")
            if input_msg := worker.input_queue.get_nowait():
                # for now, just forward the original message to the worker
                logger.info(
                    f"Put input_msg `{input_msg}` for `{worker.worker_type}` onto queue..."
                )
                worker.queue.put(input_msg)

        try:
            if last_msg := control_reader.recv_bytes(timeout=None, blocking=False):
                logger.info(f"Received termination message on control channel")
                running = False
        except dch.ChannelEmpty:
            # logger.error("Error checking control channel.", exc_info=True)
            ...
        except:
            logger.error("Error checking control channel.", exc_info=True)


def register_model_torch(output_dir: pathlib.Path) -> ModelRegistration:
    # device = "CPU"
    nn = TorchNet()
    nn.eval()

    # perform a prediction so the network is ready...
    x = torch.rand((1, 2))
    _ = nn(x)
    # traced = torch.jit.trace(nn, x)

    # buffer = io.BytesIO()
    # torch.jit.save(traced, buffer)
    path = pathlib.Path(output_dir / f"{torch_model_name}.pt")
    if path.exists():
        path.unlink()

    with open(path, "wb") as fp:
        torch.save(nn, fp)
        logger.debug(f"Model saved to {path}")

    registration = ModelRegistration(torch_model_name, path, torch_model_type)
    logger.debug(f"Model registered: {registration}")
    return registration


def register_model_tf(output_dir: pathlib.Path) -> ModelRegistration:
    name = "tf-demo-model"
    # device = "CPU"
    nn = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(16, input_shape=(2,)),
            tf.keras.layers.Dense(8),
            tf.keras.layers.Dense(1),
        ]
    )
    nn.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    # nn.eval()

    # perform a prediction so the network is ready...
    x = tf.random.normal([1000, 2], 0.5, 0.1, tf.float32, seed=1)
    _ = nn(x)
    # traced = torch.jit.trace(nn, x)

    # dir_path = pathlib.Path()
    model_dir = output_dir / name
    full_path = model_dir / "saved_model.pb"

    # cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=path, save_weights_only=True, verbose=1
    # )

    data = tf.random.normal([1000, 2], 0.5, 0.1, tf.float32, seed=1)
    values = tf.random.normal([1000, 1], 0.5, 0.1, tf.float32, seed=1)

    val_data = tf.random.normal([100, 2], 0.5, 0.1, tf.float32, seed=1)
    val_values = tf.random.normal([100, 1], 0.5, 0.1, tf.float32, seed=1)

    nn.fit(
        data,
        values,
        epochs=1,
        validation_data=(val_data, val_values),
        # callbacks=[cp_callback],
    )

    if full_path.exists():
        full_path.unlink()
        # shutil.rmtree(full_path.parent)

    # nn.save_weights(path)
    nn.save(model_dir)
    if not full_path.exists():
        raise FileNotFoundError(f"{full_path} was not written")

    registration = ModelRegistration(name, full_path, "Tensorflow")
    logger.debug(f"Model registered: {registration}")
    return registration


def register_models(
    output_dir: pathlib.Path,
    feature_store: DragonDict,
) -> t.Tuple[ModelRegistration, ModelRegistration]:
    logger.debug("Creating PyTorch model registration")
    registration1 = register_model_torch(output_dir)
    logger.info(
        f"Persisting model `{registration1.name}` to feature store key `{registration1.key}`"
    )
    feature_store[registration1.key] = registration1.model

    # # ask the dragon team if the channel avoids a mem copy that the queue cannot
    # my_bytes = iter(registration1.model)
    # some_channel.write_bytes(my_bytes)

    logger.debug("Creating TF model registration")
    registration2 = register_model_tf(output_dir)
    logger.info(
        f"Persisting model `{registration2.name}` to feature store key `{registration2.key}`"
    )
    feature_store[registration2.key] = registration2.model

    return registration1, registration2


def verify_models(
    feature_store: DragonDict,
    registrations: t.Tuple[ModelRegistration, ModelRegistration],
):
    for registration in registrations:
        try:
            logger.info(
                f"Confirm model storage... retrieving key: `{registration.key}`"
            )
            value = feature_store[registration.key]
        except:
            logger.error(f"Retrieving key {registration.key} failed.", exc_info=True)


def main():  # args: argparse.Namespace):
    # register_models(output_dir)

    # os.environ["SLURM_JOB_ID"] = "488748"
    mp.set_start_method("dragon")
    logger.info("dragon mp configured")

    feature_store = DragonDict(1, 1, 1024 * 1024 * 1024)

    registrations = register_models(output_dir, feature_store)
    verify_models(feature_store, registrations)

    logger.debug("Attaching to memory pool")
    pool_id = du.B64.str_to_bytes(dp.this_process.default_pd)
    # pool_id = du.B64.str_to_bytes("mock_simulation_mem_pool")
    pool = dm.MemoryPool.attach(pool_id)

    # reserve a channel ID for receiving control messages
    mli_control_cuid = df.BASE_USER_MANAGED_CUID
    logger.debug(f"{mli_control_cuid=}")

    # create a mock message producer that will pretend to be a running experiment
    producer_pt_queue = mp.Queue()
    pt_cuid = df.BASE_USER_MANAGED_CUID + 1
    logger.debug(f"{pt_cuid=}")
    producer = mp.Process(target=mock_simulation, args=(producer_pt_queue, pool_id))
    producer.start()

    # create workers to handle inference requests
    workers = initialize_workers(pool_id, producer_pt_queue, pt_cuid, feature_store)

    # execute the "infinite" event loop
    work_it(workers, pool, mli_control_cuid)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--root",
    #     default=False,
    #     action="store_true",
    #     help="Run as root process of demo to remove temporary files.",
    # )
    # args = parser.parse_args(sys.argv[1:])
    # main(args)
    main()
