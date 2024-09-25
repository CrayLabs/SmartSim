# # BSD 2-Clause License
# #
# # Copyright (c) 2021-2024, Hewlett Packard Enterprise
# # All rights reserved.
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# #    list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# #    this list of conditions and the following disclaimer in the documentation
# #    and/or other materials provided with the distribution.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# import io
# import logging
# import pathlib
# import time

# import pytest

# torch = pytest.importorskip("torch")
# dragon = pytest.importorskip("dragon")

# import base64
# import multiprocessing as mp

# try:
#     mp.set_start_method("dragon")
# except Exception:
#     pass

# import os

# import dragon.channels as dch
# from dragon import fli
# from dragon.mpbridge.queues import DragonQueue

# from smartsim._core.mli.comm.channel.channel import CommChannelBase
# from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
# from smartsim._core.mli.infrastructure.control.worker_manager import (
#     EnvironmentConfigLoader,
#     WorkerManager,
# )
# from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
#     DragonFeatureStore,
# )
# from smartsim._core.mli.infrastructure.storage.feature_store import FeatureStore
# from smartsim._core.mli.infrastructure.worker.torch_worker import TorchWorker
# from smartsim._core.mli.message_handler import MessageHandler
# from smartsim.log import get_logger

# from .feature_store import FileSystemFeatureStore
# from .utils.channel import FileSystemCommChannel

# logger = get_logger(__name__)
# # The tests in this file belong to the dragon group
# pytestmark = pytest.mark.dragon


# def persist_model_file(model_path: pathlib.Path) -> pathlib.Path:
#     """Create a simple torch model and persist to disk for
#     testing purposes.

#     TODO: remove once unit tests are in place"""
#     # test_path = pathlib.Path(work_dir)
#     if not model_path.parent.exists():
#         model_path.parent.mkdir(parents=True, exist_ok=True)

#     model_path.unlink(missing_ok=True)
#     # model_path = test_path / "basic.pt"

#     model = torch.nn.Linear(2, 1)
#     torch.save(model, model_path)

#     return model_path


# def mock_messages(
#     worker_manager_queue: CommChannelBase,
#     feature_store: FeatureStore,
#     feature_store_root_dir: pathlib.Path,
#     comm_channel_root_dir: pathlib.Path,
# ) -> None:
#     """Mock event producer for triggering the inference pipeline"""
#     feature_store_root_dir.mkdir(parents=True, exist_ok=True)
#     comm_channel_root_dir.mkdir(parents=True, exist_ok=True)

#     model_path = persist_model_file(feature_store_root_dir.parent / "model_original.pt")
#     model_bytes = model_path.read_bytes()
#     model_key = str(feature_store_root_dir / "model_fs.pt")

#     feature_store[model_key] = model_bytes

#     iteration_number = 0

#     while True:
#         iteration_number += 1
#         time.sleep(1)
#         # 1. for demo, ignore upstream and just put stuff into downstream
#         # 2. for demo, only one downstream but we'd normally have to filter
#         #       msg content and send to the correct downstream (worker) queue
#         # timestamp = time.time_ns()
#         # mock_channel = test_path / f"brainstorm-{timestamp}.txt"
#         # mock_channel.touch()

#         # thread - just look for key (wait for keys)
#         # call checkpoint, try to get non-persistent key, it blocks
#         # working set size > 1 has side-effects
#         # only incurs cost when working set size has been exceeded

#         channel_key = comm_channel_root_dir / f"{iteration_number}/channel.txt"
#         callback_channel = FileSystemCommChannel(pathlib.Path(channel_key))

#         input_path = feature_store_root_dir / f"{iteration_number}/input.pt"
#         output_path = feature_store_root_dir / f"{iteration_number}/output.pt"

#         input_key = str(input_path)
#         output_key = str(output_path)

#         buffer = io.BytesIO()
#         tensor = torch.randn((1, 2), dtype=torch.float32)
#         torch.save(tensor, buffer)
#         feature_store[input_key] = buffer.getvalue()
#         fsd = feature_store.descriptor

#         message_tensor_output_key = MessageHandler.build_tensor_key(output_key, fsd)
#         message_tensor_input_key = MessageHandler.build_tensor_key(input_key, fsd)
#         message_model_key = MessageHandler.build_model_key(model_key, fsd)

#         request = MessageHandler.build_request(
#             reply_channel=callback_channel.descriptor,
#             model=message_model_key,
#             inputs=[message_tensor_input_key],
#             outputs=[message_tensor_output_key],
#             output_descriptors=[],
#             custom_attributes=None,
#         )
#         request_bytes = MessageHandler.serialize_request(request)
#         worker_manager_queue.send(request_bytes)


# @pytest.fixture
# def prepare_environment(test_dir: str) -> pathlib.Path:
#     """Cleanup prior outputs to run demo repeatedly"""
#     path = pathlib.Path(f"{test_dir}/workermanager.log")
#     logging.basicConfig(filename=path.absolute(), level=logging.DEBUG)
#     return path


# def test_worker_manager(prepare_environment: pathlib.Path) -> None:
#     """Test the worker manager"""

#     test_path = prepare_environment
#     fs_path = test_path / "feature_store"
#     comm_path = test_path / "comm_store"

#     to_worker_channel = dch.Channel.make_process_local()
#     to_worker_fli = fli.FLInterface(main_ch=to_worker_channel, manager_ch=None)
#     to_worker_fli_serialized = to_worker_fli.serialize()

#     # NOTE: env vars should be set prior to instantiating EnvironmentConfigLoader
#     # or test environment may be unable to send messages w/queue
#     descriptor = base64.b64encode(to_worker_fli_serialized).decode("utf-8")
#     os.environ["_SMARTSIM_REQUEST_QUEUE"] = descriptor

#     config_loader = EnvironmentConfigLoader(
#         featurestore_factory=DragonFeatureStore.from_descriptor,
#         callback_factory=FileSystemCommChannel.from_descriptor,
#         queue_factory=DragonFLIChannel.from_descriptor,
#     )
#     integrated_worker_type = TorchWorker

#     worker_manager = WorkerManager(
#         config_loader,
#         integrated_worker_type,
#         as_service=True,
#         cooldown=5,
#         device="cpu",
#         dispatcher_queue=mp.Queue(maxsize=0),
#     )

#     worker_queue = config_loader.get_queue()
#     if worker_queue is None:
#         logger.warn(
#             f"FLI input queue not loaded correctly from config_loader: {config_loader._queue_descriptor}"
#         )

#     # create a mock client application to populate the request queue
#     msg_pump = mp.Process(
#         target=mock_messages,
#         args=(
#             worker_queue,
#             FileSystemFeatureStore(fs_path),
#             fs_path,
#             comm_path,
#         ),
#     )
#     msg_pump.start()

#     # create a process to execute commands
#     process = mp.Process(target=worker_manager.execute)
#     process.start()
#     process.join(timeout=5)
#     process.kill()
#     msg_pump.kill()
