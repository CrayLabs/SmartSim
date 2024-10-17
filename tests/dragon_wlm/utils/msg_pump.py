# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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

import io
import logging
import pathlib
import sys
import time
import typing as t

import pytest

pytest.importorskip("torch")
pytest.importorskip("dragon")


# isort: off
import dragon
import multiprocessing as mp
import torch
import torch.nn as nn

# isort: on

from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger

logger = get_logger(__name__, log_level=logging.DEBUG)

# The tests in this file belong to the dragon group
pytestmark = pytest.mark.dragon

try:
    mp.set_start_method("dragon")
except Exception:
    pass


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._name = "mini-model"
        self._net = torch.nn.Linear(2, 1)

    def forward(self, input):
        return self._net(input)

    @property
    def bytes(self) -> bytes:
        """Returns the model serialized to a byte stream"""
        buffer = io.BytesIO()
        scripted = torch.jit.trace(self._net, self.get_batch())
        torch.jit.save(scripted, buffer)
        return buffer.getvalue()

    @classmethod
    def get_batch(cls) -> "torch.Tensor":
        return torch.randn((100, 2), dtype=torch.float32)


def load_model() -> bytes:
    """Create a simple torch model in memory for testing"""
    mini_model = MiniModel()
    return mini_model.bytes


def persist_model_file(model_path: pathlib.Path) -> pathlib.Path:
    """Create a simple torch model and persist to disk for
    testing purposes.

    :returns: Path to the model file
    """
    # test_path = pathlib.Path(work_dir)
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)

    model_path.unlink(missing_ok=True)

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)

    return model_path


def _mock_messages(
    dispatch_fli_descriptor: str,
    fs_descriptor: str,
    parent_iteration: int,
    callback_descriptor: str,
) -> None:
    """Mock event producer for triggering the inference pipeline."""
    model_key = "mini-model"
    # mock_message sends 2 messages, so we offset by 2 * (# of iterations in caller)
    offset = 2 * parent_iteration

    feature_store = BackboneFeatureStore.from_descriptor(fs_descriptor)
    request_dispatcher_queue = DragonFLIChannel.from_descriptor(dispatch_fli_descriptor)

    feature_store[model_key] = load_model()

    for iteration_number in range(2):
        logged_iteration = offset + iteration_number
        logger.debug(f"Sending mock message {logged_iteration}")

        output_key = f"output-{iteration_number}"

        tensor = (
            (iteration_number + 1) * torch.ones((1, 2), dtype=torch.float32)
        ).numpy()
        fsd = feature_store.descriptor

        tensor_desc = MessageHandler.build_tensor_descriptor(
            "c", "float32", list(tensor.shape)
        )

        message_tensor_output_key = MessageHandler.build_tensor_key(output_key, fsd)
        message_model_key = MessageHandler.build_model_key(model_key, fsd)

        request = MessageHandler.build_request(
            reply_channel=callback_descriptor,
            model=message_model_key,
            inputs=[tensor_desc],
            outputs=[message_tensor_output_key],
            output_descriptors=[],
            custom_attributes=None,
        )

        logger.info(f"Sending request {iteration_number} to request_dispatcher_queue")
        request_bytes = MessageHandler.serialize_request(request)

        logger.info("Sending msg_envelope")

        # cuid = request_dispatcher_queue._channel.cuid
        # logger.info(f"\tInternal cuid: {cuid}")

        # send the header & body together so they arrive together
        try:
            request_dispatcher_queue.send_multiple([request_bytes, tensor.tobytes()])
            logger.info(f"\tenvelope 0: {request_bytes[:5]}...")
            logger.info(f"\tenvelope 1: {tensor.tobytes()[:5]}...")
        except Exception as ex:
            logger.exception("Unable to send request envelope")

    logger.info("All messages sent")

    # keep the process alive for an extra 15 seconds to let the processor
    # have access to the channels before they're destroyed
    for _ in range(15):
        time.sleep(1)


def mock_messages(
    dispatch_fli_descriptor: str,
    fs_descriptor: str,
    parent_iteration: int,
    callback_descriptor: str,
) -> int:
    """Mock event producer for triggering the inference pipeline. Used
    when starting using multiprocessing."""
    logger.info(f"{dispatch_fli_descriptor=}")
    logger.info(f"{fs_descriptor=}")
    logger.info(f"{parent_iteration=}")
    logger.info(f"{callback_descriptor=}")

    try:
        return _mock_messages(
            dispatch_fli_descriptor,
            fs_descriptor,
            parent_iteration,
            callback_descriptor,
        )
    except Exception as ex:
        logger.exception()
        return 1

    return 0


if __name__ == "__main__":
    import argparse

    args = argparse.ArgumentParser()

    args.add_argument("--dispatch-fli-descriptor", type=str)
    args.add_argument("--fs-descriptor", type=str)
    args.add_argument("--parent-iteration", type=int)
    args.add_argument("--callback-descriptor", type=str)

    args = args.parse_args()

    return_code = mock_messages(
        args.dispatch_fli_descriptor,
        args.fs_descriptor,
        args.parent_iteration,
        args.callback_descriptor,
    )
    sys.exit(return_code)
