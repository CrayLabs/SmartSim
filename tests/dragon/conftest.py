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

from __future__ import annotations

import pathlib
import subprocess
import sys
import typing as t

import pytest

dragon = pytest.importorskip("dragon")

# isort: off
import dragon.data.ddict.ddict as dragon_ddict

from dragon.channels import Channel
from dragon.data.ddict.ddict import DDict
from dragon.fli import FLInterface

# isort: on

from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.storage import dragon_util
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim._core.mli.infrastructure.storage.dragon_feature_store import (
    DragonFeatureStore,
)

class MsgPumpRequest(t.NamedTuple):
    """Fields required for starting a simulated inference request producer."""

    backbone_descriptor: str
    """The descriptor to use when connecting the message pump to a 
    backbone featurestore.
    
    Passed to the message pump as `--fs-descriptor`
    """
    work_queue_descriptor: str
    """The descriptor to use for sending work from the pump to the worker manager.
    
    Passed to the message pump as `--dispatch-fli-descriptor`
    """
    callback_descriptor: str
    """The descriptor the worker should use to returning results.
    
    Passed to the message pump as `--callback-descriptor`
    """
    iteration_index: int = 1
    """If calling the message pump repeatedly, supply an iteration index to ensure
    that logged messages appear unique instead of apparing to be duplicated logs.
    
    Passed to the message pump as `--parent-iteration`
    """

    def as_command(self) -> t.List[str]:
        """Produce CLI arguments suitable for calling subprocess.Popen that
        to execute the msg pump.

        NOTE: does NOT include the `[sys.executable, msg_pump_path, ...]`
        portion of the necessary parameters to Popen.

        :returns: The arguments of the request formatted appropriately to
        Popen the `<project_dir>/tests/dragon/utils/msg_pump.py`"""
        return [
            "--dispatch-fli-descriptor",
            self.work_queue_descriptor,
            "--fs-descriptor",
            self.backbone_descriptor,
            "--parent-iteration",
            str(self.iteration_index),
            "--callback-descriptor",
            self.callback_descriptor,
        ]


@pytest.fixture(scope="session")
def msg_pump_factory() -> t.Callable[[MsgPumpRequest], subprocess.Popen]:
    """A pytest fixture used to create a mock event producer capable of
    feeding asynchronous inference requests to tests requiring them.

    :returns: A function that opens a subprocess running a mock message pump
    """

    def run_message_pump(request: MsgPumpRequest) -> subprocess.Popen:
        """Invoke the message pump entry-point with the descriptors
        from the request.

        :param request: A request containing all parameters required to
        invoke the message pump entrypoint
        :returns: The Popen object for the subprocess that was started"""
        # <smartsim_dir>/tests/dragon/utils/msg_pump.py
        msg_pump_script = "tests/dragon/utils/msg_pump.py"
        msg_pump_path = pathlib.Path(__file__).parent / msg_pump_script

        cmd = [sys.executable, str(msg_pump_path.absolute()), *request.as_command()]

        popen = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return popen

    return run_message_pump


@pytest.fixture(scope="module")
def the_storage() -> dragon_ddict.DDict:
    """Fixture to instantiate a dragon distributed dictionary."""
    return dragon_util.create_ddict(1, 2, 32 * 1024**2)


@pytest.fixture(scope="module")
def the_worker_channel() -> DragonFLIChannel:
    """Fixture to create a valid descriptor for a worker channel
    that can be attached to."""
    channel_ = create_local()
    fli_ = FLInterface(main_ch=channel_, manager_ch=None)
    comm_channel = DragonFLIChannel(fli_)
    return comm_channel


@pytest.fixture(scope="module")
def the_backbone(
    the_storage: t.Any, the_worker_channel: DragonFLIChannel
) -> BackboneFeatureStore:
    """Fixture to create a distributed dragon dictionary and wrap it
    in a BackboneFeatureStore.

    :param the_storage: The dragon storage engine to use
    :param the_worker_channel: Pre-configured worker channel
    """

    backbone = BackboneFeatureStore(the_storage, allow_reserved_writes=True)
    backbone[BackboneFeatureStore.MLI_WORKER_QUEUE] = the_worker_channel.descriptor

    return backbone


@pytest.fixture(scope="module")
def backbone_descriptor(the_backbone: BackboneFeatureStore) -> str:
    # create a shared backbone featurestore
    return the_backbone.descriptor
