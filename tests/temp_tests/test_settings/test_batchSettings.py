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
import pytest

from smartsim.settings import BatchSettings
from smartsim.settings.batch_command import BatchSchedulerType

pytestmark = pytest.mark.group_a


@pytest.mark.parametrize(
    "scheduler_enum,formatted_batch_args",
    [
        pytest.param(
            BatchSchedulerType.Slurm, ["--launch=var", "--nodes=1"], id="slurm"
        ),
        pytest.param(
            BatchSchedulerType.Pbs, ["-l", "nodes=1", "-launch", "var"], id="pbs"
        ),
        pytest.param(
            BatchSchedulerType.Lsf, ["-launch", "var", "-nnodes", "1"], id="lsf"
        ),
    ],
)
def test_create_scheduler_settings(scheduler_enum, formatted_batch_args):
    bs_str = BatchSettings(
        batch_scheduler=scheduler_enum.value,
        batch_args={"launch": "var"},
        env_vars={"ENV": "VAR"},
    )
    bs_str.batch_args.set_nodes(1)
    assert bs_str._batch_scheduler == scheduler_enum
    assert bs_str._env_vars == {"ENV": "VAR"}
    print(bs_str.format_batch_args())
    assert bs_str.format_batch_args() == formatted_batch_args

    bs_enum = BatchSettings(
        batch_scheduler=scheduler_enum,
        batch_args={"launch": "var"},
        env_vars={"ENV": "VAR"},
    )
    bs_enum.batch_args.set_nodes(1)
    assert bs_enum._batch_scheduler == scheduler_enum
    assert bs_enum._env_vars == {"ENV": "VAR"}
    assert bs_enum.format_batch_args() == formatted_batch_args


def test_launcher_property():
    bs = BatchSettings(batch_scheduler="slurm")
    assert bs.batch_scheduler == "slurm"


def test_env_vars_property():
    bs = BatchSettings(batch_scheduler="slurm", env_vars={"ENV": "VAR"})
    assert bs.env_vars == {"ENV": "VAR"}
    ref = bs.env_vars
    assert ref is bs.env_vars
