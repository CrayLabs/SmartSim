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


def test_scheduler_str():
    """Ensure scheduler_str returns appropriate value"""
    bs = BatchSettings(batch_scheduler=BatchSchedulerType.Lsf)
    assert bs.batch_args.scheduler_str() == BatchSchedulerType.Lsf.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,), "2", "nnodes", id="set_nodes"),
        pytest.param("set_walltime", ("10:00:00",), "10:00", "W", id="set_walltime"),
        pytest.param(
            "set_hostlist", ("host_A",), "" '"host_A"' "", "m", id="set_hostlist_str"
        ),
        pytest.param(
            "set_hostlist",
            (["host_A", "host_B"],),
            "" '"host_A host_B"' "",
            "m",
            id="set_hostlist_list[str]",
        ),
        pytest.param("set_smts", (1,), "1", "alloc_flags", id="set_smts"),
        pytest.param("set_project", ("project",), "project", "P", id="set_project"),
        pytest.param("set_account", ("project",), "project", "P", id="set_account"),
        pytest.param("set_tasks", (2,), "2", "n", id="set_tasks"),
        pytest.param("set_queue", ("queue",), "queue", "q", id="set_queue"),
    ],
)
def test_update_env_initialized(function, value, flag, result):
    lsfScheduler = BatchSettings(batch_scheduler=BatchSchedulerType.Lsf)
    getattr(lsfScheduler.batch_args, function)(*value)
    assert lsfScheduler.batch_args._batch_args[flag] == result


def test_create_bsub():
    batch_args = {"core_isolation": None}
    lsfScheduler = BatchSettings(
        batch_scheduler=BatchSchedulerType.Lsf, batch_args=batch_args
    )
    lsfScheduler.batch_args.set_nodes(1)
    lsfScheduler.batch_args.set_walltime("10:10:10")
    lsfScheduler.batch_args.set_queue("default")
    args = lsfScheduler.format_batch_args()
    assert args == ["-core_isolation", "-nnodes", "1", "-W", "10:10", "-q", "default"]
