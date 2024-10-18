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
from smartsim.settings.arguments.batch.pbs import QsubBatchArguments
from smartsim.settings.batch_command import BatchSchedulerType

pytestmark = pytest.mark.group_a


def test_scheduler_str():
    """Ensure scheduler_str returns appropriate value"""
    bs = BatchSettings(batch_scheduler=BatchSchedulerType.Pbs)
    assert bs.batch_args.scheduler_str() == BatchSchedulerType.Pbs.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,), "2", "nodes", id="set_nodes"),
        pytest.param(
            "set_walltime", ("10:00:00",), "10:00:00", "walltime", id="set_walltime"
        ),
        pytest.param("set_account", ("account",), "account", "A", id="set_account"),
        pytest.param("set_queue", ("queue",), "queue", "q", id="set_queue"),
        pytest.param("set_ncpus", (2,), "2", "ppn", id="set_ncpus"),
        pytest.param(
            "set_hostlist", ("host_A",), "host_A", "hostname", id="set_hostlist_str"
        ),
        pytest.param(
            "set_hostlist",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "hostname",
            id="set_hostlist_list[str]",
        ),
    ],
)
def test_create_pbs_batch(function, value, flag, result):
    pbsScheduler = BatchSettings(batch_scheduler=BatchSchedulerType.Pbs)
    assert isinstance(pbsScheduler.batch_args, QsubBatchArguments)
    getattr(pbsScheduler.batch_args, function)(*value)
    assert pbsScheduler.batch_args._batch_args[flag] == result


def test_format_pbs_batch_args():
    pbsScheduler = BatchSettings(batch_scheduler=BatchSchedulerType.Pbs)
    pbsScheduler.batch_args.set_nodes(1)
    pbsScheduler.batch_args.set_walltime("10:00:00")
    pbsScheduler.batch_args.set_queue("default")
    pbsScheduler.batch_args.set_account("myproject")
    pbsScheduler.batch_args.set_ncpus(10)
    pbsScheduler.batch_args.set_hostlist(["host_a", "host_b", "host_c"])
    args = pbsScheduler.format_batch_args()
    assert args == [
        "-l",
        "nodes=1:ncpus=10:host=host_a+host=host_b+host=host_c",
        "-l",
        "walltime=10:00:00",
        "-q",
        "default",
        "-A",
        "myproject",
    ]
