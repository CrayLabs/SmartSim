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
from smartsim.settings.batchCommand import SchedulerType
from smartsim.settings.builders.batch.slurm import SlurmBatchArgBuilder

pytestmark = pytest.mark.group_a


def test_scheduler_str():
    """Ensure scheduler_str returns appropriate value"""
    bs = BatchSettings(batch_scheduler=SchedulerType.Slurm)
    assert bs.scheduler_args.scheduler_str() == SchedulerType.Slurm.value


@pytest.mark.parametrize(
    "function,value,result,flag",
    [
        pytest.param("set_nodes", (2,), "2", "nodes", id="set_nodes"),
        pytest.param(
            "set_walltime", ("10:00:00",), "10:00:00", "time", id="set_walltime"
        ),
        pytest.param(
            "set_account", ("account",), "account", "account", id="set_account"
        ),
        pytest.param(
            "set_partition",
            ("partition",),
            "partition",
            "partition",
            id="set_partition",
        ),
        pytest.param(
            "set_queue", ("partition",), "partition", "partition", id="set_queue"
        ),
        pytest.param(
            "set_cpus_per_task", (2,), "2", "cpus-per-task", id="set_cpus_per_task"
        ),
        pytest.param(
            "set_hostlist", ("host_A",), "host_A", "nodelist", id="set_hostlist_str"
        ),
        pytest.param(
            "set_hostlist",
            (["host_A", "host_B"],),
            "host_A,host_B",
            "nodelist",
            id="set_hostlist_list[str]",
        ),
    ],
)
def test_sbatch_class_methods(function, value, flag, result):
    slurmScheduler = BatchSettings(batch_scheduler=SchedulerType.Slurm)
    getattr(slurmScheduler.scheduler_args, function)(*value)
    assert slurmScheduler.scheduler_args._scheduler_args[flag] == result


def test_create_sbatch():
    batch_args = {"exclusive": None, "oversubscribe": None}
    slurmScheduler = BatchSettings(
        batch_scheduler=SchedulerType.Slurm, scheduler_args=batch_args
    )
    assert isinstance(slurmScheduler._arg_builder, SlurmBatchArgBuilder)
    args = slurmScheduler.format_batch_args()
    assert args == ["--exclusive", "--oversubscribe"]


def test_launch_args_input_mutation():
    # Tests that the run args passed in are not modified after initialization
    key0, key1, key2 = "arg0", "arg1", "arg2"
    val0, val1, val2 = "val0", "val1", "val2"

    default_scheduler_args = {
        key0: val0,
        key1: val1,
        key2: val2,
    }
    slurmScheduler = BatchSettings(
        batch_scheduler=SchedulerType.Slurm, scheduler_args=default_scheduler_args
    )

    # Confirm initial values are set
    assert slurmScheduler.scheduler_args._scheduler_args[key0] == val0
    assert slurmScheduler.scheduler_args._scheduler_args[key1] == val1
    assert slurmScheduler.scheduler_args._scheduler_args[key2] == val2

    # Update our common run arguments
    val2_upd = f"not-{val2}"
    default_scheduler_args[key2] = val2_upd

    # Confirm previously created run settings are not changed
    assert slurmScheduler.scheduler_args._scheduler_args[key2] == val2


def test_sbatch_settings():
    scheduler_args = {"nodes": 1, "time": "10:00:00", "account": "A3123"}
    slurmScheduler = BatchSettings(
        batch_scheduler=SchedulerType.Slurm, scheduler_args=scheduler_args
    )
    formatted = slurmScheduler.format_batch_args()
    result = ["--nodes=1", "--time=10:00:00", "--account=A3123"]
    assert formatted == result


def test_sbatch_manual():
    slurmScheduler = BatchSettings(batch_scheduler=SchedulerType.Slurm)
    slurmScheduler.scheduler_args.set_nodes(5)
    slurmScheduler.scheduler_args.set_account("A3531")
    slurmScheduler.scheduler_args.set_walltime("10:00:00")
    formatted = slurmScheduler.format_batch_args()
    print(f"here: {formatted}")
    result = ["--nodes=5", "--account=A3531", "--time=10:00:00"]
    assert formatted == result
