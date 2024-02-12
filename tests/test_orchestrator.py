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


import psutil
import pytest

from smartsim import Experiment
from smartsim.database import Orchestrator
from smartsim.error import SmartSimError
from smartsim.error.errors import SSUnsupportedError

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


def test_orc_parameters():
    threads_per_queue = 2
    inter_op_threads = 2
    intra_op_threads = 2
    db = Orchestrator(
        db_nodes=1,
        threads_per_queue=threads_per_queue,
        inter_op_threads=inter_op_threads,
        intra_op_threads=intra_op_threads,
    )
    assert db.queue_threads == threads_per_queue
    assert db.inter_threads == inter_op_threads
    assert db.intra_threads == intra_op_threads

    module_str = db._rai_module
    assert "THREADS_PER_QUEUE" in module_str
    assert "INTRA_OP_PARALLELISM" in module_str
    assert "INTER_OP_PARALLELISM" in module_str


def test_is_not_active():
    db = Orchestrator(db_nodes=1)
    assert not db.is_active()


def test_inactive_orc_get_address():
    db = Orchestrator()
    with pytest.raises(SmartSimError):
        db.get_address()


def test_orc_active_functions(test_dir, wlmutils):
    exp_name = "test_orc_active_functions"
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)

    db = Orchestrator(port=wlmutils.get_test_port())
    db.set_path(test_dir)

    exp.start(db)

    # check if the orchestrator is active
    assert db.is_active()

    # check if the orchestrator can get the address
    correct_address = db.get_address() == ["127.0.0.1:" + str(wlmutils.get_test_port())]
    if not correct_address:
        exp.stop(db)
        assert False

    exp.stop(db)

    assert not db.is_active()

    # check if orchestrator.get_address() raises an exception
    with pytest.raises(SmartSimError):
        db.get_address()


def test_multiple_interfaces(test_dir, wlmutils):
    exp_name = "test_multiple_interfaces"
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)

    net_if_addrs = psutil.net_if_addrs()
    net_if_addrs = [
        net_if_addr for net_if_addr in net_if_addrs if not net_if_addr.startswith("lo")
    ]

    net_if_addrs = ["lo", net_if_addrs[0]]

    db = Orchestrator(port=wlmutils.get_test_port(), interface=net_if_addrs)
    db.set_path(test_dir)

    exp.start(db)

    # check if the orchestrator is active
    assert db.is_active()

    # check if the orchestrator can get the address
    correct_address = db.get_address() == ["127.0.0.1:" + str(wlmutils.get_test_port())]
    if not correct_address:
        exp.stop(db)
        assert False

    exp.stop(db)


def test_catch_local_db_errors():
    # local database with more than one node not allowed
    with pytest.raises(SSUnsupportedError):
        db = Orchestrator(db_nodes=2)

    # Run command for local orchestrator not allowed
    with pytest.raises(SmartSimError):
        db = Orchestrator(run_command="srun")

    # Batch mode for local orchestrator is not allowed
    with pytest.raises(SmartSimError):
        db = Orchestrator(batch=True)


#####  PBS  ######


def test_pbs_set_run_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="pbs",
        run_command="aprun",
    )
    orc.set_run_arg("account", "ACCOUNT")
    assert all(
        [db.run_settings.run_args["account"] == "ACCOUNT" for db in orc.entities]
    )
    orc.set_run_arg("pes-per-numa-node", "5")
    assert all(
        ["pes-per-numa-node" not in db.run_settings.run_args for db in orc.entities]
    )


def test_pbs_set_batch_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="pbs",
        run_command="aprun",
    )
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("account", "ACCOUNT")

    orc2 = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        interface="lo",
        launcher="pbs",
        run_command="aprun",
    )
    orc2.set_batch_arg("account", "ACCOUNT")
    assert orc2.batch_settings.batch_args["account"] == "ACCOUNT"
    orc2.set_batch_arg("N", "another_name")
    assert "N" not in orc2.batch_settings.batch_args


##### Slurm ######


def test_slurm_set_run_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    orc.set_run_arg("account", "ACCOUNT")
    assert all(
        [db.run_settings.run_args["account"] == "ACCOUNT" for db in orc.entities]
    )


def test_slurm_set_batch_arg(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("account", "ACCOUNT")

    orc2 = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    orc2.set_batch_arg("account", "ACCOUNT")
    assert orc2.batch_settings.batch_args["account"] == "ACCOUNT"


@pytest.mark.parametrize(
    "single_cmd",
    [
        pytest.param(True, id="Single MPMD `srun`"),
        pytest.param(False, id="Multiple `srun`s"),
    ],
)
def test_orc_results_in_correct_number_of_shards(single_cmd):
    num_shards = 5
    orc = Orchestrator(
        port=12345,
        launcher="slurm",
        run_command="srun",
        db_nodes=num_shards,
        batch=False,
        single_cmd=single_cmd,
    )
    if single_cmd:
        assert len(orc.entities) == 1
        (node,) = orc.entities
        assert len(node.run_settings.mpmd) == num_shards - 1
    else:
        assert len(orc.entities) == num_shards
        assert all(node.run_settings.mpmd == [] for node in orc.entities)
    assert (
        orc.num_shards == orc.db_nodes == sum(node.num_shards for node in orc.entities)
    )


###### LSF ######


def test_catch_orc_errors_lsf(wlmutils):
    with pytest.raises(SSUnsupportedError):
        orc = Orchestrator(
            wlmutils.get_test_port(),
            db_nodes=2,
            db_per_host=2,
            batch=False,
            launcher="lsf",
            run_command="jsrun",
        )

    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        hosts=["batch", "host1", "host2"],
        launcher="lsf",
        run_command="jsrun",
    )
    with pytest.raises(SmartSimError):
        orc.set_batch_arg("P", "MYPROJECT")


def test_lsf_set_run_args(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
        launcher="lsf",
        run_command="jsrun",
    )
    orc.set_run_arg("l", "gpu-gpu")
    assert all(["l" not in db.run_settings.run_args for db in orc.entities])


def test_lsf_set_batch_args(wlmutils):
    orc = Orchestrator(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
        launcher="lsf",
        run_command="jsrun",
    )

    assert orc.batch_settings.batch_args["m"] == '"batch host1 host2"'
    orc.set_batch_arg("D", "102400000")
    assert orc.batch_settings.batch_args["D"] == "102400000"
