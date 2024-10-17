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


import typing as t

import psutil
import pytest

from smartsim import Experiment
from smartsim.database import FeatureStore
from smartsim.error import SmartSimError
from smartsim.error.errors import SSUnsupportedError

# The tests in this file belong to the slow_tests group
pytestmark = pytest.mark.slow_tests


if t.TYPE_CHECKING:
    import conftest


def test_feature_store_parameters() -> None:
    threads_per_queue = 2
    inter_op_threads = 2
    intra_op_threads = 2
    fs = FeatureStore(
        fs_nodes=1,
        threads_per_queue=threads_per_queue,
        inter_op_threads=inter_op_threads,
        intra_op_threads=intra_op_threads,
    )
    assert fs.queue_threads == threads_per_queue
    assert fs.inter_threads == inter_op_threads
    assert fs.intra_threads == intra_op_threads

    module_str = fs._rai_module
    assert "THREADS_PER_QUEUE" in module_str
    assert "INTRA_OP_PARALLELISM" in module_str
    assert "INTER_OP_PARALLELISM" in module_str


def test_is_not_active() -> None:
    fs = FeatureStore(fs_nodes=1)
    assert not fs.is_active()


def test_inactive_feature_store_get_address() -> None:
    fs = FeatureStore()
    with pytest.raises(SmartSimError):
        fs.get_address()


def test_feature_store_is_active_functions(
    local_experiment,
    prepare_fs,
    local_fs,
) -> None:
    fs = prepare_fs(local_fs).featurestore
    fs = local_experiment.reconnect_feature_store(fs.checkpoint_file)
    assert fs.is_active()

    # check if the feature store can get the address
    assert fs.get_address() == [f"127.0.0.1:{fs.ports[0]}"]


def test_multiple_interfaces(
    test_dir: str, wlmutils: t.Type["conftest.WLMUtils"]
) -> None:
    exp_name = "test_multiple_interfaces"
    exp = Experiment(exp_name, launcher="local", exp_path=test_dir)

    net_if_addrs = psutil.net_if_addrs()
    net_if_addrs = [
        net_if_addr for net_if_addr in net_if_addrs if not net_if_addr.startswith("lo")
    ]

    net_if_addrs = ["lo", net_if_addrs[0]]

    port = wlmutils.get_test_port()
    fs = FeatureStore(port=port, interface=net_if_addrs)
    fs.set_path(test_dir)

    exp.start(fs)

    # check if the FeatureStore is active
    assert fs.is_active()

    # check if the feature store can get the address
    correct_address = [f"127.0.0.1:{port}"]

    if not correct_address == fs.get_address():
        exp.stop(fs)
        assert False

    exp.stop(fs)


def test_catch_local_feature_store_errors() -> None:
    # local feature store with more than one node not allowed
    with pytest.raises(SSUnsupportedError):
        fs = FeatureStore(fs_nodes=2)

    # Run command for local FeatureStore not allowed
    with pytest.raises(SmartSimError):
        fs = FeatureStore(run_command="srun")

    # Batch mode for local FeatureStore is not allowed
    with pytest.raises(SmartSimError):
        fs = FeatureStore(batch=True)


#####  PBS  ######


def test_pbs_set_run_arg(wlmutils: t.Type["conftest.WLMUtils"]) -> None:
    feature_store = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=False,
        interface="lo",
        launcher="pbs",
        run_command="aprun",
    )
    feature_store.set_run_arg("account", "ACCOUNT")
    assert all(
        [
            fs.run_settings.run_args["account"] == "ACCOUNT"
            for fs in feature_store.entities
        ]
    )
    feature_store.set_run_arg("pes-per-numa-node", "5")
    assert all(
        [
            "pes-per-numa-node" not in fs.run_settings.run_args
            for fs in feature_store.entities
        ]
    )


def test_pbs_set_batch_arg(wlmutils: t.Type["conftest.WLMUtils"]) -> None:
    feature_store = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=False,
        interface="lo",
        launcher="pbs",
        run_command="aprun",
    )
    with pytest.raises(SmartSimError):
        feature_store.set_batch_arg("account", "ACCOUNT")

    feature_store2 = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=True,
        interface="lo",
        launcher="pbs",
        run_command="aprun",
    )
    feature_store2.set_batch_arg("account", "ACCOUNT")
    assert feature_store2.batch_settings.batch_args["account"] == "ACCOUNT"
    feature_store2.set_batch_arg("N", "another_name")
    assert "N" not in feature_store2.batch_settings.batch_args


##### Slurm ######


def test_slurm_set_run_arg(wlmutils: t.Type["conftest.WLMUtils"]) -> None:
    feature_store = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=False,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    feature_store.set_run_arg("account", "ACCOUNT")
    assert all(
        [
            fs.run_settings.run_args["account"] == "ACCOUNT"
            for fs in feature_store.entities
        ]
    )


def test_slurm_set_batch_arg(wlmutils: t.Type["conftest.WLMUtils"]) -> None:
    feature_store = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=False,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    with pytest.raises(SmartSimError):
        feature_store.set_batch_arg("account", "ACCOUNT")

    feature_store2 = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=True,
        interface="lo",
        launcher="slurm",
        run_command="srun",
    )
    feature_store2.set_batch_arg("account", "ACCOUNT")
    assert feature_store2.batch_settings.batch_args["account"] == "ACCOUNT"


@pytest.mark.parametrize(
    "single_cmd",
    [
        pytest.param(True, id="Single MPMD `srun`"),
        pytest.param(False, id="Multiple `srun`s"),
    ],
)
def test_feature_store_results_in_correct_number_of_shards(single_cmd: bool) -> None:
    num_shards = 5
    feature_store = FeatureStore(
        port=12345,
        launcher="slurm",
        run_command="srun",
        fs_nodes=num_shards,
        batch=False,
        single_cmd=single_cmd,
    )
    if single_cmd:
        assert len(feature_store.entities) == 1
        (node,) = feature_store.entities
        assert len(node.run_settings.mpmd) == num_shards - 1
    else:
        assert len(feature_store.entities) == num_shards
        assert all(node.run_settings.mpmd == [] for node in feature_store.entities)
    assert (
        feature_store.num_shards
        == feature_store.fs_nodes
        == sum(node.num_shards for node in feature_store.entities)
    )


###### LSF ######


def test_catch_feature_store_errors_lsf(wlmutils: t.Type["conftest.WLMUtils"]) -> None:
    with pytest.raises(SSUnsupportedError):
        feature_store = FeatureStore(
            wlmutils.get_test_port(),
            fs_nodes=2,
            fs_per_host=2,
            batch=False,
            launcher="lsf",
            run_command="jsrun",
        )

    feature_store = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=False,
        hosts=["batch", "host1", "host2"],
        launcher="lsf",
        run_command="jsrun",
    )
    with pytest.raises(SmartSimError):
        feature_store.set_batch_arg("P", "MYPROJECT")


def test_lsf_set_run_args(wlmutils: t.Type["conftest.WLMUtils"]) -> None:
    feature_store = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
        launcher="lsf",
        run_command="jsrun",
    )
    feature_store.set_run_arg("l", "gpu-gpu")
    assert all(["l" not in fs.run_settings.run_args for fs in feature_store.entities])


def test_lsf_set_batch_args(wlmutils: t.Type["conftest.WLMUtils"]) -> None:
    feature_store = FeatureStore(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=True,
        hosts=["batch", "host1", "host2"],
        launcher="lsf",
        run_command="jsrun",
    )

    assert feature_store.batch_settings.batch_args["m"] == '"batch host1 host2"'
    feature_store.set_batch_arg("D", "102400000")
    assert feature_store.batch_settings.batch_args["D"] == "102400000"


def test_orc_telemetry(test_dir: str, wlmutils: t.Type["conftest.WLMUtils"]) -> None:
    """Ensure the default behavior for a feature store is to disable telemetry"""
    fs = FeatureStore(port=wlmutils.get_test_port())
    fs.set_path(test_dir)

    # default is disabled
    assert not fs.telemetry.is_enabled

    # ensure updating value works as expected
    fs.telemetry.enable()
    assert fs.telemetry.is_enabled

    # toggle back
    fs.telemetry.disable()
    assert not fs.telemetry.is_enabled

    # toggle one more time
    fs.telemetry.enable()
    assert fs.telemetry.is_enabled
