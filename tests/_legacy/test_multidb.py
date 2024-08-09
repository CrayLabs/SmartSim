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
from contextlib import contextmanager

import pytest

from smartsim import Experiment
from smartsim.database import FeatureStore
from smartsim.entity.entity import SmartSimEntity
from smartsim.error.errors import SSDBIDConflictError
from smartsim.log import get_logger
from smartsim.status import JobStatus

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


logger = get_logger(__name__)

supported_fss = ["uds", "tcp"]

on_wlm = (pytest.test_launcher in pytest.wlm_options,)


@contextmanager
def make_entity_context(exp: Experiment, entity: SmartSimEntity):
    """Start entity in a context to ensure that it is always stopped"""
    exp.generate(entity, overwrite=True)
    try:
        yield entity
    finally:
        if exp.get_status(entity)[0] == JobStatus.RUNNING:
            exp.stop(entity)


def choose_host(wlmutils, index=0):
    hosts = wlmutils.get_test_hostlist()
    if hosts:
        return hosts[index]
    else:
        return None


def check_not_failed(exp, *args):
    statuses = exp.get_status(*args)
    assert all(stat is not JobStatus.FAILED for stat in statuses)


@pytest.mark.parametrize("fs_type", supported_fss)
def test_fs_identifier_standard_then_colo_error(
    fileutils, wlmutils, coloutils, fs_type, test_dir
):
    """Test that it is possible to create_feature_store then colocate_fs_uds/colocate_fs_tcp
    with unique fs_identifiers"""

    # Set experiment name
    exp_name = "test_fs_identifier_standard_then_colo"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    test_script = fileutils.get_test_conf_path("smartredis/fs_id_err.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # create regular feature store
    feature_store = exp.create_feature_store(
        port=test_port,
        interface=test_interface,
        fs_identifier="testdb_colo",
        hosts=choose_host(wlmutils),
    )
    assert feature_store.name == "testdb_colo"

    fs_args = {
        "port": test_port + 1,
        "fs_cpus": 1,
        "debug": True,
        "fs_identifier": "testdb_colo",
    }

    smartsim_model = coloutils.setup_test_colo(
        fileutils, fs_type, exp, test_script, fs_args, on_wlm=on_wlm
    )

    assert (
        smartsim_model.run_settings.colocated_fs_settings["fs_identifier"]
        == "testdb_colo"
    )

    with (
        make_entity_context(exp, feature_store),
        make_entity_context(exp, smartsim_model),
    ):
        exp.start(feature_store)
        with pytest.raises(SSDBIDConflictError) as ex:
            exp.start(smartsim_model)

        assert (
            "has already been used. Pass in a unique name for fs_identifier"
            in ex.value.args[0]
        )
        check_not_failed(exp, feature_store)


@pytest.mark.parametrize("fs_type", supported_fss)
def test_fs_identifier_colo_then_standard(
    fileutils, wlmutils, coloutils, fs_type, test_dir
):
    """Test colocate_fs_uds/colocate_fs_tcp then create_feature_store with feature store
    identifiers.
    """

    # Set experiment name
    exp_name = "test_fs_identifier_colo_then_standard"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_script = fileutils.get_test_conf_path("smartredis/dbid.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # Create run settings
    colo_settings = exp.create_run_settings("python", test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    # Create the SmartSim Model
    smartsim_model = exp.create_application("colocated_model", colo_settings)

    fs_args = {
        "port": test_port,
        "fs_cpus": 1,
        "debug": True,
        "fs_identifier": "testdb_colo",
    }

    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        fs_type,
        exp,
        test_script,
        fs_args,
        on_wlm=on_wlm,
    )

    assert (
        smartsim_model.run_settings.colocated_fs_settings["fs_identifier"]
        == "testdb_colo"
    )

    # Create feature store
    feature_store = exp.create_feature_store(
        port=test_port + 1,
        interface=test_interface,
        fs_identifier="testdb_colo",
        hosts=choose_host(wlmutils),
    )

    assert feature_store.name == "testdb_colo"

    with (
        make_entity_context(exp, feature_store),
        make_entity_context(exp, smartsim_model),
    ):
        exp.start(smartsim_model, block=True)
        exp.start(feature_store)

    check_not_failed(exp, feature_store, smartsim_model)


def test_fs_identifier_standard_twice_not_unique(wlmutils, test_dir):
    """Test uniqueness of fs_identifier several calls to create_feature_store, with non unique names,
    checking error is raised before exp start is called"""

    # Set experiment name
    exp_name = "test_fs_identifier_multiple_create_feature_store_not_unique"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # CREATE feature store with fs_identifier
    feature_store = exp.create_feature_store(
        port=test_port,
        interface=test_interface,
        fs_identifier="my_fs",
        hosts=choose_host(wlmutils),
    )

    assert feature_store.name == "my_fs"

    feature_store2 = exp.create_feature_store(
        port=test_port + 1,
        interface=test_interface,
        fs_identifier="my_fs",
        hosts=choose_host(wlmutils, index=1),
    )

    assert feature_store2.name == "my_fs"

    # CREATE feature store with fs_identifier
    with (
        make_entity_context(exp, feature_store2),
        make_entity_context(exp, feature_store),
    ):
        exp.start(feature_store)
        with pytest.raises(SSDBIDConflictError) as ex:
            exp.start(feature_store)
        assert (
            "has already been used. Pass in a unique name for fs_identifier"
            in ex.value.args[0]
        )
        check_not_failed(exp, feature_store)


def test_fs_identifier_create_standard_once(test_dir, wlmutils):
    """One call to create feature store with a feature storeidentifier"""

    # Set experiment name
    exp_name = "test_fs_identifier_create_standard_once"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create the SmartSim feature store
    fs = exp.create_feature_store(
        port=test_port,
        fs_nodes=1,
        interface=test_interface,
        fs_identifier="testdb_reg",
        hosts=choose_host(wlmutils),
    )
    with make_entity_context(exp, fs):
        exp.start(fs)

    check_not_failed(exp, fs)


def test_multifs_create_standard_twice(wlmutils, test_dir):
    """Multiple calls to create feature store with unique fs_identifiers"""

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    # start a new Experiment for this section
    exp = Experiment(
        "test_multifs_create_standard_twice", exp_path=test_dir, launcher=test_launcher
    )

    # create and start an instance of the FeatureStore feature store
    fs = exp.create_feature_store(
        port=test_port,
        interface=test_interface,
        fs_identifier="testdb_reg",
        hosts=choose_host(wlmutils, 1),
    )

    # create feature store with different fs_id
    fs2 = exp.create_feature_store(
        port=test_port + 1,
        interface=test_interface,
        fs_identifier="testdb_reg2",
        hosts=choose_host(wlmutils, 2),
    )

    # launch
    with make_entity_context(exp, fs), make_entity_context(exp, fs2):
        exp.start(fs, fs2)

    with make_entity_context(exp, fs), make_entity_context(exp, fs2):
        exp.start(fs, fs2)


@pytest.mark.parametrize("fs_type", supported_fss)
def test_multifs_colo_once(fileutils, test_dir, wlmutils, coloutils, fs_type):
    """create one model with colocated feature store with fs_identifier"""

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_port = wlmutils.get_test_port()

    test_script = fileutils.get_test_conf_path("smartredis/dbid.py")

    # start a new Experiment for this section
    exp = Experiment(
        "test_multifs_colo_once", launcher=test_launcher, exp_path=test_dir
    )

    # create run settings
    run_settings = exp.create_run_settings("python", test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)

    # Create the SmartSim Model
    smartsim_model = exp.create_application("smartsim_model", run_settings)

    fs_args = {
        "port": test_port + 1,
        "fs_cpus": 1,
        "debug": True,
        "fs_identifier": "testdb_colo",
    }
    # Create model with colocated feature store

    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        fs_type,
        exp,
        test_script,
        fs_args,
        on_wlm=on_wlm,
    )

    with make_entity_context(exp, smartsim_model):
        exp.start(smartsim_model)

    check_not_failed(exp, smartsim_model)


@pytest.mark.parametrize("fs_type", supported_fss)
def test_multifs_standard_then_colo(fileutils, test_dir, wlmutils, coloutils, fs_type):
    """Create regular feature store then colocate_fs_tcp/uds with unique fs_identifiers"""

    # Retrieve parameters from testing environment
    test_port = wlmutils.get_test_port()

    test_script = fileutils.get_test_conf_path("smartredis/multidbid.py")
    test_interface = wlmutils.get_test_interface()
    test_launcher = wlmutils.get_test_launcher()

    # start a new Experiment for this section
    exp = Experiment(
        "test_multifs_standard_then_colo", exp_path=test_dir, launcher=test_launcher
    )

    # create and generate an instance of the FeatureStore feature store
    fs = exp.create_feature_store(
        port=test_port,
        interface=test_interface,
        fs_identifier="testdb_reg",
        hosts=choose_host(wlmutils),
    )

    fs_args = {
        "port": test_port + 1,
        "fs_cpus": 1,
        "debug": True,
        "fs_identifier": "testdb_colo",
    }
    # Create model with colocated feature store
    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        fs_type,
        exp,
        test_script,
        fs_args,
        on_wlm=on_wlm,
    )

    with make_entity_context(exp, fs), make_entity_context(exp, smartsim_model):
        exp.start(fs)
        exp.start(smartsim_model, block=True)

    check_not_failed(exp, smartsim_model, fs)


@pytest.mark.parametrize("fs_type", supported_fss)
def test_multifs_colo_then_standard(fileutils, test_dir, wlmutils, coloutils, fs_type):
    """create regular feature store then colocate_fs_tcp/uds with unique fs_identifiers"""

    # Retrieve parameters from testing environment
    test_port = wlmutils.get_test_port()

    test_script = fileutils.get_test_conf_path(
        "smartredis/multidbid_colo_env_vars_only.py"
    )
    test_interface = wlmutils.get_test_interface()
    test_launcher = wlmutils.get_test_launcher()

    # start a new Experiment
    exp = Experiment(
        "test_multifs_colo_then_standard", exp_path=test_dir, launcher=test_launcher
    )

    fs_args = {
        "port": test_port,
        "fs_cpus": 1,
        "debug": True,
        "fs_identifier": "testdb_colo",
    }

    # Create model with colocated feature store
    smartsim_model = coloutils.setup_test_colo(
        fileutils, fs_type, exp, test_script, fs_args, on_wlm=on_wlm
    )

    # create and start an instance of the FeatureStore feature store
    fs = exp.create_feature_store(
        port=test_port + 1,
        interface=test_interface,
        fs_identifier="testdb_reg",
        hosts=choose_host(wlmutils),
    )

    with make_entity_context(exp, fs), make_entity_context(exp, smartsim_model):
        exp.start(smartsim_model, block=False)
        exp.start(fs)
        exp.poll(smartsim_model)

    check_not_failed(exp, fs, smartsim_model)


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
@pytest.mark.parametrize("fs_type", supported_fss)
def test_launch_cluster_feature_store_single_fsid(
    test_dir, coloutils, fileutils, wlmutils, fs_type
):
    """test clustered 3-node FeatureStore with single command with a feature store identifier"""
    # TODO detect number of nodes in allocation and skip if not sufficent

    exp_name = "test_launch_cluster_feature_store_single_fsid"
    launcher = wlmutils.get_test_launcher()
    test_port = wlmutils.get_test_port()
    test_script = fileutils.get_test_conf_path("smartredis/multidbid.py")
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    feature_store: FeatureStore = exp.create_feature_store(
        wlmutils.get_test_port(),
        fs_nodes=3,
        batch=False,
        interface=network_interface,
        single_cmd=True,
        hosts=wlmutils.get_test_hostlist(),
        fs_identifier="testdb_reg",
    )

    fs_args = {
        "port": test_port,
        "fs_cpus": 1,
        "debug": True,
        "fs_identifier": "testdb_colo",
    }

    # Create model with colocated feature store
    smartsim_model = coloutils.setup_test_colo(
        fileutils, fs_type, exp, test_script, fs_args, on_wlm=on_wlm
    )

    with (
        make_entity_context(exp, feature_store),
        make_entity_context(exp, smartsim_model),
    ):
        exp.start(feature_store, block=True)
        exp.start(smartsim_model, block=True)
        job_dict = exp._control._jobs.get_fs_host_addresses()
        assert len(job_dict[feature_store.entities[0].fs_identifier]) == 3

    check_not_failed(exp, feature_store, smartsim_model)
