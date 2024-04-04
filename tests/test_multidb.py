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
from smartsim.database import Orchestrator
from smartsim.entity.entity import SmartSimEntity
from smartsim.error.errors import SSDBIDConflictError
from smartsim.log import get_logger
from smartsim.status import SmartSimStatus

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


logger = get_logger(__name__)

supported_dbs = ["uds", "tcp"]

on_wlm = (pytest.test_launcher in pytest.wlm_options,)


@contextmanager
def make_entity_context(exp: Experiment, entity: SmartSimEntity):
    """Start entity in a context to ensure that it is always stopped"""
    exp.generate(entity, overwrite=True)
    try:
        yield entity
    finally:
        if exp.get_status(entity)[0] == SmartSimStatus.STATUS_RUNNING:
            exp.stop(entity)


def choose_host(wlmutils, index=0):
    hosts = wlmutils.get_test_hostlist()
    if hosts:
        return hosts[index]
    else:
        return None


def check_not_failed(exp, *args):
    statuses = exp.get_status(*args)
    assert all(stat is not SmartSimStatus.STATUS_FAILED for stat in statuses)


@pytest.mark.parametrize("db_type", supported_dbs)
def test_db_identifier_standard_then_colo_error(
    fileutils, wlmutils, coloutils, db_type, test_dir
):
    """Test that it is possible to create_database then colocate_db_uds/colocate_db_tcp
    with unique db_identifiers"""

    # Set experiment name
    exp_name = "test_db_identifier_standard_then_colo"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    test_script = fileutils.get_test_conf_path("smartredis/db_id_err.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # create regular database
    orc = exp.create_database(
        port=test_port,
        interface=test_interface,
        db_identifier="testdb_colo",
        hosts=choose_host(wlmutils),
    )
    assert orc.name == "testdb_colo"

    db_args = {
        "port": test_port + 1,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "testdb_colo",
    }

    smartsim_model = coloutils.setup_test_colo(
        fileutils, db_type, exp, test_script, db_args, on_wlm=on_wlm
    )

    assert (
        smartsim_model.run_settings.colocated_db_settings["db_identifier"]
        == "testdb_colo"
    )

    with make_entity_context(exp, orc), make_entity_context(exp, smartsim_model):
        exp.start(orc)
        with pytest.raises(SSDBIDConflictError) as ex:
            exp.start(smartsim_model)

        assert (
            "has already been used. Pass in a unique name for db_identifier"
            in ex.value.args[0]
        )
        check_not_failed(exp, orc)


@pytest.mark.parametrize("db_type", supported_dbs)
def test_db_identifier_colo_then_standard(
    fileutils, wlmutils, coloutils, db_type, test_dir
):
    """Test colocate_db_uds/colocate_db_tcp then create_database with database
    identifiers.
    """

    # Set experiment name
    exp_name = "test_db_identifier_colo_then_standard"

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
    smartsim_model = exp.create_model("colocated_model", colo_settings)

    db_args = {
        "port": test_port,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "testdb_colo",
    }

    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        test_script,
        db_args,
        on_wlm=on_wlm,
    )

    assert (
        smartsim_model.run_settings.colocated_db_settings["db_identifier"]
        == "testdb_colo"
    )

    # Create Database
    orc = exp.create_database(
        port=test_port + 1,
        interface=test_interface,
        db_identifier="testdb_colo",
        hosts=choose_host(wlmutils),
    )

    assert orc.name == "testdb_colo"

    with make_entity_context(exp, orc), make_entity_context(exp, smartsim_model):
        exp.start(smartsim_model, block=True)
        exp.start(orc)

    check_not_failed(exp, orc, smartsim_model)


def test_db_identifier_standard_twice_not_unique(wlmutils, test_dir):
    """Test uniqueness of db_identifier several calls to create_database, with non unique names,
    checking error is raised before exp start is called"""

    # Set experiment name
    exp_name = "test_db_identifier_multiple_create_database_not_unique"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # CREATE DATABASE with db_identifier
    orc = exp.create_database(
        port=test_port,
        interface=test_interface,
        db_identifier="my_db",
        hosts=choose_host(wlmutils),
    )

    assert orc.name == "my_db"

    orc2 = exp.create_database(
        port=test_port + 1,
        interface=test_interface,
        db_identifier="my_db",
        hosts=choose_host(wlmutils, index=1),
    )

    assert orc2.name == "my_db"

    # CREATE DATABASE with db_identifier
    with make_entity_context(exp, orc), make_entity_context(exp, orc2):
        exp.start(orc)
        with pytest.raises(SSDBIDConflictError) as ex:
            exp.start(orc2)
        assert (
            "has already been used. Pass in a unique name for db_identifier"
            in ex.value.args[0]
        )
        check_not_failed(exp, orc)


def test_db_identifier_create_standard_once(test_dir, wlmutils):
    """One call to create database with a database identifier"""

    # Set experiment name
    exp_name = "test_db_identifier_create_standard_once"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create the SmartSim database
    db = exp.create_database(
        port=test_port,
        db_nodes=1,
        interface=test_interface,
        db_identifier="testdb_reg",
        hosts=choose_host(wlmutils),
    )
    with make_entity_context(exp, db):
        exp.start(db)

    check_not_failed(exp, db)


def test_multidb_create_standard_twice(wlmutils, test_dir):
    """Multiple calls to create database with unique db_identifiers"""

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    # start a new Experiment for this section
    exp = Experiment(
        "test_multidb_create_standard_twice", exp_path=test_dir, launcher=test_launcher
    )

    # create and start an instance of the Orchestrator database
    db = exp.create_database(
        port=test_port,
        interface=test_interface,
        db_identifier="testdb_reg",
        hosts=choose_host(wlmutils, 1),
    )

    # create database with different db_id
    db2 = exp.create_database(
        port=test_port + 1,
        interface=test_interface,
        db_identifier="testdb_reg2",
        hosts=choose_host(wlmutils, 2),
    )

    # launch
    with make_entity_context(exp, db), make_entity_context(exp, db2):
        exp.start(db, db2)

    with make_entity_context(exp, db), make_entity_context(exp, db2):
        exp.start(db, db2)


@pytest.mark.parametrize("db_type", supported_dbs)
def test_multidb_colo_once(fileutils, test_dir, wlmutils, coloutils, db_type):
    """create one model with colocated database with db_identifier"""

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_port = wlmutils.get_test_port()

    test_script = fileutils.get_test_conf_path("smartredis/dbid.py")

    # start a new Experiment for this section
    exp = Experiment(
        "test_multidb_colo_once", launcher=test_launcher, exp_path=test_dir
    )

    # create run settings
    run_settings = exp.create_run_settings("python", test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)

    # Create the SmartSim Model
    smartsim_model = exp.create_model("smartsim_model", run_settings)

    db_args = {
        "port": test_port + 1,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "testdb_colo",
    }
    # Create model with colocated database

    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        test_script,
        db_args,
        on_wlm=on_wlm,
    )

    with make_entity_context(exp, smartsim_model):
        exp.start(smartsim_model)

    check_not_failed(exp, smartsim_model)


@pytest.mark.parametrize("db_type", supported_dbs)
def test_multidb_standard_then_colo(fileutils, test_dir, wlmutils, coloutils, db_type):
    """Create regular database then colocate_db_tcp/uds with unique db_identifiers"""

    # Retrieve parameters from testing environment
    test_port = wlmutils.get_test_port()

    test_script = fileutils.get_test_conf_path("smartredis/multidbid.py")
    test_interface = wlmutils.get_test_interface()
    test_launcher = wlmutils.get_test_launcher()

    # start a new Experiment for this section
    exp = Experiment(
        "test_multidb_standard_then_colo", exp_path=test_dir, launcher=test_launcher
    )

    # create and generate an instance of the Orchestrator database
    db = exp.create_database(
        port=test_port,
        interface=test_interface,
        db_identifier="testdb_reg",
        hosts=choose_host(wlmutils),
    )

    db_args = {
        "port": test_port + 1,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "testdb_colo",
    }
    # Create model with colocated database
    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        test_script,
        db_args,
        on_wlm=on_wlm,
    )

    with make_entity_context(exp, db), make_entity_context(exp, smartsim_model):
        exp.start(db)
        exp.start(smartsim_model, block=True)

    check_not_failed(exp, smartsim_model, db)


@pytest.mark.parametrize("db_type", supported_dbs)
def test_multidb_colo_then_standard(fileutils, test_dir, wlmutils, coloutils, db_type):
    """create regular database then colocate_db_tcp/uds with unique db_identifiers"""

    # Retrieve parameters from testing environment
    test_port = wlmutils.get_test_port()

    test_script = fileutils.get_test_conf_path("smartredis/multidbid.py")
    test_interface = wlmutils.get_test_interface()
    test_launcher = wlmutils.get_test_launcher()

    # start a new Experiment
    exp = Experiment(
        "test_multidb_colo_then_standard", exp_path=test_dir, launcher=test_launcher
    )

    db_args = {
        "port": test_port,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "testdb_colo",
    }

    # Create model with colocated database
    smartsim_model = coloutils.setup_test_colo(
        fileutils, db_type, exp, test_script, db_args, on_wlm=on_wlm
    )

    # create and start an instance of the Orchestrator database
    db = exp.create_database(
        port=test_port + 1,
        interface=test_interface,
        db_identifier="testdb_reg",
        hosts=choose_host(wlmutils),
    )

    with make_entity_context(exp, db), make_entity_context(exp, smartsim_model):
        exp.start(db)
        exp.start(smartsim_model, block=True)

    check_not_failed(exp, db, smartsim_model)


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
@pytest.mark.parametrize("db_type", supported_dbs)
def test_launch_cluster_orc_single_dbid(
    test_dir, coloutils, fileutils, wlmutils, db_type
):
    """test clustered 3-node orchestrator with single command with a database identifier"""
    # TODO detect number of nodes in allocation and skip if not sufficent

    exp_name = "test_launch_cluster_orc_single_dbid"
    launcher = wlmutils.get_test_launcher()
    test_port = wlmutils.get_test_port()
    test_script = fileutils.get_test_conf_path("smartredis/multidbid.py")
    exp = Experiment(exp_name, launcher=launcher, exp_path=test_dir)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc: Orchestrator = exp.create_database(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface=network_interface,
        single_cmd=True,
        hosts=wlmutils.get_test_hostlist(),
        db_identifier="testdb_reg",
    )

    db_args = {
        "port": test_port,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "testdb_colo",
    }

    # Create model with colocated database
    smartsim_model = coloutils.setup_test_colo(
        fileutils, db_type, exp, test_script, db_args, on_wlm=on_wlm
    )

    with make_entity_context(exp, orc), make_entity_context(exp, smartsim_model):
        exp.start(orc, block=True)
        exp.start(smartsim_model, block=True)
        job_dict = exp._control._jobs.get_db_host_addresses()
        assert len(job_dict[orc.entities[0].db_identifier]) == 3

    check_not_failed(exp, orc, smartsim_model)
