# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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

import sys

import os

import pytest

from smartsim import Experiment, status
from smartsim._core.utils import installed_redisai_backends
from smartsim.error.errors import SSDBIDConflictError
from smartsim.log import get_logger

from smartsim.entity.dbobject import DBScript

from smartredis import *

logger = get_logger(__name__)

should_run = True

supported_dbs = ["uds", "tcp"]

@contextmanager
def start_in_context(exp, entity):
    """Start entity in a context to ensure that it is always stopped"""
    exp.generate(entity)
    try:
        exp.start(entity)
        yield entity
    finally:
        exp.stop(entity)

@pytest.mark.parametrize("db_type", supported_dbs)
def test_db_identifier_standard_then_colo(fileutils, wlmutils, coloutils, db_type):
    """Test that it is possible to create_database then colocate_db_uds/colocate_db_tcp
    with unique db_identifiers"""

    # Set experiment name
    exp_name = "test_db_identifier_standard_then_colo"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("smartredis/db_id_err.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, test_dir, launcher=test_launcher)

    # create regular database
    orc = exp.create_database(
        port=test_port, interface=test_interface, db_identifier="my_db",
        hosts=wlmutils.get_test_hostlist(),
    )
    assert orc.name == "my_db"

    # create run settings
    colo_settings = exp.create_run_settings("python", test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    #  # Create the SmartSim Model
    smartsim_model = exp.create_model("colocated_model", colo_settings)

    db_args = {
        "port": test_port + 1,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "my_db",
    }

    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis_with_dbid_error_test.py",
        db_args,
    )

    assert smartsim_model.run_settings.colocated_db_settings["db_identifier"] == "my_db"

    with start_in_context(exp, orc) as orc:
        with pytest.raises(SSDBIDConflictError) as ex:
            exp.start(smartsim_model, block=True)
        assert (
            "has already been used. Pass in a unique name for db_identifier"
            in ex.value.args[0]
        )


@pytest.mark.parametrize("db_type", supported_dbs)
def test_db_identifier_colo_then_standard(fileutils, wlmutils, coloutils, db_type):
    """Test colocate_db_uds/colocate_db_tcp then create_database with database
    identifiers.
    """

    # Set experiment name
    exp_name = "test_db_identifier_colo_then_standard"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("smartredis/db_id_err.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, test_dir, launcher=test_launcher)

    # Create run settings
    colo_settings = exp.create_run_settings("python", test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    # Create the SmartSim Model
    smartsim_model = exp.create_model("colocated_model", colo_settings)
    smartsim_model.set_path(test_dir)

    db_args = {
        "port": test_port,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "my_db",
    }

    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis_with_dbid_error_test.py",
        db_args,
    )

    assert smartsim_model.run_settings.colocated_db_settings["db_identifier"] == "my_db"

    # Create Database
    orc = exp.create_database(
        port=test_port + 1, interface=test_interface, db_identifier="my_db",
        hosts=wlmutils.get_test_hostlist(),
    )

    exp.generate(orc)
    assert orc.name == "my_db"

    exp.start(smartsim_model, block=True)
    exp.start(orc)


def test_db_identifier_standard_twice_not_unique(wlmutils):
    """Test uniqueness of db_identifier several calls to create_database, with non unique names,
    checking error is raised before exp start is called"""

    # Set experiment name
    exp_name = "test_db_identifier_multiple_create_database_not_unique"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()

    # Create SmartSim Experiment
    exp = Experiment(exp_name, test_dir, launcher=test_launcher)

    # CREATE DATABASE with db_identifier
    orc = exp.create_database(
        port=test_port, interface=test_interface, db_identifier="my_db",
        hosts=wlmutils.get_test_hostlist(),
    )
    exp.generate(orc)

    assert orc.name == "my_db"

    orc2 = exp.create_database(
        port=test_port + 1, interface=test_interface, db_identifier="my_db",
        hosts=wlmutils.get_test_hostlist(),
    )
    exp.generate(orc2)

    assert orc2.name == "my_db"

    # CREATE DATABASE with db_identifier
    with start_in_context(exp, orc) as orc:
        with pytest.raises(SSDBIDConflictError) as ex:
            with start_in_context(exp, orc2) as orc2:
                assert (
                    "has already been used. Pass in a unique name for db_identifier"
                    in ex.value.args[0]
                )


def test_db_identifier_create_standard_once(fileutils, wlmutils):
    """One call to create database with a database identifier"""

    # Set experiment name
    exp_name = "test_db_identifier_create_standard_once"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create the SmartSim database
    db = exp.create_database(
        port=test_port,
        db_nodes=1,
        interface=test_interface,
        db_identifier="testdb_reg",
    )
    exp.generate(db)

    exp.start(db)
    exp.stop(db)

def test_multidb_create_standard_twice(fileutils, wlmutils):
    """Multiple calls to create database with unique db_identifiers"""

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()

    # start a new Experiment for this section
    exp = Experiment(
        "test_multidb_create_standard_twice", exp_path=test_dir, launcher=test_launcher
    )

    # create and start an instance of the Orchestrator database
    db = exp.create_database(
        port=test_port, interface=test_interface, db_identifier="testdb_reg",
        hosts=wlmutils.get_test_hostlist(),
    )

    # create database with different db_id
    db2 = exp.create_database(
        port=test_port + 1, interface=test_interface, db_identifier="testdb_reg2",
        hosts=wlmutils.get_test_hostlist(),
    )

    # launch
    with start_in_context(exp, db) as db, start_in_context(exp, db2) as db2:
        print("Databases started")

    with start_in_context(exp, db) as db, start_in_context(exp, db2) as db2:
        print("Databases restarted")

@pytest.mark.parametrize("db_type", supported_dbs)
def test_multidb_colo_once(fileutils, wlmutils, coloutils, db_type):
    """create one model with colocated database with db_identifier"""

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("smartredis/dbid.py")

    # start a new Experiment for this section
    exp = Experiment("test_multidb_colo_once", test_dir, launcher=test_launcher)

    # create run settings
    run_settings = exp.create_run_settings("python", test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)

    # Create the SmartSim Model
    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

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
        "send_data_local_smartredis_with_dbid.py",
        db_args,
    )

    exp.start(smartsim_model)

    exp.stop(smartsim_model)
    print(exp.summary())


@pytest.mark.parametrize("db_type", supported_dbs)
def test_multidb_standard_then_colo(fileutils, wlmutils, coloutils, db_type):
    """Create regular database then colocate_db_tcp/uds with unique db_identifiers"""

    # Retrieve parameters from testing environment
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("smartredis/multidbid.py")
    test_interface = wlmutils.get_test_interface()
    test_launcher = wlmutils.get_test_launcher()

    # start a new Experiment for this section
    exp = Experiment(
        "test_multidb_standard_then_colo", exp_path=test_dir, launcher=test_launcher
    )

    # create run settings
    run_settings = exp.create_run_settings("python", test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)

    # create and start an instance of the Orchestrator database
    db = exp.create_database(
        port=test_port, interface=test_interface, db_identifier="testdb_reg",
        hosts=wlmutils.get_test_hostlist(),
    )
    exp.generate(db)

    # Create the SmartSim Model
    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

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
        "send_data_local_smartredis_with_dbid.py",
        db_args,
    )

    exp.start(db)
    exp.start(smartsim_model, block=True)

    # test restart colocated db
    exp.start(smartsim_model)

    exp.stop(db)
    # test restart standard db
    exp.start(db)

    exp.stop(db)
    exp.stop(smartsim_model)
    print(exp.summary())


@pytest.mark.parametrize("db_type", supported_dbs)
def test_multidb_colo_then_standard(fileutils, wlmutils, coloutils, db_type):
    """create regular database then colocate_db_tcp/uds with unique db_identifiers"""

    # Retrieve parameters from testing environment
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("smartredis/multidbid.py")
    test_interface = wlmutils.get_test_interface()
    test_launcher = wlmutils.get_test_launcher()

    # start a new Experiment
    exp = Experiment(
        "test_multidb_colo_then_standard", exp_path=test_dir, launcher=test_launcher
    )

    # create run settings
    run_settings = exp.create_run_settings("python", test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)

    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    db_args = {
        "port": test_port,
        "db_cpus": 1,
        "debug": True,
        "db_identifier": "testdb_colo",
    }

    # Create model with colocated database
    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis_with_dbid.py",
        db_args,
    )

    # create and start an instance of the Orchestrator database
    db = exp.create_database(
        port=test_port + 1, interface=test_interface, db_identifier="testdb_reg",
        hosts=wlmutils.get_test_hostlist(),
    )
    exp.generate(db)

    exp.start(db)
    exp.start(smartsim_model)

    # test restart colocated db
    exp.start(smartsim_model)

    exp.stop(db)

    # test restart standard db
    exp.start(db)

    exp.stop(smartsim_model)
    exp.stop(db)
    print(exp.summary())


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_launch_cluster_orc_single_dbid(fileutils, wlmutils):
    """test clustered 3-node orchestrator with single command with a database identifier"""
    # TODO detect number of nodes in allocation and skip if not sufficent

    exp_name = "test_launch_cluster_orc_single_dbid"
    launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, test_dir, launcher=launcher)
    test_dir = fileutils.make_test_dir()

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = exp.create_database(
        wlmutils.get_test_port(),
        db_nodes=3,
        batch=False,
        interface=network_interface,
        single_cmd=True,
        hosts=wlmutils.get_test_hostlist(),
        db_identifier="testdb_reg",
    )
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that orc we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])
