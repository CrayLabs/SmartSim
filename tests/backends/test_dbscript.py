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
from smartsim.error.errors import SSUnsupportedError, DBIDConflictError
from smartsim.log import get_logger

from smartsim.entity.dbobject import DBScript

from smartredis import *

logger = get_logger(__name__)

should_run = True

supported_dbs = ["uds", "tcp"]

try:
    import torch
except ImportError:
    should_run = False

should_run &= "torch" in installed_redisai_backends()


def timestwo(x):
    return 2 * x


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script(fileutils, wlmutils, mlutils):
    """Test DB scripts on remote DB"""

    # Set experiment name
    exp_name = "test-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create the RunSettings
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)

    # Create the SmartSim Model
    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    # Create the SmartSim database
    db = exp.create_database(port=test_port, interface=test_interface)
    exp.generate(db)

    # Define the torch script string
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    # Add the script via file
    smartsim_model.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Add script via string
    smartsim_model.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Add script function
    smartsim_model.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Assert we have all three scripts
    assert len(smartsim_model._db_scripts) == 3

    # Launch and check successful completion
    try:
        exp.start(db, smartsim_model, block=True)
        statuses = exp.get_status(smartsim_model)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    finally:
        exp.stop(db)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script_ensemble(fileutils, wlmutils, mlutils):
    """Test DB scripts on remote DB"""

    # Set experiment name
    exp_name = "test-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create RunSettings
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)

    # Create Ensemble with two identical models
    ensemble = exp.create_ensemble(
        "dbscript_ensemble", run_settings=run_settings, replicas=2
    )
    ensemble.set_path(test_dir)

    # Create SmartSim model
    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    # Create SmartSim database
    db = exp.create_database(port=test_port, interface=test_interface)
    exp.generate(db)

    # Create the script string
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    # Add script via file for the Ensemble object
    ensemble.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Add script via string for each ensemble entity
    for entity in ensemble:
        entity.disable_key_prefixing()
        entity.add_script(
            "test_script2",
            script=torch_script_str,
            device=test_device,
            devices_per_node=test_num_gpus,
        )

    # Add script via function
    ensemble.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Add an additional ensemble member and attach a script to the new member
    ensemble.add_model(smartsim_model)
    smartsim_model.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Assert we have added both models to the ensemble
    assert len(ensemble._db_scripts) == 2

    # Assert we have added all three models to entities in ensemble
    assert all([len(entity._db_scripts) == 3 for entity in ensemble])

    try:
        exp.start(db, ensemble, block=True)
        statuses = exp.get_status(ensemble)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    finally:
        exp.stop(db)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script(fileutils, wlmutils, mlutils):
    """Test DB Scripts on colocated DB"""

    # Set the experiment name
    exp_name = "test-colocated-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher)

    # Create RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    # Create model with colocated database
    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db_tcp(
        port=test_port, db_cpus=1, debug=True, ifname=test_interface
    )

    # Create string for script creation
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    # Add script via file
    colo_model.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
    )
    # Add script via string
    colo_model.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Assert we have added both models
    assert len(colo_model._db_scripts) == 2

    for db_script in colo_model._db_scripts:
        logger.debug(db_script)

    try:
        exp.start(colo_model, block=True)
        statuses = exp.get_status(colo_model)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    finally:
        exp.stop(colo_model)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script_ensemble(fileutils, wlmutils, mlutils):
    """Test DB Scripts on colocated DB from ensemble, first colocating DB,
    then adding script.
    """

    # Set experiment name
    exp_name = "test-colocated-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher)

    # Create RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    # Create SmartSim Ensemble with two identical models
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    # Create a SmartSim model
    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)

    # Colocate a db with each ensemble entity and add a script
    # to each entity via file
    for i, entity in enumerate(colo_ensemble):
        entity.disable_key_prefixing()
        entity.colocate_db_tcp(
            port=test_port + i,
            db_cpus=1,
            debug=True,
            ifname=test_interface,
        )

        entity.add_script(
            "test_script1",
            script_path=torch_script,
            device=test_device,
            devices_per_node=test_num_gpus,
        )

    # Colocate a db with the non-ensemble Model
    colo_model.colocate_db_tcp(
        port=test_port + len(colo_ensemble),
        db_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    # Add a script to the non-ensemble model
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"
    colo_ensemble.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Add the third SmartSim model to the ensemble
    colo_ensemble.add_model(colo_model)

    # Add another script via file to the entire ensemble
    colo_model.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Assert we have added one model to the ensemble
    assert len(colo_ensemble._db_scripts) == 1
    # Assert we have added both models to each entity
    assert all([len(entity._db_scripts) == 2 for entity in colo_ensemble])

    # Launch and check successful completion
    try:
        exp.start(colo_ensemble, block=True)
        statuses = exp.get_status(colo_ensemble)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    finally:
        exp.stop(colo_ensemble)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script_ensemble_reordered(fileutils, wlmutils, mlutils):
    """Test DB Scripts on colocated DB from ensemble, first adding the
    script to the ensemble, then colocating the DB"""

    # Set Experiment name
    exp_name = "test-colocated-db-script-reord"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher)

    # Create RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    # Create Ensemble with two identical SmartSim Model
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    # Create an additional SmartSim Model entity
    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)

    # Add a script via string to the ensemble members
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"
    colo_ensemble.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Add a colocated database to the ensemble members
    # and then add a script via file
    for i, entity in enumerate(colo_ensemble):
        entity.disable_key_prefixing()
        entity.colocate_db_tcp(
            port=test_port + i,
            db_cpus=1,
            debug=True,
            ifname=test_interface,
        )

        entity.add_script(
            "test_script1",
            script_path=torch_script,
            device=test_device,
            devices_per_node=test_num_gpus,
        )

    # Add a colocated database to the non-ensemble SmartSim Model
    colo_model.colocate_db_tcp(
        port=test_port + len(colo_ensemble),
        db_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    # Add the non-ensemble SmartSim Model to the Ensemble
    # and then add a script via file
    colo_ensemble.add_model(colo_model)
    colo_model.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Assert we have added one model to the ensemble
    assert len(colo_ensemble._db_scripts) == 1
    # Assert we have added both models to each entity
    assert all([len(entity._db_scripts) == 2 for entity in colo_ensemble])

    # Launch and check successful completion
    try:
        exp.start(colo_ensemble, block=True)
        statuses = exp.get_status(colo_ensemble)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    finally:
        exp.stop(colo_ensemble)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script_errors(fileutils, wlmutils, mlutils):
    """Test DB Scripts error when setting a serialized function on colocated DB"""

    # Set Experiment name
    exp_name = "test-colocated-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create SmartSim experiment
    exp = Experiment(exp_name, launcher=test_launcher)

    # Create RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    # Create a SmartSim model with a colocated database
    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db_tcp(
        port=test_port,
        db_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    # Check that an error is raised for adding in-memory
    # function when using colocated deployment
    with pytest.raises(SSUnsupportedError):
        colo_model.add_function(
            "test_func",
            function=timestwo,
            device=test_device,
            devices_per_node=test_num_gpus,
        )

    # Create ensemble with two identical SmartSim Model entities
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    # Add a colocated database for each ensemble member
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_db_tcp(
            port=test_port + i,
            db_cpus=1,
            debug=True,
            ifname=test_interface,
        )

    # Check that an exception is raised when adding an in-memory
    # function to the ensemble with colocated databases
    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_function(
            "test_func",
            function=timestwo,
            device=test_device,
            devices_per_node=test_num_gpus,
        )

    # Create an ensemble with two identical SmartSim Model entities
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    # Add an in-memory function to the ensemble
    colo_ensemble.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
    )

    # Check that an error is raised when trying to add
    # a colocated database to ensemble members that have
    # an in-memory script
    for i, entity in enumerate(colo_ensemble):
        with pytest.raises(SSUnsupportedError):
            entity.colocate_db_tcp(
                port=test_port + i,
                db_cpus=1,
                debug=True,
                ifname=test_interface,
            )

    # Check that an error is raised when trying to add
    # a colocated database to an Ensemble that has
    # an in-memory script
    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_model(colo_model)


def test_inconsistent_params_db_script(fileutils):
    """Test error when devices_per_node>1 and when devices is set to CPU in DBScript constructor"""

    torch_script = fileutils.get_test_conf_path("torchscript.py")
    with pytest.raises(SSUnsupportedError) as ex:
        db_script = DBScript(
            name="test_script_db",
            script_path=torch_script,
            device="CPU",
            devices_per_node=2,
        )
    assert (
        ex.value.args[0]
        == "Cannot set devices_per_node>1 if CPU is specified under devices"
    )


@pytest.mark.parametrize("db_type", supported_dbs)
def test_db_identifier_create_database_then_colocated_db_model(
    fileutils, wlmutils, coloutils, db_type
):
    """Test that it is possible to create_database then colocate_db_uds/colocate_db_tcp
    with unique db_identifiers"""

    # Set experiment name
    exp_name = "test_db_identifier_create_database_then_colocated_db_model_tcp"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher)

    assert exp.dbs_in_use() == set()

    # create regular database fist
    orc = exp.create_database(
        port=test_port, interface=test_interface, db_identifier="my_db"
    )

    exp.generate(orc)

    assert orc.name == "my_db"
    assert exp.dbs_in_use() == {"my_db"}

    # Create colocated RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    #  # Create the SmartSim Model
    smartsim_model = exp.create_model("colocated_model", colo_settings)
    smartsim_model.set_path(test_dir)

    db_args = {
        "port": test_port,
        "db_cpus": 1,
        "debug": True,
        "ifname": test_interface,
        "db_identifier": "my_db",
    }

    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis.py",
        db_args,
    )

    assert smartsim_model.run_settings.colocated_db_settings["db_identifier"] == "my_db"

    with pytest.raises(DBIDConflictError) as ex:
        exp.start(orc, smartsim_model, block=False)

    assert (
        "has already been used. Pass in a unique name for db_identifier"
        in ex.value.args[0]
    )


@pytest.mark.parametrize("db_type", supported_dbs)
def test_db_identifier_colocate_db_then_create_database(
    fileutils, wlmutils, coloutils, db_type
):
    """Test colocate_db_uds/colocate_db_tcp then create_database db_identifier uniqueness"""

    # Set experiment name
    exp_name = "test_db_identifier_colocate_db_then_create_database"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("run_tf_dbmodel_smartredis.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher)

    assert exp.dbs_in_use() == set()

    # Create colocated RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks_per_node(1)

    # Create the SmartSim Model
    smartsim_model = exp.create_model("colocated_model", colo_settings)
    smartsim_model.set_path(test_dir)

    db_args = {
        "port": test_port,
        "db_cpus": 1,
        "debug": True,
        "ifname": test_interface,
        "db_identifier": "my_db",
    }

    smartsim_model = coloutils.setup_test_colo(
        fileutils,
        db_type,
        exp,
        "send_data_local_smartredis_with_dbid.py",
        db_args,
    )

    assert smartsim_model.run_settings.colocated_db_settings["db_identifier"] == "my_db"

    # Create Database
    orc = exp.create_database(
        port=test_port, interface=test_interface, db_identifier="my_db"
    )

    exp.generate(orc)
    assert orc.name == "my_db"

    # assert exp.dbs_in_use() == {"my_db"}

    with pytest.raises(DBIDConflictError) as ex:
        exp.start(orc, smartsim_model, block=False)

    assert (
        "has already been used. Pass in a unique name for db_identifier"
        in ex.value.args[0]
    )


def test_db_identifier_multiple_create_database_not_unique(
    fileutils, wlmutils, mlutils
):
    """Test uniqueness of db_identifier several calls to create_database, with non unique names,
    checking error is raised before exp start is called"""

    # Set experiment name
    exp_name = "test_db_identifier_multiple_create_database_not_unique"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher)

    assert exp.dbs_in_use() == set()

    # CREATE DATABASE with db_identifier
    orc = exp.create_database(
        port=test_port, interface=test_interface, db_identifier="my_db"
    )
    exp.generate(orc)

    assert orc.name == "my_db"
    assert exp.dbs_in_use() == {"my_db"}

    # CREATE DATABASE with db_identifier
    with pytest.raises(DBIDConflictError) as ex:
        orc2 = exp.create_database(
            port=test_port, interface=test_interface, db_identifier="my_db"
        )
    assert (
        "has already been used. Pass in a unique name for db_identifier"
        in ex.value.args[0]
    )


def test_db_identifier_create_standard_once(fileutils, wlmutils, mlutils):
    """One call to create database with a database identifier"""

    # Set experiment name
    exp_name = "test_db_identifier_env_vars_good_create_standard_once"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create the RunSettings
    run_settings = exp.create_run_settings("python", "smartredis/dbid.py")
    run_settings.set_tasks_per_node(1)

    # Create the SmartSim database
    db = exp.create_database(
        port=test_port,
        db_nodes=1,
        interface=test_interface,
        db_identifier="testdb_colo",
    )
    exp.generate(db)

    #try:
    exp.start(db)

    #finally:
    exp.stop(db)

    print(exp.summary())


def test_multidb_create_standard_twice(fileutils, wlmutils):
    """Multiple calls to create database with unique db_identifiers"""

    # Retrieve parameters from testing environment
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()

    # start a new Experiment for this section
    exp = Experiment(
        "test_db_identifier_env_vars_good_create_standard_twice", launcher="local"
    )

    # create and start an instance of the Orchestrator database
    db = exp.create_database(
        port=test_port, interface=test_interface, db_identifier="testdb_reg"
    )
    exp.generate(db)

    # create databse
    db2 = exp.create_database(
        port=test_port + 1, interface=test_interface, db_identifier="testdb_colo"
    )
    # create regular database fist
    exp.generate(db2)

    # launch
    #try:
    exp.start(db, db2)
    #finally:
    exp.stop(db, db2)
    print(exp.summary())


@pytest.mark.parametrize("db_type", supported_dbs)
def test_multidb_colo_once(fileutils, wlmutils, coloutils, db_type):
    """create one model with colocated database with db_identifier"""

    # Retrieve parameters from testing environment
    test_port = wlmutils.get_test_port()
    test_dir = fileutils.make_test_dir()
    test_script = fileutils.get_test_conf_path("smartredis/dbid.py")

    # start a new Experiment for this section
    exp = Experiment("test_db_identifier_colo_once", launcher="local")

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

    # try:
    exp.start(smartsim_model)
    #     statuses = exp.get_status(smartsim_model)
    #     assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    # finally:
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

    # start a new Experiment for this section
    exp = Experiment("test_db_identifier_standard_then_colo", launcher="local")

    # create run settings
    run_settings = exp.create_run_settings("python", test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks_per_node(1)

    # create and start an instance of the Orchestrator database
    db = exp.create_database(
        port=test_port, interface=test_interface, db_identifier="testdb_reg"
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

    #try:
    exp.start(db)
    exp.start(smartsim_model)
    statuses = exp.get_status(smartsim_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    #finally:
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

    # start a new Experiment
    exp = Experiment(
        "test_db_identifier_colo_then_standard", exp_path=test_dir, launcher="local"
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

    smartsim_model.set_path(test_dir)

    # create and start an instance of the Orchestrator database
    db = exp.create_database(
        port=test_port + 1, interface=test_interface, db_identifier="testdb_reg"
    )
    exp.generate(db)

    #try:
    
    exp.start(db)
    exp.start(smartsim_model)
    #statuses = exp.get_status(smartsim_model)
    #assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    # check that model can be restarted
   # exp.start(smartsim_model)
    #finally:
    exp.stop(db)
    exp.stop(smartsim_model)
    print(exp.summary())


@pytest.mark.skipif(
    pytest.test_launcher not in pytest.wlm_options,
    reason="Not testing WLM integrations",
)
def test_launch_cluster_orc_single_dbidentifier(fileutils, wlmutils):
    """test clustered 3-node orchestrator with single command with a database identifier"""
    # TODO detect number of nodes in allocation and skip if not sufficent

    exp_name = "test-launch-auto-cluster-orc-single-dbid"
    launcher = wlmutils.get_test_launcher()
    exp = Experiment(exp_name, launcher=launcher)
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
