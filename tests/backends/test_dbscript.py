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

import os
import sys

import pytest
from smartredis import *

from smartsim import Experiment, status
from smartsim._core.utils import installed_redisai_backends
from smartsim.entity.dbobject import DBScript
from smartsim.error.errors import SSUnsupportedError
from smartsim.log import get_logger
from smartsim.settings import MpiexecSettings, MpirunSettings

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
def test_db_script(fileutils, test_dir, wlmutils, mlutils):
    """Test DB scripts on remote DB"""

    # Set experiment name
    exp_name = "test-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create the RunSettings
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    # Create the SmartSim Model
    smartsim_model = exp.create_model("smartsim_model", run_settings)

    # Create the SmartSim database
    host = wlmutils.choose_host(run_settings)
    db = exp.create_database(port=test_port, interface=test_interface, hosts=host)
    exp.generate(db, smartsim_model)

    # Define the torch script string
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    # Add the script via file
    smartsim_model.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add script via string
    smartsim_model.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add script function
    smartsim_model.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
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
def test_db_script_ensemble(fileutils, test_dir, wlmutils, mlutils):
    """Test DB scripts on remote DB"""

    # Set experiment name
    exp_name = "test-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create RunSettings
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    # Create Ensemble with two identical models
    ensemble = exp.create_ensemble(
        "dbscript_ensemble", run_settings=run_settings, replicas=2
    )

    # Create SmartSim model
    smartsim_model = exp.create_model("smartsim_model", run_settings)

    # Create SmartSim database
    host = wlmutils.choose_host(run_settings)
    db = exp.create_database(port=test_port, interface=test_interface, hosts=host)
    exp.generate(db)

    # Create the script string
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    # Add script via file for the Ensemble object
    ensemble.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add script via string for each ensemble entity
    for entity in ensemble:
        entity.disable_key_prefixing()
        entity.add_script(
            "test_script2",
            script=torch_script_str,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
        )

    # Add script via function
    ensemble.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add an additional ensemble member and attach a script to the new member
    ensemble.add_model(smartsim_model)
    smartsim_model.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added both models to the ensemble
    assert len(ensemble._db_scripts) == 2

    # Assert we have added all three models to entities in ensemble
    assert all([len(entity._db_scripts) == 3 for entity in ensemble])

    exp.generate(ensemble)

    try:
        exp.start(db, ensemble, block=True)
        statuses = exp.get_status(ensemble)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    finally:
        exp.stop(db)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script(fileutils, test_dir, wlmutils, mlutils):
    """Test DB Scripts on colocated DB"""

    # Set the experiment name
    exp_name = "test-colocated-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create the SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # Create RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create model with colocated database
    colo_model = exp.create_model("colocated_model", colo_settings)
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
        first_device=0,
    )
    # Add script via string
    colo_model.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added both models
    assert len(colo_model._db_scripts) == 2

    exp.generate(colo_model)

    for db_script in colo_model._db_scripts:
        logger.debug(db_script)

    try:
        exp.start(colo_model, block=True)
        statuses = exp.get_status(colo_model)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    finally:
        exp.stop(colo_model)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script_ensemble(fileutils, test_dir, wlmutils, mlutils):
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
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # Create RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create SmartSim Ensemble with two identical models
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )

    # Create a SmartSim model
    colo_model = exp.create_model("colocated_model", colo_settings)

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
            first_device=0,
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
        first_device=0,
    )

    # Add the third SmartSim model to the ensemble
    colo_ensemble.add_model(colo_model)

    # Add another script via file to the entire ensemble
    colo_model.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added one model to the ensemble
    assert len(colo_ensemble._db_scripts) == 1
    # Assert we have added both models to each entity
    assert all([len(entity._db_scripts) == 2 for entity in colo_ensemble])

    exp.generate(colo_ensemble)

    # Launch and check successful completion
    try:
        exp.start(colo_ensemble, block=True)
        statuses = exp.get_status(colo_ensemble)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    finally:
        exp.stop(colo_ensemble)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script_ensemble_reordered(fileutils, test_dir, wlmutils, mlutils):
    """Test DB Scripts on colocated DB from ensemble, first adding the
    script to the ensemble, then colocating the DB"""

    # Set Experiment name
    exp_name = "test-colocated-db-script-reord"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # Create RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create Ensemble with two identical SmartSim Model
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )

    # Create an additional SmartSim Model entity
    colo_model = exp.create_model("colocated_model", colo_settings)

    # Add a script via string to the ensemble members
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"
    colo_ensemble.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
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
            first_device=0,
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
        first_device=0,
    )

    # Assert we have added one model to the ensemble
    assert len(colo_ensemble._db_scripts) == 1
    # Assert we have added both models to each entity
    assert all([len(entity._db_scripts) == 2 for entity in colo_ensemble])

    exp.generate(colo_ensemble)

    # Launch and check successful completion
    try:
        exp.start(colo_ensemble, block=True)
        statuses = exp.get_status(colo_ensemble)
        assert all([stat == status.STATUS_COMPLETED for stat in statuses])
    finally:
        exp.stop(colo_ensemble)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script_errors(fileutils, test_dir, wlmutils, mlutils):
    """Test DB Scripts error when setting a serialized function on colocated DB"""

    # Set Experiment name
    exp_name = "test-colocated-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")

    # Create SmartSim experiment
    exp = Experiment(exp_name, launcher=test_launcher, exp_path=test_dir)

    # Create RunSettings
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_settings.set_nodes(1)
    colo_settings.set_tasks(1)

    # Create a SmartSim model with a colocated database
    colo_model = exp.create_model("colocated_model", colo_settings)
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
            first_device=0,
        )

    # Create ensemble with two identical SmartSim Model entities
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )

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
            first_device=0,
        )

    # Create an ensemble with two identical SmartSim Model entities
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )

    # Add an in-memory function to the ensemble
    colo_ensemble.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
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
        _ = DBScript(
            name="test_script_db",
            script_path=torch_script,
            device="CPU",
            devices_per_node=2,
            first_device=0,
        )
    assert (
        ex.value.args[0]
        == "Cannot set devices_per_node>1 if CPU is specified under devices"
    )
    with pytest.raises(SSUnsupportedError) as ex:
        _ = DBScript(
            name="test_script_db",
            script_path=torch_script,
            device="CPU",
            devices_per_node=1,
            first_device=5,
        )
    assert (
        ex.value.args[0]
        == "Cannot set first_device>0 if CPU is specified under devices"
    )


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script_ensemble_duplicate(fileutils, test_dir, wlmutils, mlutils):
    """Test DB scripts on remote DB"""

    # Set experiment name
    exp_name = "test-db-script"

    # Retrieve parameters from testing environment
    test_launcher = wlmutils.get_test_launcher()
    test_interface = wlmutils.get_test_interface()
    test_port = wlmutils.get_test_port()
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create SmartSim Experiment
    exp = Experiment(exp_name, exp_path=test_dir, launcher=test_launcher)

    # Create RunSettings
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    # Create Ensemble with two identical models
    ensemble = exp.create_ensemble(
        "dbscript_ensemble", run_settings=run_settings, replicas=2
    )

    # Create SmartSim model
    smartsim_model = exp.create_model("smartsim_model", run_settings)
    # Create 2nd SmartSim model
    smartsim_model_2 = exp.create_model("smartsim_model_2", run_settings)
    # Create the script string
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    # Add the first ML script to all of the ensemble members
    ensemble.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Attempt to add a duplicate ML model to Ensemble via Ensemble.add_script()
    with pytest.raises(SSUnsupportedError) as ex:
        ensemble.add_script(
            "test_script1",
            script_path=torch_script,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
        )
    assert ex.value.args[0] == 'A Script with name "test_script1" already exists'

    # Add the first function to all of the ensemble members
    ensemble.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Attempt to add a duplicate ML model to Ensemble via Ensemble.add_function()
    with pytest.raises(SSUnsupportedError) as ex:
        ensemble.add_function(
            "test_func",
            function=timestwo,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
        )
    assert ex.value.args[0] == 'A Script with name "test_func" already exists'

    # Add a script with a non-unique name to a SmartSim Model
    smartsim_model.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    with pytest.raises(SSUnsupportedError) as ex:
        ensemble.add_model(smartsim_model)
    assert ex.value.args[0] == 'A Script with name "test_script1" already exists'

    # Add a function with a non-unique name to a SmartSim Model
    smartsim_model_2.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    with pytest.raises(SSUnsupportedError) as ex:
        ensemble.add_model(smartsim_model_2)
    assert ex.value.args[0] == 'A Script with name "test_func" already exists'
