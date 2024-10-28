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

import sys

import pytest
from smartredis import *

from smartsim import Experiment
from smartsim.entity.dbobject import FSScript
from smartsim.error.errors import SSUnsupportedError
from smartsim.log import get_logger
from smartsim.status import JobStatus

logger = get_logger(__name__)

should_run = True

supported_fss = ["uds", "tcp"]

try:
    import torch
except ImportError:
    should_run = False

should_run &= "torch" in []  # todo: update test to replace installed_redisai_backends()


def timestwo(x):
    return 2 * x


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_fs_script(wlm_experiment, prepare_fs, single_fs, fileutils, mlutils):
    """Test FS scripts on remote Fs"""

    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create the RunSettings
    run_settings = wlm_experiment.create_run_settings(
        exe=sys.executable, exe_args=test_script
    )
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    # Create the SmartSim Application
    smartsim_application = wlm_experiment.create_application(
        "smartsim_application", run_settings
    )

    # Create the SmartSim feature store
    fs = prepare_fs(single_fs).featurestore
    wlm_experiment.reconnect_orchestrator(fs.checkpoint_file)
    wlm_experiment.generate(smartsim_application)

    # Define the torch script string
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    # Add the script via file
    smartsim_application.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add script via string
    smartsim_application.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add script function
    smartsim_application.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have all three scripts
    assert len(smartsim_application._fs_scripts) == 3

    # Launch and check successful completion
    wlm_experiment.start(smartsim_application, block=True)
    statuses = wlm_experiment.get_status(smartsim_application)
    assert all([stat == JobStatus.COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_fs_script_ensemble(wlm_experiment, prepare_fs, single_fs, fileutils, mlutils):
    """Test FS scripts on remote FS"""

    # Set wlm_experimenteriment name
    wlm_experiment_name = "test-fs-script"

    # Retrieve parameters from testing environment
    test_device = mlutils.get_test_device()
    test_num_gpus = mlutils.get_test_num_gpus() if pytest.test_device == "GPU" else 1

    test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # Create RunSettings
    run_settings = wlm_experiment.create_run_settings(
        exe=sys.executable, exe_args=test_script
    )
    run_settings.set_nodes(1)
    run_settings.set_tasks(1)

    fs = prepare_fs(single_fs).featurestore
    wlm_experiment.reconnect_feature_store(fs.checkpoint_file)

    # Create Ensemble with two identical applications
    ensemble = wlm_experiment.create_ensemble(
        "fsscript_ensemble", run_settings=run_settings, replicas=2
    )

    # Create SmartSim application
    smartsim_application = wlm_experiment.create_application(
        "smartsim_application", run_settings
    )

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
    ensemble.add_application(smartsim_application)
    smartsim_application.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added both models to the ensemble
    assert len(ensemble._fs_scripts) == 2

    # Assert we have added all three models to entities in ensemble
    assert all([len(entity._fs_scripts) == 3 for entity in ensemble])

    wlm_experiment.generate(ensemble)

    wlm_experiment.start(ensemble, block=True)
    statuses = wlm_experiment.get_status(ensemble)
    assert all([stat == JobStatus.COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_fs_script(fileutils, test_dir, wlmutils, mlutils):
    """Test fs Scripts on colocated fs"""

    # Set the experiment name
    exp_name = "test-colocated-fs-script"

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

    # Create application with colocated feature store
    colo_application = exp.create_application("colocated_application", colo_settings)
    colo_application.colocate_fs_tcp(
        port=test_port, fs_cpus=1, debug=True, ifname=test_interface
    )

    # Create string for script creation
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    # Add script via file
    colo_application.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )
    # Add script via string
    colo_application.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added both models
    assert len(colo_application._fs_scripts) == 2

    exp.generate(colo_application)

    for fs_script in colo_application._fs_scripts:
        logger.debug(fs_script)

    try:
        exp.start(colo_application, block=True)
        statuses = exp.get_status(colo_application)
        assert all([stat == JobStatus.COMPLETED for stat in statuses])
    finally:
        exp.stop(colo_application)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_fs_script_ensemble(fileutils, test_dir, wlmutils, mlutils):
    """Test fs Scripts on colocated fs from ensemble, first colocating fs,
    then adding script.
    """

    # Set experiment name
    exp_name = "test-colocated-fs-script"

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

    # Create SmartSim Ensemble with two identical applications
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )

    # Create a SmartSim application
    colo_application = exp.create_application("colocated_application", colo_settings)

    # Colocate a fs with each ensemble entity and add a script
    # to each entity via file
    for i, entity in enumerate(colo_ensemble):
        entity.disable_key_prefixing()
        entity.colocate_fs_tcp(
            port=test_port + i,
            fs_cpus=1,
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

    # Colocate a feature store with the non-ensemble Application
    colo_application.colocate_fs_tcp(
        port=test_port + len(colo_ensemble),
        fs_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    # Add a script to the non-ensemble application
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"
    colo_ensemble.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add the third SmartSim application to the ensemble
    colo_ensemble.add_application(colo_application)

    # Add another script via file to the entire ensemble
    colo_application.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added one application to the ensemble
    assert len(colo_ensemble._fs_scripts) == 1
    # Assert we have added both applications to each entity
    assert all([len(entity._fs_scripts) == 2 for entity in colo_ensemble])

    exp.generate(colo_ensemble)

    # Launch and check successful completion
    try:
        exp.start(colo_ensemble, block=True)
        statuses = exp.get_status(colo_ensemble)
        assert all([stat == JobStatus.COMPLETED for stat in statuses])
    finally:
        exp.stop(colo_ensemble)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_fs_script_ensemble_reordered(fileutils, test_dir, wlmutils, mlutils):
    """Test fs Scripts on colocated fs from ensemble, first adding the
    script to the ensemble, then colocating the fs"""

    # Set Experiment name
    exp_name = "test-colocated-fs-script-reord"

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

    # Create Ensemble with two identical SmartSim Application
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )

    # Create an additional SmartSim Application entity
    colo_application = exp.create_application("colocated_application", colo_settings)

    # Add a script via string to the ensemble members
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"
    colo_ensemble.add_script(
        "test_script2",
        script=torch_script_str,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Add a colocated feature store to the ensemble members
    # and then add a script via file
    for i, entity in enumerate(colo_ensemble):
        entity.disable_key_prefixing()
        entity.colocate_fs_tcp(
            port=test_port + i,
            fs_cpus=1,
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

    # Add a colocated feature store to the non-ensemble SmartSim Application
    colo_application.colocate_fs_tcp(
        port=test_port + len(colo_ensemble),
        fs_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    # Add the non-ensemble SmartSim Application to the Ensemble
    # and then add a script via file
    colo_ensemble.add_application(colo_application)
    colo_application.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    # Assert we have added one application to the ensemble
    assert len(colo_ensemble._fs_scripts) == 1
    # Assert we have added both applications to each entity
    assert all([len(entity._fs_scripts) == 2 for entity in colo_ensemble])

    exp.generate(colo_ensemble)

    # Launch and check successful completion
    try:
        exp.start(colo_ensemble, block=True)
        statuses = exp.get_status(colo_ensemble)
        assert all([stat == JobStatus.COMPLETED for stat in statuses])
    finally:
        exp.stop(colo_ensemble)


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_fs_script_errors(fileutils, test_dir, wlmutils, mlutils):
    """Test fs Scripts error when setting a serialized function on colocated fs"""

    # Set Experiment name
    exp_name = "test-colocated-fs-script"

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

    # Create a SmartSim application with a colocated feature store
    colo_application = exp.create_application("colocated_application", colo_settings)
    colo_application.colocate_fs_tcp(
        port=test_port,
        fs_cpus=1,
        debug=True,
        ifname=test_interface,
    )

    # Check that an error is raised for adding in-memory
    # function when using colocated deployment
    with pytest.raises(SSUnsupportedError):
        colo_application.add_function(
            "test_func",
            function=timestwo,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
        )

    # Create ensemble with two identical SmartSim Application entities
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=test_script)
    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )

    # Add a colocated feature store for each ensemble member
    for i, entity in enumerate(colo_ensemble):
        entity.colocate_fs_tcp(
            port=test_port + i,
            fs_cpus=1,
            debug=True,
            ifname=test_interface,
        )

    # Check that an exception is raised when adding an in-memory
    # function to the ensemble with colocated feature stores
    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_function(
            "test_func",
            function=timestwo,
            device=test_device,
            devices_per_node=test_num_gpus,
            first_device=0,
        )

    # Create an ensemble with two identical SmartSim Application entities
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
    # a colocated feature store to ensemble members that have
    # an in-memory script
    for i, entity in enumerate(colo_ensemble):
        with pytest.raises(SSUnsupportedError):
            entity.colocate_fs_tcp(
                port=test_port + i,
                fs_cpus=1,
                debug=True,
                ifname=test_interface,
            )

    # Check that an error is raised when trying to add
    # a colocated feature store to an Ensemble that has
    # an in-memory script
    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_application(colo_application)


def test_inconsistent_params_fs_script(fileutils):
    """Test error when devices_per_node>1 and when devices is set to CPU in FSScript constructor"""

    torch_script = fileutils.get_test_conf_path("torchscript.py")
    with pytest.raises(SSUnsupportedError) as ex:
        _ = FSScript(
            name="test_script_fs",
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
        _ = FSScript(
            name="test_script_fs",
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
def test_fs_script_ensemble_duplicate(fileutils, test_dir, wlmutils, mlutils):
    """Test fs scripts on remote fs"""

    # Set experiment name
    exp_name = "test-fs-script"

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

    # Create Ensemble with two identical applications
    ensemble = exp.create_ensemble(
        "fsscript_ensemble", run_settings=run_settings, replicas=2
    )

    # Create SmartSim application
    smartsim_application = exp.create_application("smartsim_application", run_settings)
    # Create 2nd SmartSim application
    smartsim_application_2 = exp.create_application(
        "smartsim_application_2", run_settings
    )
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

    # Add a script with a non-unique name to a SmartSim application
    smartsim_application.add_script(
        "test_script1",
        script_path=torch_script,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    with pytest.raises(SSUnsupportedError) as ex:
        ensemble.add_application(smartsim_application)
    assert ex.value.args[0] == 'A Script with name "test_script1" already exists'

    # Add a function with a non-unique name to a SmartSim Application
    smartsim_application_2.add_function(
        "test_func",
        function=timestwo,
        device=test_device,
        devices_per_node=test_num_gpus,
        first_device=0,
    )

    with pytest.raises(SSUnsupportedError) as ex:
        ensemble.add_application(smartsim_application_2)
    assert ex.value.args[0] == 'A Script with name "test_func" already exists'
