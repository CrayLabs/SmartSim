import sys

import pytest

from smartsim import Experiment, status
from smartsim._core.utils import installed_redisai_backends
from smartsim.error.errors import SSUnsupportedError

should_run = True

try:
    import torch
except ImportError:
    should_run = False

should_run &= "torch" in installed_redisai_backends()


def timestwo(x):
    return 2 * x


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script(fileutils, wlmutils):
    """Test DB scripts on remote DB"""

    exp_name = "test-db-script"

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    exp = Experiment(exp_name, exp_path=test_dir, launcher="local")
    # create colocated model
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    db = exp.create_database(port=wlmutils.get_test_port(), interface="lo")
    exp.generate(db)

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    smartsim_model.add_script("test_script1", script_path=torch_script, device="CPU")
    smartsim_model.add_script("test_script2", script=torch_script_str, device="CPU")
    smartsim_model.add_function("test_func", function=timestwo, device="CPU")

    # Assert we have all three models
    assert len(smartsim_model._db_scripts) == 3

    exp.start(db, smartsim_model, block=True)
    statuses = exp.get_status(smartsim_model)
    exp.stop(db)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script_ensemble(fileutils, wlmutils):
    """Test DB scripts on remote DB"""

    exp_name = "test-db-script"

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    exp = Experiment(exp_name, exp_path=test_dir, launcher="local")
    # create colocated model
    run_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    ensemble = exp.create_ensemble(
        "dbscript_ensemble", run_settings=run_settings, replicas=2
    )
    ensemble.set_path(test_dir)

    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    db = exp.create_database(port=wlmutils.get_test_port(), interface="lo")
    exp.generate(db)

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    ensemble.add_script("test_script1", script_path=torch_script, device="CPU")

    for entity in ensemble:
        entity.disable_key_prefixing()
        entity.add_script("test_script2", script=torch_script_str, device="CPU")

    ensemble.add_function("test_func", function=timestwo, device="CPU")

    # Ensemble must add all available DBScripts to new entity
    ensemble.add_model(smartsim_model)
    smartsim_model.add_script("test_script2", script=torch_script_str, device="CPU")

    # Assert we have added both models to the ensemble
    assert len(ensemble._db_scripts) == 2
    # Assert we have added all three models to entities in ensemble
    assert all([len(entity._db_scripts) == 3 for entity in ensemble])

    exp.start(db, ensemble, block=True)
    statuses = exp.get_status(ensemble)
    exp.stop(db)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script(fileutils, wlmutils):
    """Test DB Scripts on colocated DB"""

    exp_name = "test-colocated-db-script"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=wlmutils.get_test_port(),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    colo_model.add_script("test_script1", script_path=torch_script, device="CPU")
    colo_model.add_script("test_script2", script=torch_script_str, device="CPU")

    # Assert we have added both models
    assert len(colo_model._db_scripts) == 2

    for db_script in colo_model._db_scripts:
        print(db_script)

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script_ensemble(fileutils, wlmutils):
    """Test DB Scripts on colocated DB from ensemble, first colocating DB,
    then adding script.
    """

    exp_name = "test-colocated-db-script"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)

    for i, entity in enumerate(colo_ensemble):
        entity.disable_key_prefixing()
        entity.colocate_db(
            port=wlmutils.get_test_port() + i,
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo",
        )

        entity.add_script("test_script1", script_path=torch_script, device="CPU")

    colo_model.colocate_db(
        port=wlmutils.get_test_port() + len(colo_ensemble),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    colo_ensemble.add_script("test_script2", script=torch_script_str, device="CPU")

    colo_ensemble.add_model(colo_model)
    colo_model.add_script("test_script1", script_path=torch_script, device="CPU")

    # Assert we have added one model to the ensemble
    assert len(colo_ensemble._db_scripts) == 1
    # Assert we have added both models to each entity
    assert all([len(entity._db_scripts) == 2 for entity in colo_ensemble])

    exp.start(colo_ensemble, block=True)
    statuses = exp.get_status(colo_ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script_ensemble_reordered(fileutils, wlmutils):
    """Test DB Scripts on colocated DB from ensemble, first adding the
    script to the ensemble, then colocating the DB"""

    exp_name = "test-colocated-db-script"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"
    colo_ensemble.add_script("test_script2", script=torch_script_str, device="CPU")

    for i, entity in enumerate(colo_ensemble):
        entity.disable_key_prefixing()
        entity.colocate_db(
            port=wlmutils.get_test_port() + i,
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo",
        )

        entity.add_script("test_script1", script_path=torch_script, device="CPU")

    colo_model.colocate_db(
        port=wlmutils.get_test_port() + len(colo_ensemble),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )

    colo_ensemble.add_model(colo_model)
    colo_model.add_script("test_script1", script_path=torch_script, device="CPU")

    # Assert we have added one model to the ensemble
    assert len(colo_ensemble._db_scripts) == 1
    # Assert we have added both models to each entity
    assert all([len(entity._db_scripts) == 2 for entity in colo_ensemble])

    exp.start(colo_ensemble, block=True)
    statuses = exp.get_status(colo_ensemble)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script_errors(fileutils, wlmutils):
    """Test DB Scripts error when setting a serialized function on colocated DB"""

    exp_name = "test-colocated-db-script"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=wlmutils.get_test_port(),
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo",
    )

    with pytest.raises(SSUnsupportedError):
        colo_model.add_function("test_func", function=timestwo, device="CPU")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    for i, entity in enumerate(colo_ensemble):
        entity.colocate_db(
            port=wlmutils.get_test_port() + i,
            db_cpus=1,
            limit_app_cpus=False,
            debug=True,
            ifname="lo",
        )

    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_function("test_func", function=timestwo, device="CPU")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_ensemble = exp.create_ensemble(
        "colocated_ensemble", run_settings=colo_settings, replicas=2
    )
    colo_ensemble.set_path(test_dir)

    colo_ensemble.add_function("test_func", function=timestwo, device="CPU")

    for i, entity in enumerate(colo_ensemble):
        with pytest.raises(SSUnsupportedError):
            entity.colocate_db(
                port=wlmutils.get_test_port() + i,
                db_cpus=1,
                limit_app_cpus=False,
                debug=True,
                ifname="lo",
            )

    with pytest.raises(SSUnsupportedError):
        colo_ensemble.add_model(colo_model)
