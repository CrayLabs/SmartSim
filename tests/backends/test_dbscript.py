import sys
import os.path as osp
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
    return 2*x

@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_colocated_db_script(fileutils):
    """Test DB Scripts on colocated DB"""

    exp_name = "test-colocated-db-script"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    # create colocated model
    colo_settings = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=6780,
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo"
        )
    
    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    colo_model.add_script("test_script1", script_path=torch_script, device="CPU")
    colo_model.add_script("test_script2", script=torch_script_str, device="CPU")

    # Assert we have added both models
    assert(len(colo_model._db_scripts) == 2)

    for db_script in colo_model._db_scripts:
        print(db_script)

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])


@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script(fileutils):
    """Test DB scripts on remote DB"""

    exp_name = "test-db-script"

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")
    torch_script = fileutils.get_test_conf_path("torchscript.py")

    exp = Experiment(exp_name, exp_path=test_dir, launcher="local")
    # create colocated model
    run_settings = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

    smartsim_model = exp.create_model("smartsim_model", run_settings)
    smartsim_model.set_path(test_dir)

    db = exp.create_database(port=6780, interface="lo")
    exp.generate(db)

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

    smartsim_model.add_script("test_script1", script_path=torch_script, device="CPU")
    smartsim_model.add_script("test_script2", script=torch_script_str, device="CPU")
    smartsim_model.add_function("test_func", function=timestwo, device="CPU")

    # Assert we have added both models
    assert(len(smartsim_model._db_scripts) == 3)

    exp.start(db, smartsim_model, block=True)
    statuses = exp.get_status(smartsim_model)
    exp.stop(db)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

@pytest.mark.skipif(not should_run, reason="Test needs Torch to run")
def test_db_script_error(fileutils):
    """Test DB Scripts error when setting a function on colocated DB"""

    exp_name = "test-colocated-db-script"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("run_dbscript_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)
    colo_model.colocate_db(
        port=6780,
        db_cpus=1,
        limit_app_cpus=False,
        debug=True,
        ifname="lo"
        )
    
    with pytest.raises(SSUnsupportedError):
        colo_model.add_function("test_func", function=timestwo, device="CPU")

  