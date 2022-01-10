import pytest

from smartsim import Experiment, status
from smartsim.database import SlurmOrchestrator

# retrieved from pytest fixtures
if pytest.test_launcher not in pytest.wlm_options:
    pytestmark = pytest.mark.skip(reason="Not testing WLM integrations")


def test_launch_slurm_orc(fileutils, wlmutils):
    """test single node orchestrator"""
    launcher = wlmutils.get_test_launcher()
    if launcher != "slurm":
        pytest.skip("Test only runs on systems with Slurm as WLM")

    exp_name = "test-launch-slurm-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = SlurmOrchestrator(6780, batch=False, interface=network_interface)
    orc.set_path(test_dir)

    exp.start(orc, block=True)
    statuses = exp.get_status(orc)

    # don't use assert so that we don't leave an orphan process
    if status.STATUS_FAILED in statuses:
        exp.stop(orc)
        assert False

    exp.stop(orc)
    statuses = exp.get_status(orc)
    assert all([stat == status.STATUS_CANCELLED for stat in statuses])


def test_launch_slurm_cluster_orc(fileutils, wlmutils):
    """test clustered 3-node orchestrator"""

    # TODO detect number of nodes in allocation and skip if not sufficent
    launcher = wlmutils.get_test_launcher()
    if launcher != "slurm":
        pytest.skip("Test only runs on systems with Slurm as WLM")

    exp_name = "test-launch-slurm-cluster-orc"
    exp = Experiment(exp_name, launcher=launcher)
    test_dir = fileutils.make_test_dir(exp_name)

    # batch = False to launch on existing allocation
    network_interface = wlmutils.get_test_interface()
    orc = SlurmOrchestrator(6780, db_nodes=3, batch=False, interface=network_interface)
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


def test_incoming_entities(fileutils, wlmutils):
    """Mirroring of how SmartSim generates SSKEYIN"""

    launcher = wlmutils.get_test_launcher()
    if launcher != "slurm":
        pytest.skip("Test only runs on systems with Slurm as WLM")

    exp_name = "test-incoming-entities"
    exp = Experiment(exp_name, launcher=wlmutils.get_test_launcher())
    test_dir = fileutils.make_test_dir(exp_name)

    network_interface = wlmutils.get_test_interface()
    orc = SlurmOrchestrator(6780, db_nodes=1, batch=False, interface=network_interface)
    orc.set_path(test_dir)

    sleep = fileutils.get_test_conf_path("sleep.py")
    sleep_settings = exp.create_run_settings("python", f"{sleep} --time=3")
    sleep_settings.set_tasks(1)

    sleep_ensemble = exp.create_ensemble(
        "sleep-ensemble", run_settings=sleep_settings, replicas=2
    )

    sskeyin_reader = fileutils.get_test_conf_path("incoming_entities_reader.py")
    sskeyin_reader_settings = exp.create_run_settings("python", f"{sskeyin_reader}")
    sskeyin_reader_settings.set_tasks(1)

    sskeyin_reader_settings.env_vars["NAME_0"] = sleep_ensemble.entities[0].name
    sskeyin_reader_settings.env_vars["NAME_1"] = sleep_ensemble.entities[1].name
    sskeyin_reader = exp.create_model(
        "sskeyin_reader", path=test_dir, run_settings=sskeyin_reader_settings
    )
    sskeyin_reader.register_incoming_entity(sleep_ensemble.entities[0])
    sskeyin_reader.register_incoming_entity(sleep_ensemble.entities[1])

    exp.start(orc, block=False)
    try:
        exp.start(sskeyin_reader, block=True)
        assert exp.get_status(sskeyin_reader)[0] == status.STATUS_COMPLETED
    except Exception as e:
        exp.stop(orc)
        raise e

    exp.stop(orc)
