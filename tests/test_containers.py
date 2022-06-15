import pytest
from shutil import which
from pathlib import Path
import os

from smartsim import Experiment, status
from smartsim._core.utils import installed_redisai_backends
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model
from smartsim.settings.containers import Singularity

# Check if singularity is available as command line tool
singularity_exists = which('singularity') is not None
containerURI = 'docker://alrigazzi/smartsim-testing:latest'

def test_singularity_commands(fileutils):
    '''Test generation of singularity commands.'''

    # Note: We skip first element so singularity is not needed to run test

    c = Singularity(containerURI)
    cmd = ' '.join(c._container_cmds()[1:])
    assert cmd == f'exec {containerURI}'

    c = Singularity(containerURI, args='--verbose')
    cmd = ' '.join(c._container_cmds()[1:])
    assert cmd == f'exec --verbose {containerURI}'

    c = Singularity(containerURI, args=['--verbose', '--cleanenv'])
    cmd = ' '.join(c._container_cmds()[1:])
    assert cmd == f'exec --verbose --cleanenv {containerURI}'

    c = Singularity(containerURI, mount='/usr/local/bin')
    cmd = ' '.join(c._container_cmds()[1:])
    assert cmd == f'exec --bind /usr/local/bin {containerURI}'

    c = Singularity(containerURI, mount=['/usr/local/bin', '/lus/datasets'])
    cmd = ' '.join(c._container_cmds()[1:])
    assert cmd == f'exec --bind /usr/local/bin,/lus/datasets {containerURI}'

    c = Singularity(containerURI, mount={'/usr/local/bin':'/bin',
                                         '/lus/datasets':'/datasets',
                                         '/cray/css/smartsim':None})
    cmd = ' '.join(c._container_cmds()[1:])
    assert cmd == f'exec --bind /usr/local/bin:/bin,/lus/datasets:/datasets,/cray/css/smartsim {containerURI}'

    c = Singularity(containerURI, args='--verbose', mount='/usr/local/bin')
    cmd = ' '.join(c._container_cmds()[1:])
    assert cmd == f'exec --verbose --bind /usr/local/bin {containerURI}'


@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_basic(fileutils):
    '''Basic argument-less Singularity test'''
    test_dir = fileutils.make_test_dir()

    container = Singularity(containerURI)

    exp = Experiment("singularity_basic", exp_path=test_dir, launcher="local")
    run_settings = exp.create_run_settings("python3", "sleep.py --time=3",
                                            container=container)
    model = exp.create_model("singularity_basic", run_settings)

    script = fileutils.get_test_conf_path("sleep.py")
    model.attach_generator_files(to_copy=[script])
    exp.generate(model, overwrite=True)

    exp.start(model, summary=False)

    # get and confirm status
    stat = exp.get_status(model)[0]
    assert stat == status.STATUS_COMPLETED

    print(exp.summary())


@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_args(fileutils):
    '''Test combinations of args and mount arguments for Singularity'''
    test_dir = fileutils.make_test_dir()
    hometest_dir = os.path.join(str(Path.home()), 'test') # $HOME/test
    mount_paths = {test_dir + '/singularity_args': hometest_dir}
    container = Singularity(containerURI, args='--contain', mount=mount_paths)

    exp = Experiment("singularity_args", launcher="local", exp_path=test_dir)

    run_settings = exp.create_run_settings('python3', 'test/check_dirs.py',
                                           container=container)
    model = exp.create_model("singularity_args", run_settings)
    script = fileutils.get_test_conf_path("check_dirs.py")
    model.attach_generator_files(to_copy=[script])
    exp.generate(model, overwrite=True)

    exp.start(model, summary=False)

    # get and confirm status
    stat = exp.get_status(model)[0]
    assert stat == status.STATUS_COMPLETED

    print(exp.summary())


@pytest.mark.skipif(not singularity_exists, reason="Test needs singularity to run")
def test_singularity_smartredis(fileutils, wlmutils):
    """Run two processes, each process puts a tensor on
    the DB, then accesses the other process's tensor.
    Finally, the tensor is used to run a model.

    Note: This is a containerized port of test_smartredis.py
    """

    test_dir = fileutils.make_test_dir()
    exp = Experiment(
        "smartredis_ensemble_exchange", exp_path=test_dir, launcher="local"
    )

    # create and start a database
    orc = Orchestrator(port=wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc, block=False)

    container = Singularity(containerURI)

    rs = exp.create_run_settings("python3", "producer.py --exchange", container=container)
    params = {"mult": [1, -10]}
    ensemble = Ensemble(
        name="producer",
        params=params,
        run_settings=rs,
        perm_strat="step",
    )

    ensemble.register_incoming_entity(ensemble["producer_0"])
    ensemble.register_incoming_entity(ensemble["producer_1"])

    config = fileutils.get_test_conf_path("smartredis")
    ensemble.attach_generator_files(to_copy=[config])

    exp.generate(ensemble, overwrite=True)

    # start the models
    exp.start(ensemble, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    if not all([stat == status.STATUS_COMPLETED for stat in statuses]):
        exp.stop(orc)
        assert False  # client ensemble failed

    # stop the orchestrator
    exp.stop(orc)

    print(exp.summary())

