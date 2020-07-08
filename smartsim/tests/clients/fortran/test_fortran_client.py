import pytest
from smartsim.tests.decorators import compiled_client_test
from smartsim import Experiment
from shutil import which
from os import path

if not which("srun"):
    pytestmark = pytest.mark.skip()

test_dir=path.dirname(path.abspath(__file__))

@compiled_client_test(test_dir = test_dir, target_names=["client_tester"])
def test_put_get_fortran(*args, **kwargs):
    """ This funtion tests putting and getting
        1D and 2D arrays from database.
    """
    experiment = Experiment("client_test")
    alloc = experiment.get_allocation(nodes=4, ppn=2)
    run_settings = {"nodes":1,
                    "ppn": 2,
                    "executable":kwargs['binary_names'][0],
                    "exe_args": "10000",
                    "alloc": alloc}
    client_model = experiment.create_model("client_test",
                                            run_settings=run_settings)
    experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop()
    experiment.release()
