import pytest
import subprocess
from os import getcwd, path, environ, mkdir, remove
import time
from shutil import rmtree, which
from smartsim import Experiment

# skip all tests
#pytestmark = pytest.mark.skip()

def compiled_client_test_builder(test_path, target_name, exe_args):
    """ This function is used to build and run
        compiled client tests.
    """

    # Check for slurm
    if not which("srun"):
        pytest.skip()

    compile_dir = test_path + "/compile"

    if path.isdir(compile_dir):
        rmtree(compile_dir)
    mkdir(compile_dir)

    binary_name = compile_dir + '/' + target_name
    p = subprocess.run(["cmake", "../"], cwd=compile_dir, capture_output=True)
    p = subprocess.run(["make", target_name], cwd=compile_dir, capture_output=True)
    assert(path.isfile(binary_name))

    if path.isdir("client_test"):
        rmtree("client_test")

    experiment = Experiment("client_test")
    alloc = experiment.get_allocation(nodes=4, ppn=1)
    run_settings = {"nodes":1,
                  "ppn": 1,
                  "executable":binary_name,
                  "exe_args": exe_args,
                  "alloc": alloc}
    client_model = experiment.create_model("client_test", run_settings=run_settings)
    experiment.create_orchestrator_cluster(alloc, db_nodes=3)
    experiment.generate()
    experiment.start()
    experiment.poll(interval=5)
    assert(experiment.get_status(client_model) == "COMPLETED")
    experiment.stop()
    experiment.release()

    if path.isdir(compile_dir):
        rmtree(compile_dir)
    if path.isdir("client_test"):
        rmtree("client_test")

    return True
