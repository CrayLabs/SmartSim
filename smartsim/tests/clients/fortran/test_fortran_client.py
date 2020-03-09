import pytest
import subprocess

from os import getcwd, listdir, path, environ, mkdir, remove

from glob import glob
from shutil import rmtree, which, copyfile

from smartsim import Generator, Controller, State
import time
from subprocess import Popen

def test_put_get_one_dimensional_array_fortran():
    """ This funtion tests putting a one dimensional
        array into the database and then gets it from
        the database and does a comparison.
    """

    # Check for slurm
    if not which("mpiexec"):
        pytest.skip()

    test_path = path.dirname(path.abspath(__file__))
    compile_dir = test_path + "/compile"

    if path.isdir(compile_dir):
        rmtree(compile_dir)
    binary_name = compile_dir + '/' + "client_tester"
    mkdir(compile_dir)
        
    p = subprocess.run(["cmake", "../"], cwd=compile_dir, capture_output=True)
    p = subprocess.run(["make", "client_tester"], cwd=compile_dir, capture_output=True)
    
    assert(path.isfile(binary_name))

    n_db_nodes = 3
    state=State(experiment="client_test")
    state.create_orchestrator(cluster_size=n_db_nodes)

    control_dict = {"launcher": "slurm",
                    "ppn": 1}
    control = Controller(state, **control_dict)
    control.start()

    db_nodes = control._jobs.get_db_nodes()
    db_node = db_nodes[0]
    db_port = state.orc.port

    environ["SSDB"] = db_node + ":" + str(db_port)
    environ["SSNAME"] = "cpp_test"
    environ["SSDATAIN"] = "cpp_test"

    p = subprocess.run(["mpiexec","-n","1",binary_name], cwd=compile_dir, capture_output=True)

    control.stop(ensembles=[], nodes=[], stop_orchestrator=True)
    time.sleep(10)
    assert(control.finished())
    control.release()

    assert(not p.stderr.decode('ascii'))

    for i in range(n_db_nodes):
        fnames = []
        fnames.append("nodes-orchestrator_"+str(i)+"-6379.conf")
        fnames.append("orchestrator_"+str(i)+".err")
        fnames.append("orchestrator_"+str(i)+".out")
        for fname in fnames:
            if path.isfile(fname):
                remove(fname)

    if path.isdir(compile_dir):
        rmtree(compile_dir)
                
    if(environ["SSDB"]):
        environ.pop("SSDB")
    if(environ["SSNAME"]):
        environ.pop("SSNAME")
    if(environ["SSDATAIN"]):
        environ.pop("SSDATAIN")
