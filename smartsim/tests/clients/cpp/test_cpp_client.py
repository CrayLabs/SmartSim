import pytest
import subprocess

from os import getcwd, listdir, path, environ, mkdir, remove

from glob import glob
from shutil import rmtree, which, copyfile

from smartsim import Generator, Controller, State
import time
from subprocess import Popen

def cpp_client_test_builder(cpp_target_name):
    """ This function is used to run each standalone 
    cpp test cases.  The only thing that varies between 
    cases is the name of the target to be built.
    """

    # Check for slurm
    if not which("mpiexec"):
        pytest.skip()
        
    test_path = path.dirname(path.abspath(__file__))
    compile_dir = test_path + "/compile"
    
    if path.isdir(compile_dir):
        rmtree(compile_dir)
    binary_name = compile_dir + '/' + cpp_target_name
    mkdir(compile_dir)
        
    p = subprocess.run(["cmake", "../"], cwd=compile_dir, capture_output=True)
    p = subprocess.run(["make", cpp_target_name], cwd=compile_dir, capture_output=True)
    
    if not path.isfile(binary_name):
        return False

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
    
    p = subprocess.run(["mpiexec","-n","1",binary_name, "10"], cwd=compile_dir, capture_output=True)
        
    control.stop(ensembles=[], nodes=[], stop_orchestrator=True)
    time.sleep(10)
    if not control.finished():
        return False
    control.release()

    if p.stderr.decode('ascii'):
        return False

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

    return True

def test_put_get_one_dimensional_array_cpp():
    """ This funtion tests putting a one dimensional
        array into the database and then gets it from
        the database and does a comparison.
    """

    assert(cpp_client_test_builder("client_tester_1D"))
    
def test_put_get_two_dimensional_array_cpp():
    """ This funtion tests putting a two dimensional
        array into the database and then gets it from
        the database and does a comparison.
    """

    assert(cpp_client_test_builder("client_tester_2D"))

def test_put_get_three_dimensional_array_cpp():
    """ This funtion tests putting a three dimensional
        array into the database and then gets it from
        the database and does a comparison.
    """

    assert(cpp_client_test_builder("client_tester_3D"))

