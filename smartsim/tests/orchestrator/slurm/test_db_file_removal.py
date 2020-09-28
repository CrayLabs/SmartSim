import time
import pytest
import filecmp
from shutil import which
from os import path, getcwd, environ

from smartsim import Experiment
from smartsim.tests.decorators import orchestrator_test_slurm


# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./orchestrator_test/")
db_test_alloc = None

def test_setup_alloc():
    """Not a test, just used to ensure that at test time, the
       allocation is added to the controller. This has to be a
       test because otherwise it will run on pytest startup.
    """
    global db_test_alloc
    if not which("srun"):
        pytest.skip()
    assert("TEST_ALLOCATION_ID" in environ)
    db_test_alloc = environ["TEST_ALLOCATION_ID"]


# --- existing db files -----------------------------------------------


@orchestrator_test_slurm
def test_db_file_removal():
    """test that existing .conf, .out, and .err do not prevent
       launch of database.
    """
    global db_test_alloc
    exp = Experiment("DB-File-Remove-Test")
    exp.add_allocation(db_test_alloc)
    O3 = exp.create_orchestrator(path=test_path, db_nodes=3, dpn=3, alloc=db_test_alloc)

    for dbnode in O3.dbnodes:
        for port in dbnode.ports:
            conf_file = "/".join((dbnode.path, dbnode._get_dbnode_conf_fname(port)))
            open(conf_file, 'w').close()
        out_file = dbnode.run_settings["out_file"]
        err_file = dbnode.run_settings["err_file"]
        open(err_file, 'w').close()
        open(out_file, 'w').close()
    exp.start(orchestrator=O3)
    time.sleep(5)
    statuses = exp.get_status(O3)
    assert("FAILED" not in statuses)
    exp.stop(orchestrator=O3)
