import time
from os import environ, getcwd, path

from smartsim import Experiment
from smartsim.utils.test.decorators import orchestrator_test_slurm

# --- Setup ---------------------------------------------------

# Path to test outputs
test_path = path.join(getcwd(),  "./orchestrator_test/")

# --- existing db files -----------------------------------------------

@orchestrator_test_slurm
def test_db_file_removal():
    """test that existing .conf, .out, and .err do not prevent
       launch of database.
    """
    db_test_alloc = environ["TEST_ALLOCATION_ID"]
    exp = Experiment("DB-File-Remove-Test")
    O3 = exp.create_orchestrator(path=test_path, port=6780, db_nodes=3, dpn=3, alloc=db_test_alloc)

    for dbnode in O3.entities:
        for port in dbnode.ports:
            conf_file = "/".join((dbnode.path, dbnode._get_db_conf_filename(port)))
            open(conf_file, 'w').close()
        out_file = dbnode.run_settings["out_file"]
        err_file = dbnode.run_settings["err_file"]
        open(err_file, 'w').close()
        open(out_file, 'w').close()

    exp.start(O3)
    time.sleep(5)
    statuses = exp.get_status(O3)
    assert("FAILED" not in statuses)
    exp.stop(O3)
