import os
import signal
import time
from threading import Thread

from smartsim import Experiment
from smartsim.settings import RunSettings


# def signal_handler(sig, frame):
#     print('You pressed Ctrl+C!')
def keyboard_interrupt(pid):
    """Interrupt main thread"""
    time.sleep(8)  # allow time for jobs to start before interrupting
    os.kill(pid, signal.SIGINT)


def test_interrupt_jobs(fileutils):
    # signal.signal(signal.SIGINT, signal_handler)
    test_dir = fileutils.make_test_dir()
    exp_name = "test_interrupt_jobs"
    exp = Experiment(exp_name, exp_path=test_dir)
    model = exp.create_model(
        "interrupt_model", path=test_dir, run_settings=RunSettings("sleep", "100")
    )
    ensemble = exp.create_ensemble(
        "interrupt_ensemble", replicas=2, run_settings=RunSettings("sleep", "100")
    )
    ensemble.set_path(test_dir)
    num_jobs = 1 + len(ensemble)
    try:
        pid = os.getpid()
        keyboard_interrupt_thread = Thread(
            name="sigint_thread", target=keyboard_interrupt, args=(pid,)
        )
        keyboard_interrupt_thread.start()
        exp.start(model, ensemble, block=True, kill_on_interrupt=True)
    except KeyboardInterrupt:
        time.sleep(2)  # allow time for jobs to be stopped
        active_jobs = exp._control._jobs.jobs
        active_db_jobs = exp._control._jobs.db_jobs
        completed_jobs = exp._control._jobs.completed
        assert len(active_jobs) + len(active_db_jobs) == 0
        assert len(completed_jobs) == num_jobs
