# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os
import signal
import time
from threading import Thread

from smartsim import Experiment
from smartsim.settings import RunSettings


def keyboard_interrupt(pid):
    """Interrupt main thread"""
    time.sleep(8)  # allow time for jobs to start before interrupting
    os.kill(pid, signal.SIGINT)


def test_interrupt_blocked_jobs(test_dir):
    """
    Launches and polls a model and an ensemble with two more models.
    Once polling starts, the SIGINT signal is sent to the main thread,
    and consequently, all running jobs are killed.
    """

    exp_name = "test_interrupt_blocked_jobs"
    exp = Experiment(exp_name, exp_path=test_dir)
    model = exp.create_model(
        "interrupt_blocked_model",
        path=test_dir,
        run_settings=RunSettings("sleep", "100"),
    )
    ensemble = exp.create_ensemble(
        "interrupt_blocked_ensemble",
        replicas=2,
        run_settings=RunSettings("sleep", "100"),
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


def test_interrupt_multi_experiment_unblocked_jobs(test_dir):
    """
    Starts two Experiments, each having one model
    and an ensemble with two more models. Since
    blocking is False, the main thread sleeps until
    the SIGINT signal is sent, resulting in both
    Experiment's running jobs to be killed.
    """

    exp_names = ["test_interrupt_jobs_0", "test_interrupt_jobs_1"]
    experiments = [Experiment(exp_names[i], exp_path=test_dir) for i in range(2)]
    jobs_per_experiment = [0] * len(experiments)
    for i, experiment in enumerate(experiments):
        model = experiment.create_model(
            "interrupt_model_" + str(i),
            path=test_dir,
            run_settings=RunSettings("sleep", "100"),
        )
        ensemble = experiment.create_ensemble(
            "interrupt_ensemble_" + str(i),
            replicas=2,
            run_settings=RunSettings("sleep", "100"),
        )
        ensemble.set_path(test_dir)
        jobs_per_experiment[i] = 1 + len(ensemble)
    try:
        pid = os.getpid()
        keyboard_interrupt_thread = Thread(
            name="sigint_thread", target=keyboard_interrupt, args=(pid,)
        )
        keyboard_interrupt_thread.start()
        for experiment in experiments:
            experiment.start(model, ensemble, block=False, kill_on_interrupt=True)
        time.sleep(9)  # since jobs aren't blocked, wait for SIGINT
    except KeyboardInterrupt:
        time.sleep(2)  # allow time for jobs to be stopped
        for i, experiment in enumerate(experiments):
            active_jobs = experiment._control._jobs.jobs
            active_db_jobs = experiment._control._jobs.db_jobs
            completed_jobs = experiment._control._jobs.completed
            assert len(active_jobs) + len(active_db_jobs) == 0
            assert len(completed_jobs) == jobs_per_experiment[i]
