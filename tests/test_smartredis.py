# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
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


import pytest

from smartsim import Experiment
from smartsim.status import SmartSimStatus
from smartsim._core.utils import installed_redisai_backends
from smartsim.database import Orchestrator
from smartsim.entity import Ensemble, Model

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b


"""Test smartredis integration for ensembles. Two copies of the same
   program will be executed concurrently, and name collisions
   will be avoided through smartredis prefixing:
   smartredis will prefix each instance's tensors with a prefix
   set through environment variables by SmartSim.
"""

shouldrun = True
try:
    import torch
except ImportError:
    shouldrun = False

torch_available = "torch" in installed_redisai_backends()

shouldrun &= torch_available

pytestmark = pytest.mark.skipif(
    not shouldrun,
    reason="requires PyTorch, SmartRedis, and RedisAI's Torch backend",
)


def test_exchange(fileutils, test_dir, wlmutils):
    """Run two processes, each process puts a tensor on
    the DB, then accesses the other process's tensor.
    Finally, the tensor is used to run a model.
    """

    exp = Experiment(
        "smartredis_ensemble_exchange", exp_path=test_dir, launcher="local"
    )

    # create and start a database
    orc = Orchestrator(port=wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc, block=False)

    rs = exp.create_run_settings("python", "producer.py --exchange")
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

    exp.generate(ensemble)

    # start the models
    exp.start(ensemble, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    try:
        assert all([stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses])
    finally:
        # stop the orchestrator
        exp.stop(orc)


def test_consumer(fileutils, test_dir, wlmutils):
    """Run three processes, each one of the first two processes
    puts a tensor on the DB; the third process accesses the
    tensors put by the two producers.
    Finally, the tensor is used to run a model by each producer
    and the consumer accesses the two results.
    """

    exp = Experiment(
        "smartredis_ensemble_consumer", exp_path=test_dir, launcher="local"
    )

    # create and start a database
    orc = Orchestrator(port=wlmutils.get_test_port())
    exp.generate(orc)
    exp.start(orc, block=False)

    rs_prod = exp.create_run_settings("python", "producer.py")
    rs_consumer = exp.create_run_settings("python", "consumer.py")
    params = {"mult": [1, -10]}
    ensemble = Ensemble(
        name="producer", params=params, run_settings=rs_prod, perm_strat="step"
    )

    consumer = Model(
        "consumer", params={}, path=ensemble.path, run_settings=rs_consumer
    )
    ensemble.add_model(consumer)

    ensemble.register_incoming_entity(ensemble["producer_0"])
    ensemble.register_incoming_entity(ensemble["producer_1"])

    config = fileutils.get_test_conf_path("smartredis")
    ensemble.attach_generator_files(to_copy=[config])

    exp.generate(ensemble)

    # start the models
    exp.start(ensemble, summary=False)

    # get and confirm statuses
    statuses = exp.get_status(ensemble)
    try:
        assert all([stat == SmartSimStatus.STATUS_COMPLETED for stat in statuses])
    finally:
        # stop the orchestrator
        exp.stop(orc)
