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


import sys
import warnings

import pytest

from smartsim import Experiment, status

if sys.platform == "darwin":
    supported_dbs = ["tcp", "deprecated"]
else:
    supported_dbs = ["uds", "tcp", "deprecated"]


@pytest.mark.parametrize("db_type", supported_dbs)
def test_launch_colocated_model(fileutils, db_type):
    """Test the launch of a model with a colocated database and local launcher"""

    exp_name = "test-launch-colocated-model-with-restart"
    exp = Experiment(exp_name, launcher="local")

    # get test setup
    test_dir = fileutils.make_test_dir()
    sr_test_script = fileutils.get_test_conf_path("send_data_local_smartredis.py")

    # create colocated model
    colo_settings = exp.create_run_settings(exe=sys.executable, exe_args=sr_test_script)

    colo_model = exp.create_model("colocated_model", colo_settings)
    colo_model.set_path(test_dir)

    db_args = {
        "db_cpus": 1,
        "limit_app_cpus": False,
        "debug": True,
    }

    if db_type in ["tcp", "deprecated"]:
        colocate_fun = {
            "tcp": colo_model.colocate_db_tcp,
            "deprecated": colo_model.colocate_db,
        }
        with warnings.catch_warnings(record=True) as w:
            colocate_fun[db_type](port=6780, ifname="lo", **db_args)
            if db_type == "deprecated":
                assert len(w) == 1
                assert issubclass(w[-1].category, DeprecationWarning)
                assert "Please use `colocate_db_tcp` or `colocate_db_uds`" in str(
                    w[-1].message
                )
    elif db_type == "uds":
        colo_model.colocate_db_uds(**db_args)

    # assert model will launch with colocated db
    assert colo_model.colocated

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])

    # test restarting the colocated model

    exp.start(colo_model, block=True)
    statuses = exp.get_status(colo_model)
    assert all([stat == status.STATUS_COMPLETED for stat in statuses])
