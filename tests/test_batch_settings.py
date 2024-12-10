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

from smartsim.settings import QsubBatchSettings, SbatchSettings
from smartsim.settings.settings import create_batch_settings

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_create_pbs_batch():
    pbs_batch = create_batch_settings(
        "pbs", nodes=1, time="10:00:00", queue="default", account="myproject", ncpus=10
    )  # test that kwargs make it to class init
    args = pbs_batch.format_batch_args()
    assert isinstance(pbs_batch, QsubBatchSettings)
    assert args == [
        "-l nodes=1:ncpus=10",
        "-l walltime=10:00:00",
        "-q default",
        "-A myproject",
    ]


def test_create_sbatch():
    batch_args = {"exclusive": None, "oversubscribe": None}
    slurm_batch = create_batch_settings(
        "slurm",
        nodes=1,
        time="10:00:00",
        queue="default",  # actually sets partition
        account="myproject",
        batch_args=batch_args,
        ncpus=10,
    )  # test that kwargs from
    # pbs doesn't effect slurm (im thinking this will be common)

    assert isinstance(slurm_batch, SbatchSettings)
    assert slurm_batch.batch_args["partition"] == "default"
    args = slurm_batch.format_batch_args()
    expected_args = [
        "--exclusive",
        "--oversubscribe",
        "--nodes=1",
        "--time=10:00:00",
        "--partition=default",
        "--account=myproject",
    ]
    assert all(arg in expected_args for arg in args)
    assert len(expected_args) == len(args)
