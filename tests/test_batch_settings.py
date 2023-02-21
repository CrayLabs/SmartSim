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


from smartsim.settings import BsubBatchSettings, QsubBatchSettings, SbatchSettings
from smartsim.settings.settings import create_batch_settings


def test_create_pbs_batch():
    pbs_batch = create_batch_settings(
        "pbs", nodes=1, time="10:00:00", queue="default", account="myproject", ncpus=10
    )  # test that kwargs make it to class init
    args = pbs_batch.format_batch_args()
    assert isinstance(pbs_batch, QsubBatchSettings)
    assert args == [
        "-l select=1:ncpus=10",
        "-l place=scatter",
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
    assert args == [
        "--exclusive",
        "--oversubscribe",
        "--nodes=1",
        "--time=10:00:00",
        "--partition=default",
        "--account=myproject",
    ]


def test_create_bsub():
    batch_args = {"core_isolation": None}
    bsub = create_batch_settings(
        "lsf",
        nodes=1,
        time="10:00:00",
        account="myproject",  # test that account is set
        queue="default",
        batch_args=batch_args,
    )
    assert isinstance(bsub, BsubBatchSettings)
    args = bsub.format_batch_args()
    assert args == ["-core_isolation", "-nnodes 1", "-q default"]
