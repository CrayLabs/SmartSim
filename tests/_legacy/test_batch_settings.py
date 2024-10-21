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

from smartsim.settings import BsubBatchSettings, QsubBatchSettings, SbatchSettings
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


def test_existing_batch_args_mutation():
    """
    Ensure that if the batch_args dict is changed, any previously
    created batch settings don't reflect the change due to pass-by-ref
    """
    batch_args = {"k1": "v1", "k2": "v2"}
    orig_bargs = {"k1": "v1", "k2": "v2"}
    bsub = create_batch_settings(
        "lsf",
        nodes=1,
        time="10:00:00",
        account="myproject",  # test that account is set
        queue="default",
        batch_args=batch_args,
    )

    # verify initial expectations
    assert "k1" in bsub.batch_args
    assert "k2" in bsub.batch_args

    # modify the batch_args dict
    batch_args["k1"] = f'not-{batch_args["k1"]}'

    # verify that the batch_settings do not reflect the change
    assert bsub.batch_args["k1"] == orig_bargs["k1"]
    assert bsub.batch_args["k1"] != batch_args["k1"]


def test_direct_set_batch_args_mutation():
    """
    Ensure that if the batch_args dict is set directly, any previously
    created batch settings don't reflect the change due to pass-by-ref
    """
    batch_args = {"k1": "v1", "k2": "v2"}
    orig_bargs = {"k1": "v1", "k2": "v2"}
    bsub = create_batch_settings(
        "lsf",
        nodes=1,
        time="10:00:00",
        account="myproject",  # test that account is set
        queue="default",
    )
    bsub.batch_args = batch_args

    # verify initial expectations
    assert "k1" in bsub.batch_args
    assert "k2" in bsub.batch_args

    # modify the batch_args dict
    batch_args["k1"] = f'not-{batch_args["k1"]}'

    # verify that the batch_settings do not reflect the change
    assert bsub.batch_args["k1"] == orig_bargs["k1"]
    assert bsub.batch_args["k1"] != batch_args["k1"]


@pytest.mark.parametrize(
    "batch_args",
    [
        pytest.param({"core_isolation": None}, id="null batch arg"),
        pytest.param({"core_isolation": "abc"}, id="valued batch arg"),
        pytest.param({"core_isolation": None, "xyz": "pqr"}, id="multi batch arg"),
    ],
)
def test_stringification(batch_args):
    """Ensure stringification includes expected fields"""
    bsub_cmd = "bsub"
    num_nodes = 1

    bsub = create_batch_settings(
        "lsf",
        nodes=num_nodes,
        time="10:00:00",
        account="myproject",  # test that account is set
        queue="default",
        batch_args=batch_args,
    )

    repl = str(bsub).replace("\t", " ").replace("\n", "")

    assert repl.startswith(f"Batch Command: {bsub_cmd}")
    assert repl.index("Batch arguments: ") > -1
    for key, val in batch_args.items():
        assert repl.index(f"{key} = {val}") > -1
    assert repl.index(f"nnodes = {num_nodes}") > -1


def test_preamble():
    """Ensure that preable lines are appended and do not overwrite"""
    bsub = create_batch_settings(
        "lsf",
        nodes=1,
        time="10:00:00",
        account="myproject",  # test that account is set
        queue="default",
        batch_args={},
    )

    bsub.add_preamble(["single line"])
    assert len(list(bsub.preamble)) == 1

    bsub.add_preamble(["another line"])
    assert len(list(bsub.preamble)) == 2

    bsub.add_preamble(["first line", "last line"])
    assert len(list(bsub.preamble)) == 4
