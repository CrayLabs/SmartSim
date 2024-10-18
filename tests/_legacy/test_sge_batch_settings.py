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

import os.path as osp

import pytest

from smartsim import Experiment
from smartsim._core.launcher.sge.sge_parser import parse_qstat_jobid_xml
from smartsim.error import SSConfigError
from smartsim.settings import SgeQsubBatchSettings
from smartsim.settings.mpiSettings import _BaseMPISettings

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b

qstat_example = """<?xml version='1.0'?>
<job_info  xmlns:xsd="http://arc.liv.ac.uk/repos/darcs/sge/source/dist/util/resources/schemas/qstat/qstat.xsd">
  <queue_info>
    <job_list state="running">
      <JB_job_number>1387693</JB_job_number>
      <JAT_prio>3.50000</JAT_prio>
      <JB_name>test_1</JB_name>
      <JB_owner>user1</JB_owner>
      <state>r</state>
      <JAT_start_time>2024-06-06T04:04:21</JAT_start_time>
      <queue_name>example_node1</queue_name>
      <slots>1600</slots>
    </job_list>
  </queue_info>
  <job_info>
    <job_list state="pending">
      <JB_job_number>1387695</JB_job_number>
      <JAT_prio>3.48917</JAT_prio>
      <JB_name>test_2</JB_name>
      <JB_owner>user1</JB_owner>
      <state>qw</state>
      <JB_submission_time>2024-05-20T16:47:46</JB_submission_time>
      <queue_name></queue_name>
      <slots>1600</slots>
    </job_list>
  </job_info>
</job_info>
"""


@pytest.mark.parametrize("pe_type", ["mpi", "smp"])
def test_pe_config(pe_type):
    settings = SgeQsubBatchSettings(ncpus=8, pe_type=pe_type)
    assert settings._create_resource_list() == [f"-pe {pe_type} 8"]


def test_walltime():
    settings = SgeQsubBatchSettings(time="01:00:00")
    assert settings._create_resource_list() == [
        f"-l h_rt=01:00:00",
    ]


def test_ngpus():
    settings = SgeQsubBatchSettings()
    settings.set_ngpus(1)
    assert settings._create_resource_list() == [f"-l gpu=1"]


def test_account():
    settings = SgeQsubBatchSettings(account="foo")
    assert settings.format_batch_args() == ["-A foo"]


def test_project():
    settings = SgeQsubBatchSettings()
    settings.set_project("foo")
    assert settings.format_batch_args() == ["-P foo"]


def test_update_context_variables():
    settings = SgeQsubBatchSettings()
    settings.update_context_variables("ac", "foo")
    settings.update_context_variables("sc", "foo", "bar")
    settings.update_context_variables("dc", "foo")
    assert settings._create_resource_list() == ["-ac foo", "-sc foo=bar", "-dc foo"]


def test_invalid_dc_and_value_update_context_variables():
    settings = SgeQsubBatchSettings()
    with pytest.raises(SSConfigError):
        settings.update_context_variables("dc", "foo", "bar")


@pytest.mark.parametrize("enable", [True, False])
def test_set_hyperthreading(enable):
    settings = SgeQsubBatchSettings()
    settings.set_hyperthreading(enable)
    assert settings._create_resource_list() == [f"-l threads={int(enable)}"]


def test_default_set_hyperthreading():
    settings = SgeQsubBatchSettings()
    settings.set_hyperthreading()
    assert settings._create_resource_list() == ["-l threads=1"]


def test_resources_is_a_copy():
    settings = SgeQsubBatchSettings()
    resources = settings.resources
    assert resources is not settings._resources


def test_resources_not_set_on_error():
    settings = SgeQsubBatchSettings()
    unaltered_resources = settings.resources
    with pytest.raises(TypeError):
        settings.resources = {"meep": Exception}

    assert unaltered_resources == settings.resources


def test_qstat_jobid_xml():
    assert parse_qstat_jobid_xml(qstat_example, "1387693") == "r"
    assert parse_qstat_jobid_xml(qstat_example, "1387695") == "qw"
    assert parse_qstat_jobid_xml(qstat_example, "9999999") is None


def test_sge_launcher_defaults(monkeypatch, fileutils):

    stub_path = osp.join("mpi_impl_stubs", "openmpi4")
    stub_path = fileutils.get_test_dir_path(stub_path)
    monkeypatch.setenv("PATH", stub_path, prepend=":")
    exp = Experiment("test_sge_run_settings", launcher="sge")

    bs = exp.create_batch_settings()
    assert isinstance(bs, SgeQsubBatchSettings)
    rs = exp.create_run_settings("echo")
    assert isinstance(rs, _BaseMPISettings)
