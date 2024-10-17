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

from smartsim.entity.application import Application
from smartsim.launchable.base_job import BaseJob
from smartsim.launchable.job import Job
from smartsim.launchable.job_group import JobGroup
from smartsim.settings.launch_settings import LaunchSettings

pytestmark = pytest.mark.group_a

app_1 = Application("app_1", "python")
app_2 = Application("app_2", "python")
app_3 = Application("app_3", "python")


class MockJob(BaseJob):
    def get_launch_steps(self):
        raise NotImplementedError


def test_invalid_job_name(wlmutils):
    job_1 = Job(app_1, LaunchSettings("slurm"))
    job_2 = Job(app_2, LaunchSettings("slurm"))
    with pytest.raises(ValueError):
        _ = JobGroup([job_1, job_2], name="name/not/allowed")


def test_create_JobGroup():
    job_1 = MockJob()
    job_group = JobGroup([job_1])
    assert len(job_group) == 1


def test_name_setter(wlmutils):
    job_1 = Job(app_1, LaunchSettings("slurm"))
    job_2 = Job(app_2, LaunchSettings("slurm"))
    job_group = JobGroup([job_1, job_2])
    job_group.name = "new_name"
    assert job_group.name == "new_name"


def test_getitem_JobGroup(wlmutils):
    job_1 = Job(app_1, LaunchSettings("slurm"))
    job_2 = Job(app_2, LaunchSettings("slurm"))
    job_group = JobGroup([job_1, job_2])
    get_value = job_group[0].entity.name
    assert get_value == job_1.entity.name


def test_setitem_JobGroup(wlmutils):
    job_1 = Job(app_1, LaunchSettings("slurm"))
    job_2 = Job(app_2, LaunchSettings("slurm"))
    job_group = JobGroup([job_1, job_2])
    job_3 = Job(app_3, LaunchSettings("slurm"))
    job_group[1] = job_3
    assert len(job_group) == 2
    get_value = job_group[1]
    assert get_value.entity.name == job_3.entity.name


def test_delitem_JobGroup():
    job_1 = MockJob()
    job_2 = MockJob()
    job_group = JobGroup([job_1, job_2])
    assert len(job_group) == 2
    del job_group[1]
    assert len(job_group) == 1


def test_len_JobGroup():
    job_1 = MockJob()
    job_2 = MockJob()
    job_group = JobGroup([job_1, job_2])
    assert len(job_group) == 2


def test_insert_JobGroup():
    job_1 = MockJob()
    job_2 = MockJob()
    job_group = JobGroup([job_1, job_2])
    job_3 = MockJob()
    job_group.insert(0, job_3)
    get_value = job_group[0]
    assert get_value == job_3
