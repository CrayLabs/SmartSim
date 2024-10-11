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

import itertools

import pytest

from smartsim._core.utils.helpers import expand_exe_path
from smartsim._core.utils.launcher import create_job_id
from smartsim.entity.application import Application
from smartsim.launchable.job import Job, Record
from smartsim.settings.launch_settings import LaunchSettings

pytestmark = pytest.mark.group_a


def test_cannot_mutate_record_job():
    app = Application("my-test-app", "echo", ["spam", "eggs"])
    settings = LaunchSettings("local")
    job = Job(app, settings)

    id_ = create_job_id()
    record = Record(id_, job)
    assert record.launched_id == id_
    assert all(
        x is not y for x, y in itertools.combinations([job, record.job, record._job], 2)
    )

    app.name = "Modified orignal app name"
    job.name = "Modified original job name"
    record.job.name = "Modified reference to job off record name"
    record.job.name = "Modified reference to app to job off record name"
    assert record.job.name == record.job.entity.name == "my-test-app"

    record.job.entity.exe = "sleep"
    app.exe_args = ["120"]
    assert [record.job.entity.exe] + record.job.entity.exe_args == [
        expand_exe_path("echo"),
        "spam",
        "eggs",
    ]
