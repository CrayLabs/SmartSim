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

import pytest

from smartsim.error import SSUnsupportedError
from smartsim.settings import PalsMpiexecSettings

default_exe = sys.executable
default_kwargs = {"fail_if_missing_exec": False}

# Uncomment when
# @pytest.mark.parametrize(
#    "function_name",[
#        'set_task_map',
#        'set_cpus_per_task',
#        'set_quiet_launch',
#        'set_walltime'
#    ]
# )
# def test_unsupported_methods(function_name):
#    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
#    func = getattr(settings, function_name)
#    with pytest.raises(SSUnsupportedError):
#        func(None)


def test_cpu_binding_type():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_cpu_binding_type("numa")
    assert settings.format_run_args() == ["--cpu-bind", "numa"]


def test_tasks_per_node():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_tasks_per_node(48)
    assert settings.format_run_args() == ["--ppn", "48"]


def test_broadcast():
    settings = PalsMpiexecSettings(default_exe, **default_kwargs)
    settings.set_broadcast()
    assert settings.format_run_args() == ["--transfer"]


def test_format_env_vars():
    example_env_vars = {"FOO_VERSION": "3.14", "PATH": None, "LD_LIBRARY_PATH": None}
    settings = PalsMpiexecSettings(
        default_exe, **default_kwargs, env_vars=example_env_vars
    )
    formatted = " ".join(settings.format_env_vars())
    expected = "--env FOO_VERSION=3.14 --envlist PATH,LD_LIBRARY_PATH"
    assert formatted == expected
