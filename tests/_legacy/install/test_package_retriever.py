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

import contextlib
import filecmp
import os
import pathlib
import random
import string
import tarfile
import zipfile

import pytest

from smartsim._core._install.utils import retrieve

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


@contextlib.contextmanager
def temp_cd(path):
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def make_test_file(test_file):
    data = "".join(random.choices(string.ascii_letters + string.digits, k=1024))
    with open(test_file, "w") as f:
        f.write(data)


def test_local_archive_zip(test_dir):
    with temp_cd(test_dir):
        test_file = "./test.data"
        make_test_file(test_file)

        zip_file = "./test.zip"
        with zipfile.ZipFile(zip_file, "w") as f:
            f.write(test_file)

        retrieve(zip_file, pathlib.Path("./output"))

        assert filecmp.cmp(
            test_file, pathlib.Path("./output") / "test.data", shallow=False
        )


def test_local_archive_tgz(test_dir):
    with temp_cd(test_dir):
        test_file = "./test.data"
        make_test_file(test_file)

        tgz_file = "./test.tgz"
        with tarfile.open(tgz_file, "w:gz") as f:
            f.add(test_file)

        retrieve(tgz_file, pathlib.Path("./output"))

        assert filecmp.cmp(
            test_file, pathlib.Path("./output") / "test.data", shallow=False
        )


def test_git(test_dir):
    retrieve(
        "https://github.com/CrayLabs/SmartSim.git",
        f"{test_dir}/smartsim_git",
        branch="master",
    )
    assert pathlib.Path(f"{test_dir}/smartsim_git").is_dir()


def test_https(test_dir):
    output_dir = pathlib.Path(test_dir) / "output"
    retrieve(
        "https://github.com/CrayLabs/SmartSim/archive/refs/tags/v0.5.0.zip", output_dir
    )
    assert output_dir.exists()
