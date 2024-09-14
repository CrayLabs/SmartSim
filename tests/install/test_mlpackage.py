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

import os
import pathlib
from unittest.mock import MagicMock

import pytest

from smartsim._core._install.mlpackages import (
    MLPackage,
    MLPackageCollection,
    RAIPatch,
    load_platform_configs,
)
from smartsim._core._install.platform import Platform

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

mock_platform = MagicMock(spec=Platform)


@pytest.fixture
def mock_ml_packages():
    foo = MagicMock(spec=MLPackage)
    foo.name = "foo"
    bar = MagicMock(spec=MLPackage)
    bar.name = "bar"
    yield [foo, bar]


@pytest.mark.parametrize(
    "patch",
    [MagicMock(spec=RAIPatch), [MagicMock(spec=RAIPatch) for i in range(3)], ()],
    ids=["one patch", "multiple patches", "no patch"],
)
def test_mlpackage_constructor(patch):
    MLPackage(
        "foo",
        "0.0.0",
        "https://nothing.com",
        ["bar==0.1", "baz==0.2"],
        pathlib.Path("/nothing/fake"),
        patch,
    )


def test_mlpackage_collection_constructor(mock_ml_packages):
    MLPackageCollection(mock_platform, mock_ml_packages)


def test_mlpackage_collection_mutable_mapping_methods(mock_ml_packages):
    ml_packages = MLPackageCollection(mock_platform, mock_ml_packages)
    for val in ml_packages._ml_packages.values():
        val.version = "0.0.0"
    assert ml_packages._ml_packages == ml_packages

    # Test iter
    package_names = [pkg.name for pkg in mock_ml_packages]
    assert [name for name in ml_packages] == package_names

    # Test get item
    for pkg in mock_ml_packages:
        assert ml_packages[pkg.name] is pkg

    # Test len
    assert len(ml_packages) == len(mock_ml_packages)

    # Test delitem
    key = next(iter(mock_ml_packages)).name
    del ml_packages[key]
    with pytest.raises(KeyError):
        ml_packages[key]
    assert len(ml_packages) == (len(mock_ml_packages) - 1)

    # Test setitem
    with pytest.raises(TypeError):
        ml_packages["baz"] = MagicMock(spec=MLPackage)

    # Test contains
    name, package = next(iter(ml_packages.items()))
    assert name in ml_packages

    # Test str
    assert "Package" in str(ml_packages)
    assert "Version" in str(ml_packages)
    assert package.version in str(ml_packages)
    assert name in str(ml_packages)


def test_load_configs_raises_when_dir_dne(test_dir):
    dne_dir = pathlib.Path(test_dir, "dne")
    dir_str = os.fspath(dne_dir)
    with pytest.raises(
        FileNotFoundError,
        match=f"Platform configuration directory `{dir_str}` does not exist",
    ):
        load_platform_configs(dne_dir)
