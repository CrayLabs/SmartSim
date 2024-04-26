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

import io
import json
import pathlib
import random
import string
import tarfile
import textwrap
import typing as t
from collections import namedtuple

import pytest
from github import Github
from github.GitReleaseAsset import GitReleaseAsset
from github.Requester import Requester

import smartsim
from conftest import FileUtils
from smartsim._core.entrypoints.dragon_install import (
    check_for_utility,
    cleanup,
    expand_archive,
    filter_assets,
    install_dragon,
    install_package,
    is_crayex_platform,
    retrieve_asset,
    retrieve_asset_info,
)
from smartsim.error.errors import SmartSimCLIActionCancelled

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


mock_archive_name = "dragon-0.8-py3.9.4.1-CRAYEX-ac132fe95.tar.gz"
_git_attr = namedtuple("_git_attr", "value")


@pytest.fixture
def asset_list() -> t.Dict[str, GitReleaseAsset]:
    requester = Requester(
        auth=None,
        base_url="https://github.com",
        user_agent="mozilla",
        per_page=10,
        verify=False,
        timeout=1,
        retry=1,
        pool_size=1,
    )
    headers = {"mock-header": "mock-value"}
    attributes = {"mock-attr": "mock-attr-value"}
    completed = True

    assets: t.Dict[str, GitReleaseAsset] = {}

    for python_version in ["py3.9", "py3.10", "py3.11"]:
        for dragon_version in ["dragon-0.8", "dragon-0.9", "dragon-0.10"]:
            for platform in ["", "CRAYEX-"]:

                asset1 = GitReleaseAsset(requester, headers, attributes, completed)

                asset1_name = mock_archive_name
                asset1_name = asset1_name.replace("py3.9", python_version)
                asset1_name = asset1_name.replace("dragon-0.8", dragon_version)
                asset1_name = asset1_name.replace("CRAYEX-", platform)

                setattr(
                    asset1,
                    "_browser_download_url",
                    _git_attr(value=f"http://foo/{asset1_name}"),
                )
                setattr(asset1, "_name", _git_attr(value=asset1_name))
                assets[asset1_name] = asset1

    return assets


def test_cleanup_no_op(test_dir: str) -> None:
    """Ensure that the cleanup method doesn't bomb when called with
    missing archive path and extraction directory; simulate a failed
    download"""
    archive_path = pathlib.Path(test_dir) / mock_archive_name
    extract_path = pathlib.Path(test_dir) / mock_archive_name.replace(".tar.gz", "")

    # confirm assets do not exist
    assert not archive_path.exists()
    assert not extract_path.exists()

    # call cleanup. any exceptions should break test...
    cleanup(archive_path, extract_path)


def test_cleanup_empty_extraction_directory(test_dir: str) -> None:
    """Ensure that the cleanup method works when the extraction directory
    is empty"""
    archive_path = pathlib.Path(test_dir) / mock_archive_name
    extract_path = pathlib.Path(test_dir) / mock_archive_name.replace(".tar.gz", "")
    extract_path.mkdir()

    # verify archive doesn't exist & folder does
    assert not archive_path.exists()
    assert extract_path.exists()

    cleanup(archive_path, extract_path)

    # verify folder is gone after cleanup
    assert not archive_path.exists()
    assert not extract_path.exists()


def test_cleanup_nonempty_extraction_directory(test_dir: str) -> None:
    """Ensure that the cleanup method works when the extraction directory
    is NOT empty"""
    archive_path = pathlib.Path(test_dir) / mock_archive_name
    extract_path = pathlib.Path(test_dir) / mock_archive_name.replace(".tar.gz", "")
    extract_path.mkdir()

    num_files = 10
    for i in range(num_files):
        content = extract_path / f"{i:04}.txt"
        content.write_text(f"i am file {i}\n")

    files = list(extract_path.rglob("*.txt"))

    assert len(files) == num_files

    cleanup(archive_path, extract_path)

    # verify folder is gone after cleanup
    assert not archive_path.exists()
    assert not extract_path.exists()


def test_cleanup_no_extract_path(test_dir: str) -> None:
    """Ensure that the cleanup method doesn't bomb when called with
    missing extraction directory; simulate failed extract"""
    archive_path = pathlib.Path(test_dir) / mock_archive_name
    extract_path = pathlib.Path(test_dir) / mock_archive_name.replace(".tar.gz", "")

    # create an archive to clean up
    with tarfile.TarFile.open(archive_path, mode="w:gz") as tar:
        tar.add(__file__)  # add current file to avoid empty tar

    # verify archive exists before cleanup
    assert archive_path.exists()

    cleanup(archive_path, extract_path)

    # verify archive is gone after cleanup
    assert not archive_path.exists()
    assert not extract_path.exists()


def test_cleanup_no_archive(test_dir: str) -> None:
    """Ensure that the cleanup method doesn't bomb when called with
    missing archive"""
    archive_path = pathlib.Path(test_dir) / mock_archive_name
    extract_path = pathlib.Path(test_dir) / mock_archive_name.replace(".tar.gz", "")

    extract_path.mkdir()

    # verify archive exists before cleanup
    assert extract_path.exists()

    cleanup(archive_path, extract_path)

    # verify archive is gone after cleanup
    assert not archive_path.exists()
    assert not extract_path.exists()


def test_expand_archive(test_dir: str) -> None:
    """Verify archive is expanded into expected location w/correct content"""
    archive_path = pathlib.Path(test_dir) / mock_archive_name
    exp_path = pathlib.Path(test_dir) / mock_archive_name.replace(".tar.gz", "")
    num_files = 10

    # create an archive to clean up
    with tarfile.TarFile.open(archive_path, mode="w:gz") as tar:
        for i in range(num_files):
            content = pathlib.Path(test_dir) / f"{i:04}.txt"
            content.write_text(f"i am file {i}\n")
            tar.add(content)

    extract_path = expand_archive(archive_path)

    files = list(extract_path.rglob("*.txt"))

    assert len(files) == num_files
    assert extract_path == exp_path


def test_expand_archive_path_path(test_dir: str) -> None:
    """Verify the expand method responds to a bad path with a ValueError"""
    archive_path = pathlib.Path(test_dir) / mock_archive_name

    with pytest.raises(ValueError) as ex:
        expand_archive(archive_path)

    assert str(archive_path) in str(ex.value.args[0])


def test_retrieve_cached(test_dir: str) -> None:
    """Verify that a previously retrieved asset archive is re-used"""
    working_dir = pathlib.Path(test_dir)
    archive_path = working_dir / mock_archive_name
    num_files = 10

    # create an archive to simulate re-use
    with tarfile.TarFile.open(archive_path, mode="w:gz") as tar:
        for i in range(num_files):
            content = working_dir / f"{i:04}.txt"
            content.write_text(f"i am file {i}\n")
            tar.add(content)

    ts1 = archive_path.stat().st_ctime

    requester = Requester(
        auth=None,
        base_url="https://github.com",
        user_agent="mozilla",
        per_page=10,
        verify=False,
        timeout=1,
        retry=1,
        pool_size=1,
    )
    headers = {"mock-header": "mock-value"}
    attributes = {"mock-attr": "mock-attr-value"}
    completed = True

    asset = GitReleaseAsset(requester, headers, attributes, completed)

    # ensure mocked asset has values that we use...
    setattr(asset, "_browser_download_url", _git_attr(value="http://foo"))
    setattr(asset, "_name", _git_attr(value=mock_archive_name))

    asset_path = retrieve_asset(working_dir, asset)
    ts2 = asset_path.stat().st_ctime

    assert (
        asset_path == archive_path
    )  # show that the expected path matches the output path
    assert ts1 == ts2  # show that the file wasn't changed...


@pytest.mark.parametrize(
    "dragon_pin,pyv,is_found,is_crayex",
    [
        pytest.param("dragon-0.8", "py3.8", False, False, id="0.8,python 3.8"),
        pytest.param("dragon-0.8", "py3.9", True, False, id="0.8,python 3.9"),
        pytest.param("dragon-0.8", "py3.10", True, False, id="0.8,python 3.10"),
        pytest.param("dragon-0.8", "py3.11", True, False, id="0.8,python 3.11"),
        pytest.param("dragon-0.8", "py3.12", False, False, id="0.8,python 3.12"),
        pytest.param("dragon-0.8", "py3.8", False, True, id="0.8,python 3.8,CrayEX"),
        pytest.param("dragon-0.8", "py3.9", True, True, id="0.8,python 3.9,CrayEX"),
        pytest.param("dragon-0.8", "py3.10", True, True, id="0.8,python 3.10,CrayEX"),
        pytest.param("dragon-0.8", "py3.11", True, True, id="0.8,python 3.11,CrayEX"),
        pytest.param("dragon-0.8", "py3.12", False, True, id="0.8,python 3.12,CrayEX"),
        pytest.param("dragon-0.9", "py3.8", False, False, id="0.9,python 3.8"),
        pytest.param("dragon-0.9", "py3.9", True, False, id="0.9,python 3.9"),
        pytest.param("dragon-0.9", "py3.10", True, False, id="0.9,python 3.10"),
        pytest.param("dragon-0.9", "py3.11", True, False, id="0.9,python 3.11"),
        pytest.param("dragon-0.9", "py3.12", False, False, id="0.9,python 3.12"),
        pytest.param("dragon-0.9", "py3.8", False, True, id="0.9,python 3.8,CrayEX"),
        pytest.param("dragon-0.9", "py3.9", True, True, id="0.9,python 3.9,CrayEX"),
        pytest.param("dragon-0.9", "py3.10", True, True, id="0.9,python 3.10,CrayEX"),
        pytest.param("dragon-0.9", "py3.11", True, True, id="0.9,python 3.11,CrayEX"),
        pytest.param("dragon-0.9", "py3.12", False, True, id="0.9,python 3.12,CrayEX"),
        # add a couple variants for a dragon version that isn't in the asset list
        pytest.param("dragon-0.7", "py3.9", False, False, id="0.7,python 3.9"),
        pytest.param("dragon-0.7", "py3.9", False, True, id="0.7,python 3.9,CrayEX"),
    ],
)
def test_filter_assets(
    asset_list: t.Dict[str, GitReleaseAsset],
    monkeypatch: pytest.MonkeyPatch,
    dragon_pin: str,
    pyv: str,
    is_found: bool,
    is_crayex: bool,
) -> None:
    """Verify that an asset list is filtered correctly based on the python
    version, platform (e.g. CrayEX, !CrayEx), and target dragon pin"""

    with monkeypatch.context() as ctx:
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "python_version",
            lambda: pyv,
        )
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "is_crayex_platform",
            lambda: is_crayex,
        )
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "dragon_pin",
            lambda: dragon_pin,
        )
        chosen_asset = filter_assets(asset_list)

        if is_found:
            assert chosen_asset
            assert pyv in chosen_asset.name
            assert dragon_pin in chosen_asset.name

            if is_crayex:
                assert "crayex" in chosen_asset.name.lower()
        else:
            assert not chosen_asset


@pytest.mark.parametrize(
    "dragon_pin,pyv,is_found,is_crayex",
    [
        pytest.param("dragon-0.8", "py3.8", False, False, id="0.8,python 3.8"),
        pytest.param("dragon-0.8", "py3.9", True, False, id="0.8,python 3.9"),
        pytest.param("dragon-0.8", "py3.10", True, False, id="0.8,python 3.10"),
        pytest.param("dragon-0.8", "py3.11", True, False, id="0.8,python 3.11"),
        pytest.param("dragon-0.8", "py3.12", False, False, id="0.8,python 3.12"),
        pytest.param("dragon-0.8", "py3.8", False, True, id="0.8,python 3.8,CrayEX"),
        pytest.param("dragon-0.8", "py3.9", True, True, id="0.8,python 3.9,CrayEX"),
        pytest.param("dragon-0.8", "py3.10", True, True, id="0.8,python 3.10,CrayEX"),
        pytest.param("dragon-0.8", "py3.11", True, True, id="0.8,python 3.11,CrayEX"),
        pytest.param("dragon-0.8", "py3.12", False, True, id="0.8,python 3.12,CrayEX"),
        pytest.param("dragon-0.9", "py3.8", False, False, id="0.9,python 3.8"),
        pytest.param("dragon-0.9", "py3.9", True, False, id="0.9,python 3.9"),
        pytest.param("dragon-0.9", "py3.10", True, False, id="0.9,python 3.10"),
        pytest.param("dragon-0.9", "py3.11", True, False, id="0.9,python 3.11"),
        pytest.param("dragon-0.9", "py3.12", False, False, id="0.9,python 3.12"),
        pytest.param("dragon-0.9", "py3.8", False, True, id="0.9,python 3.8,CrayEX"),
        pytest.param("dragon-0.9", "py3.9", True, True, id="0.9,python 3.9,CrayEX"),
        pytest.param("dragon-0.9", "py3.10", True, True, id="0.9,python 3.10,CrayEX"),
        pytest.param("dragon-0.9", "py3.11", True, True, id="0.9,python 3.11,CrayEX"),
        pytest.param("dragon-0.9", "py3.12", False, True, id="0.9,python 3.12,CrayEX"),
        # add a couple variants for a dragon version that isn't in the asset list
        pytest.param("dragon-0.7", "py3.9", False, False, id="0.7,python 3.9"),
        pytest.param("dragon-0.7", "py3.9", False, True, id="0.7,python 3.9,CrayEX"),
    ],
)
def test_retrieve_asset_info(
    asset_list: t.Dict[str, GitReleaseAsset],
    monkeypatch: pytest.MonkeyPatch,
    dragon_pin: str,
    pyv: str,
    is_found: bool,
    is_crayex: bool,
) -> None:
    """Verify that an asset list is filtered correctly based on the python
    version, platform (e.g. CrayEX, !CrayEx), and target dragon pin"""

    with monkeypatch.context() as ctx:
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "python_version",
            lambda: pyv,
        )
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "is_crayex_platform",
            lambda: is_crayex,
        )
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "dragon_pin",
            lambda: dragon_pin,
        )
        # avoid hitting github API
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "_get_release_assets",
            lambda: asset_list,
        )

        if is_found:
            chosen_asset = retrieve_asset_info()

            assert chosen_asset
            assert pyv in chosen_asset.name
            assert dragon_pin in chosen_asset.name

            if is_crayex:
                assert "crayex" in chosen_asset.name.lower()
        else:
            with pytest.raises(SmartSimCLIActionCancelled):
                retrieve_asset_info()


def test_check_for_utility_missing(test_dir: str) -> None:
    """Ensure that looking for a missing utility doesn't raise an exception"""
    ld_config = pathlib.Path(test_dir) / "ldconfig"

    utility = check_for_utility(ld_config)

    assert not utility


def test_check_for_utility_exists(test_dir: str) -> None:
    """Ensure that looking for an existing utility returns a non-empty path"""
    utility = check_for_utility("ls")

    assert utility


def test_is_crayex_missing_ldconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the cray ex platform check doesn't fail when ldconfig isn't
    available for use"""

    def mock_util_check(util: str) -> bool:
        if util == "ldconfig":
            return False
        return True

    with monkeypatch.context() as ctx:
        # mock utility existence
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "check_for_utility",
            mock_util_check,
        )

        is_cray = is_crayex_platform()
        assert not is_cray


def test_is_crayex_missing_fi_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the cray ex platform check doesn't fail when fi_info isn't
    available for use"""

    def mock_util_check(util: str) -> bool:
        if util == "fi_info":
            return False
        return True

    with monkeypatch.context() as ctx:
        # mock utility existence
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "check_for_utility",
            mock_util_check,
        )

        is_cray = is_crayex_platform()
        assert not is_cray


@pytest.mark.parametrize(
    "is_cray",
    [
        pytest.param(True, id="CrayEX"),
        pytest.param(False, id="Non-CrayEX"),
    ],
)
def test_is_cray_ex(monkeypatch: pytest.MonkeyPatch, is_cray: bool) -> None:
    """Test that cray ex platform check result is returned as expected"""

    def mock_util_check(util: str) -> bool:
        # mock that we have the necessary tools
        return True

    with monkeypatch.context() as ctx:
        # make it look like the utilies always exist
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "check_for_utility",
            mock_util_check,
        )
        # mock
        ctx.setattr(
            smartsim._core.entrypoints.dragon_install,
            "_execute_platform_cmd",
            lambda x: is_cray,
        )

        platform_result = is_crayex_platform()
        assert is_cray == platform_result


def test_install_package__no_wheel(test_dir: str):
    """Verify that a missing wheel does not blow up and has a failure retcode"""
    exp_path = pathlib.Path(test_dir) / mock_archive_name.replace(".tar.gz", "")

    result = install_package(exp_path)
    assert result != 0
