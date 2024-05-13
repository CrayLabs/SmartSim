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

import pathlib
import sys
import tarfile
import typing as t
from collections import namedtuple

import pytest
from github.GitReleaseAsset import GitReleaseAsset
from github.Requester import Requester

import smartsim
import smartsim._core.utils.helpers as helpers
from smartsim._core._cli.scripts.dragon_install import (
    cleanup,
    create_dotenv,
    install_dragon,
    install_package,
    retrieve_asset,
    retrieve_asset_info,
)
from smartsim.error.errors import SmartSimCLIActionCancelled

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


mock_archive_name = "dragon-0.8-py3.9.4.1-CRAYEX-ac132fe95.tar.gz"
_git_attr = namedtuple("_git_attr", "value")


@pytest.fixture
def test_archive(test_dir: str, archive_path: pathlib.Path) -> pathlib.Path:
    """Fixture for returning a simple tarfile to test on"""
    num_files = 10
    with tarfile.TarFile.open(archive_path, mode="w:gz") as tar:
        mock_whl = pathlib.Path(test_dir) / "mock.whl"
        mock_whl.touch()

        for i in range(num_files):
            content = pathlib.Path(test_dir) / f"{i:04}.txt"
            content.write_text(f"i am file {i}\n")
            tar.add(content)
    return archive_path


@pytest.fixture
def archive_path(test_dir: str) -> pathlib.Path:
    """Fixture for returning a dir path based on the default mock asset archive name"""
    path = pathlib.Path(test_dir) / mock_archive_name
    return path


@pytest.fixture
def extraction_dir(test_dir: str) -> pathlib.Path:
    """Fixture for returning a dir path based on the default mock asset archive name"""
    path = pathlib.Path(test_dir) / mock_archive_name.replace(".tar.gz", "")
    return path


@pytest.fixture
def test_assets(monkeypatch: pytest.MonkeyPatch) -> t.Dict[str, GitReleaseAsset]:
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

    assets: t.List[GitReleaseAsset] = []
    mock_archive_name_tpl = "{}-{}.4.1-{}ac132fe95.tar.gz"

    for python_version in ["py3.9", "py3.10", "py3.11"]:
        for dragon_version in ["dragon-0.8", "dragon-0.9", "dragon-0.10"]:
            for platform in ["", "CRAYEX-"]:

                asset = GitReleaseAsset(requester, headers, attributes, completed)

                archive_name = mock_archive_name_tpl.format(
                    dragon_version, python_version, platform
                )

                monkeypatch.setattr(
                    asset,
                    "_browser_download_url",
                    _git_attr(value=f"http://foo/{archive_name}"),
                )
                monkeypatch.setattr(asset, "_name", _git_attr(value=archive_name))
                assets.append(asset)

    return assets


def test_cleanup_no_op(archive_path: pathlib.Path) -> None:
    """Ensure that the cleanup method doesn't bomb when called with
    missing archive path; simulate a failed download"""
    # confirm assets do not exist
    assert not archive_path.exists()

    # call cleanup. any exceptions should break test...
    cleanup(archive_path)


def test_cleanup_archive_exists(test_archive: pathlib.Path) -> None:
    """Ensure that the cleanup method removes the archive"""
    assert test_archive.exists()

    cleanup(test_archive)

    # verify archive is gone after cleanup
    assert not test_archive.exists()


def test_retrieve_cached(
    test_dir: str,
    # archive_path: pathlib.Path,
    test_archive: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that a previously retrieved asset archive is re-used"""
    with tarfile.TarFile.open(test_archive) as tar:
        tar.extractall(test_dir)

    ts1 = test_archive.parent.stat().st_ctime

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
    monkeypatch.setattr(asset, "_browser_download_url", _git_attr(value="http://foo"))
    monkeypatch.setattr(asset, "_name", _git_attr(value=mock_archive_name))

    asset_path = retrieve_asset(test_archive.parent, asset)
    ts2 = asset_path.stat().st_ctime

    assert (
        asset_path == test_archive.parent
    )  # show that the expected path matches the output path
    assert ts1 == ts2  # show that the file wasn't changed...


@pytest.mark.parametrize(
    "dragon_pin,pyv,is_found,is_crayex",
    [
        pytest.param("0.8", "py3.8", False, False, id="0.8,python 3.8"),
        pytest.param("0.8", "py3.9", True, False, id="0.8,python 3.9"),
        pytest.param("0.8", "py3.10", True, False, id="0.8,python 3.10"),
        pytest.param("0.8", "py3.11", True, False, id="0.8,python 3.11"),
        pytest.param("0.8", "py3.12", False, False, id="0.8,python 3.12"),
        pytest.param("0.8", "py3.8", False, True, id="0.8,python 3.8,CrayEX"),
        pytest.param("0.8", "py3.9", True, True, id="0.8,python 3.9,CrayEX"),
        pytest.param("0.8", "py3.10", True, True, id="0.8,python 3.10,CrayEX"),
        pytest.param("0.8", "py3.11", True, True, id="0.8,python 3.11,CrayEX"),
        pytest.param("0.8", "py3.12", False, True, id="0.8,python 3.12,CrayEX"),
        pytest.param("0.9", "py3.8", False, False, id="0.9,python 3.8"),
        pytest.param("0.9", "py3.9", True, False, id="0.9,python 3.9"),
        pytest.param("0.9", "py3.10", True, False, id="0.9,python 3.10"),
        pytest.param("0.9", "py3.11", True, False, id="0.9,python 3.11"),
        pytest.param("0.9", "py3.12", False, False, id="0.9,python 3.12"),
        pytest.param("0.9", "py3.8", False, True, id="0.9,python 3.8,CrayEX"),
        pytest.param("0.9", "py3.9", True, True, id="0.9,python 3.9,CrayEX"),
        pytest.param("0.9", "py3.10", True, True, id="0.9,python 3.10,CrayEX"),
        pytest.param("0.9", "py3.11", True, True, id="0.9,python 3.11,CrayEX"),
        pytest.param("0.9", "py3.12", False, True, id="0.9,python 3.12,CrayEX"),
        # add a couple variants for a dragon version that isn't in the asset list
        pytest.param("0.7", "py3.9", False, False, id="0.7,python 3.9"),
        pytest.param("0.7", "py3.9", False, True, id="0.7,python 3.9,CrayEX"),
    ],
)
def test_retrieve_asset_info(
    test_assets: t.Collection[GitReleaseAsset],
    monkeypatch: pytest.MonkeyPatch,
    dragon_pin: str,
    pyv: str,
    is_found: bool,
    is_crayex: bool,
) -> None:
    """Verify that an information is retrieved correctly based on the python
    version, platform (e.g. CrayEX, !CrayEx), and target dragon pin"""

    with monkeypatch.context() as ctx:
        ctx.setattr(
            smartsim._core._cli.scripts.dragon_install,
            "python_version",
            lambda: pyv,
        )
        ctx.setattr(
            smartsim._core._cli.scripts.dragon_install,
            "is_crayex_platform",
            lambda: is_crayex,
        )
        ctx.setattr(
            smartsim._core._cli.scripts.dragon_install,
            "dragon_pin",
            lambda: dragon_pin,
        )
        # avoid hitting github API
        ctx.setattr(
            smartsim._core._cli.scripts.dragon_install,
            "_get_release_assets",
            lambda: test_assets,
        )

        if is_found:
            chosen_asset = retrieve_asset_info()

            assert chosen_asset
            assert pyv in chosen_asset.name
            assert dragon_pin in chosen_asset.name

            if is_crayex:
                assert "crayex" in chosen_asset.name.lower()
            else:
                assert "crayex" not in chosen_asset.name.lower()
        else:
            with pytest.raises(SmartSimCLIActionCancelled):
                retrieve_asset_info()


def test_check_for_utility_missing(test_dir: str) -> None:
    """Ensure that looking for a missing utility doesn't raise an exception"""
    ld_config = pathlib.Path(test_dir) / "ldconfig"

    utility = helpers.check_for_utility(ld_config)

    assert not utility


def test_check_for_utility_exists() -> None:
    """Ensure that looking for an existing utility returns a non-empty path"""
    utility = helpers.check_for_utility("ls")
    assert utility


def test_is_crayex_missing_ldconfig(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the cray ex platform check doesn't fail when ldconfig isn't
    available for use"""

    def mock_util_check(util: str) -> str:
        if util == "ldconfig":
            return ""
        return "w00t!"

    with monkeypatch.context() as ctx:
        # mock utility existence
        ctx.setattr(
            helpers,
            "check_for_utility",
            mock_util_check,
        )

        is_cray = helpers.is_crayex_platform()
        assert not is_cray


def test_is_crayex_missing_fi_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the cray ex platform check doesn't fail when fi_info isn't
    available for use"""

    def mock_util_check(util: str) -> str:
        if util == "fi_info":
            return ""
        return "w00t!"

    with monkeypatch.context() as ctx:
        # mock utility existence
        ctx.setattr(
            helpers,
            "check_for_utility",
            mock_util_check,
        )

        is_cray = helpers.is_crayex_platform()
        assert not is_cray


@pytest.mark.parametrize(
    "is_cray,output,return_code",
    [
        pytest.param(True, "cray pmi2.so\ncxi\ncray pmi.so\npni.so", 0, id="CrayEX"),
        pytest.param(False, "cray pmi2.so\ncxi\npni.so", 0, id="No PMI"),
        pytest.param(False, "cxi\ncray pmi.so\npni.so", 0, id="No PMI 2"),
        pytest.param(False, "cray pmi2.so\ncray pmi.so\npni.so", 0, id="No CXI"),
        pytest.param(False, "pmi.so\ncray pmi2.so\ncxi", 0, id="Non Cray PMI"),
        pytest.param(False, "cray pmi.so\npmi2.so\ncxi", 0, id="Non Cray PMI2"),
    ],
)
def test_is_cray_ex(
    monkeypatch: pytest.MonkeyPatch, is_cray: bool, output: str, return_code: int
) -> None:
    """Test that cray ex platform check result is returned as expected"""

    def mock_util_check(util: str) -> bool:
        # mock that we have the necessary tools
        return True

    with monkeypatch.context() as ctx:
        # make it look like the utilies always exist
        ctx.setattr(
            helpers,
            "check_for_utility",
            mock_util_check,
        )
        # mock
        ctx.setattr(
            helpers,
            "execute_platform_cmd",
            lambda x: (output, return_code),
        )

        platform_result = helpers.is_crayex_platform()
        assert is_cray == platform_result


def test_install_package_no_wheel(extraction_dir: pathlib.Path):
    """Verify that a missing wheel does not blow up and has a failure retcode"""
    exp_path = extraction_dir

    result = install_package(exp_path)
    assert result != 0


def test_install_macos(monkeypatch: pytest.MonkeyPatch, extraction_dir: pathlib.Path):
    """Verify that installation exits cleanly if installing on unsupported platform"""
    with monkeypatch.context() as ctx:
        ctx.setattr(sys, "platform", "darwin")

        result = install_dragon(extraction_dir)
        assert result == 1


def test_create_dotenv(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Verify that attempting to create a .env file without any existing
    file or container directory works"""
    test_path = pathlib.Path(test_dir)
    mock_dragon_root = pathlib.Path(test_dir) / "dragon"
    exp_env_path = pathlib.Path(test_dir) / "dragon" / ".env"

    with monkeypatch.context() as ctx:
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", test_path)

        # ensure no .env exists before trying to create it.
        assert not exp_env_path.exists()

        create_dotenv(mock_dragon_root)

        # ensure the .env is created as side-effect of create_dotenv
        assert exp_env_path.exists()


def test_create_dotenv_existing_dir(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Verify that attempting to create a .env file in an existing
    target dir works"""
    test_path = pathlib.Path(test_dir)
    mock_dragon_root = pathlib.Path(test_dir) / "dragon"
    exp_env_path = pathlib.Path(test_dir) / "dragon" / ".env"

    # set up the parent directory that will contain the .env
    exp_env_path.parent.mkdir(parents=True)

    with monkeypatch.context() as ctx:
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", test_path)

        # ensure no .env exists before trying to create it.
        assert not exp_env_path.exists()

        create_dotenv(mock_dragon_root)

        # ensure the .env is created as side-effect of create_dotenv
        assert exp_env_path.exists()


def test_create_dotenv_existing_dotenv(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Verify that attempting to create a .env file when one exists works as expected"""
    test_path = pathlib.Path(test_dir)
    mock_dragon_root = pathlib.Path(test_dir) / "dragon"
    exp_env_path = pathlib.Path(test_dir) / "dragon" / ".env"

    # set up the parent directory that will contain the .env
    exp_env_path.parent.mkdir(parents=True)

    # write something into file to verify it is overwritten
    var_name = "DRAGON_BASE_DIR"
    exp_env_path.write_text(f"{var_name}=/foo/bar")

    with monkeypatch.context() as ctx:
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", test_path)

        # ensure .env exists so we can update it
        assert exp_env_path.exists()

        create_dotenv(mock_dragon_root)

        # ensure the .env is created as side-effect of create_dotenv
        assert exp_env_path.exists()

        # ensure file was overwritten and env vars are not duplicated
        dotenv_content = exp_env_path.read_text(encoding="utf-8")
        split_content = dotenv_content.split(var_name)

        # split to confirm env var only appars once
        assert len(split_content) == 2


def test_create_dotenv_format(monkeypatch: pytest.MonkeyPatch, test_dir: str):
    """Verify that created .env files are correctly formatted"""
    test_path = pathlib.Path(test_dir)
    mock_dragon_root = pathlib.Path(test_dir) / "dragon"
    exp_env_path = pathlib.Path(test_dir) / "dragon" / ".env"

    with monkeypatch.context() as ctx:
        ctx.setattr(smartsim._core.config.CONFIG, "conf_dir", test_path)

        create_dotenv(mock_dragon_root)

        # ensure the .env is created as side-effect of create_dotenv
        content = exp_env_path.read_text(encoding="utf-8")

        # ensure we have values written, but ignore empty lines
        lines = [line for line in content.split("\n") if line]
        assert lines

        # ensure each line is formatted as key=value
        for line in lines:
            line_split = line.split("=")
            assert len(line_split) == 2
