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
import stat

import pytest
from sympy import public

from smartsim._core.config.config import get_config
from smartsim._core.utils.security import KeyManager, _KeyLocator, _KeyPermissions

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_keylocator_filename_resolution(test_dir: str) -> None:
    """Ensure the key locator resolves filenames as expected."""
    key_path = pathlib.Path(test_dir)
    key_category = "mycategory"
    key_file = "mykey"
    locator = _KeyLocator(key_path, key_file, key_category)

    assert locator.public_filename == f"{key_file}.key", "public mismatch"
    assert locator.private_filename == f"{key_file}.key_secret", "private mismatch"


def test_keylocator_dir_resolution(test_dir: str) -> None:
    """Ensure the key locator resolves paths as expected."""
    key_path = pathlib.Path(test_dir)
    key_name = "test"
    key_category = "mycategory"

    locator = _KeyLocator(key_path, key_name, key_category)

    # we expect a category and pub/priv subdirectory
    exp_pub = pathlib.Path(f"{test_dir}/{key_category}/pub").resolve()
    assert str(locator.public_dir) == str(exp_pub)

    exp_priv = pathlib.Path(f"{test_dir}/{key_category}/priv").resolve()
    assert str(locator.private_dir) == str(exp_priv)

    # and to be explicit... prove pub & priv are not same directory
    assert str(locator.private_dir) != str(locator.public_dir)


def test_key_manager_dir_preparation(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the KeyManager creates the appropriate directory
    structure required for public/private key pairs."""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg)

        km.create_directories()

        # verify the expected paths are created
        server_locator = _KeyLocator(pathlib.Path(test_dir), "curve", "server")
        client_locator = _KeyLocator(pathlib.Path(test_dir), "curve", "client")

        locators = [server_locator, client_locator]

        for locator in locators:
            assert locator.public_dir.exists()
            assert locator.private_dir.exists()


def test_key_manager_get_existing_keys_only_no_keys_found(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager cannot load keys when
    directed not to create missing keys."""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg)

        # use create=False to only load pre-existing keys
        server_keys, client_keys = km.get_keys(create=False)

        assert server_keys.empty
        assert client_keys.empty


def test_key_manager_get_existing_keys_only_existing(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager can load keys when
    they exist from a previous call."""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()

        # use a KeyManager to create some keys
        km = KeyManager(cfg, as_server=True, as_client=True)
        old_server_keys, old_client_keys = km.get_keys(create=True)

        # create a new KM to verify keys reload
        km = KeyManager(cfg, as_server=True, as_client=True)

        # use create=True to manifest any bugs missing existing keys
        server_keys, client_keys = km.get_keys(create=True)

        # ensure we loaded something
        assert not server_keys.empty
        assert not client_keys.empty

        # and show the old keys were reloaded from disk
        assert server_keys == old_server_keys
        assert client_keys == old_client_keys


def test_key_manager_get_or_create_keys_default(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager creates keys when none can be loaded"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg)

        key_set = km.get_keys()

        # public keys are returned by default
        assert key_set[0].public != b""
        assert key_set[1].public != b""

        # default behavior will only return public keys
        assert not key_set[0].private
        assert not key_set[1].private


@pytest.mark.parametrize(
    "as_server, as_client",
    [
        pytest.param(False, True, id="as-client"),
        pytest.param(True, False, id="as-server"),
        pytest.param(True, True, id="as-both"),
        pytest.param(False, False, id="public-only"),
    ],
)
def test_key_manager_as_context(
    as_server: bool,
    as_client: bool,
    test_dir: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ensure the key manager loads the correct keys
    when passed `as_server=True` and `as_client=True`"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg, as_server=as_server, as_client=as_client)

        server_keyset, client_keyset = km.get_keys()

        assert bool(server_keyset.public) == True
        assert bool(server_keyset.private) == as_server

        assert bool(client_keyset.public) == True
        assert bool(client_keyset.private) == as_client


def test_key_manager_applied_permissions(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager applies the appropriate file-system
    permissions to the keys and directories"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg, as_client=True, as_server=True)

        server_keys, client_keys = km.get_keys()

        # ensure public dirs are open for reading by others
        s_pub_stat = km._server_locator.public_dir.stat()
        c_pub_stat = km._client_locator.public_dir.stat()

        assert stat.S_IMODE(s_pub_stat.st_mode) == _KeyPermissions.PUBLIC_DIR
        assert stat.S_IMODE(c_pub_stat.st_mode) == _KeyPermissions.PUBLIC_DIR

        # ensure private dirs are open only to owner
        s_priv_stat = km._server_locator.private_dir.stat()
        c_priv_stat = km._client_locator.private_dir.stat()

        assert stat.S_IMODE(s_priv_stat.st_mode) == _KeyPermissions.PRIVATE_DIR
        assert stat.S_IMODE(c_priv_stat.st_mode) == _KeyPermissions.PRIVATE_DIR

        # ensure public files are open for reading by others
        s_pub_stat = km._server_locator.public.stat()
        c_pub_stat = km._client_locator.public.stat()

        assert stat.S_IMODE(s_pub_stat.st_mode) == _KeyPermissions.PUBLIC_KEY
        assert stat.S_IMODE(c_pub_stat.st_mode) == _KeyPermissions.PUBLIC_KEY

        # ensure private files are read-only for owner
        s_priv_stat = km._server_locator.private.stat()
        c_priv_stat = km._client_locator.private.stat()

        assert stat.S_IMODE(s_priv_stat.st_mode) == _KeyPermissions.PRIVATE_KEY
        assert stat.S_IMODE(c_priv_stat.st_mode) == _KeyPermissions.PRIVATE_KEY
