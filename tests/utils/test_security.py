import pathlib
import stat

import pytest

from smartsim._core.config.config import get_config
from smartsim._core.utils.security import KeyLocator, KeyManager


@pytest.mark.parametrize(
    "separate_keys",
    [
        pytest.param(False, id="do not separate"),
        pytest.param(True, id="do separate keys"),
    ],
)
def test_keylocator_filename_resolution(separate_keys: bool, test_dir: str) -> None:
    """Ensure the key locator resolves filenames as expected. The value
    for `separate_keys` should not affect the file name"""
    key_path = pathlib.Path(test_dir)
    key_category = "mycategory"
    key_file = "mykey"
    locator = KeyLocator(key_path, key_file, key_category, separate_keys=separate_keys)

    assert locator.public_filename == f"{key_file}.key", "public mismatch"
    assert locator.private_filename == f"{key_file}.key_secret", "private mismatch"


def test_keylocator_separate_dir_resolution(test_dir: str) -> None:
    """Ensure the key locator resolves paths as expected. The value
    for `separate_keys` defaults to `True`"""
    key_path = pathlib.Path(test_dir)
    key_name = "test"
    key_category = "mycategory"

    locator = KeyLocator(key_path, key_name, key_category)

    # we expect a category AND pub/priv subdirectory
    exp_pub = pathlib.Path(f"{test_dir}/{key_category}/pub").resolve()
    assert str(locator.public_dir) == str(exp_pub)

    exp_priv = pathlib.Path(f"{test_dir}/{key_category}/priv").resolve()
    assert str(locator.private_dir) == str(exp_priv)


def test_keylocator_dir_resolution(test_dir: str) -> None:
    """Ensure the key locator resolves paths as expected. Passing
    `separate_keys=True` should result in matching pub/priv paths"""
    key_path = pathlib.Path(test_dir)
    key_name = "test"
    key_category = "mycategory"

    locator = KeyLocator(key_path, key_name, key_category, separate_keys=False)

    # we expect a category but NO pub/priv subdirectory
    exp_pub = pathlib.Path(f"{test_dir}/{key_category}").resolve()
    assert str(locator.public_dir) == str(exp_pub)

    exp_priv = pathlib.Path(f"{test_dir}/{key_category}").resolve()
    assert str(locator.private_dir) == str(exp_priv)

    # and to be explicit... prove that they're the same directory
    assert str(exp_pub) == str(exp_priv)


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
        server_locator = KeyLocator(pathlib.Path(test_dir), "curve", "server")
        client_locator = KeyLocator(pathlib.Path(test_dir), "curve", "client")

        locators = [server_locator, client_locator]

        for locator in locators:
            assert locator.public_dir.exists()
            assert locator.private_dir.exists()


def test_key_manager_get_existing_keys_only(
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


def test_key_manager_server_context(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager does not load private client
    keys when the context is set to server=True"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg, as_server=True)

        server_keyset, client_keyset = km.get_keys()

        # as_server=True returns pub/priv server keys...
        assert len(server_keyset.public) > 0
        assert len(server_keyset.private) > 0

        # as_server=True returns only public client key
        assert len(client_keyset.public) > 0
        assert len(client_keyset.private) == 0


def test_key_manager_client_context(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager does not load private client
    keys when the context is set to client=True"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg, as_client=True)

        server_keyset, client_keyset = km.get_keys()

        # as_client=True returns pub/priv client keys...
        assert len(server_keyset.public) > 0
        assert len(server_keyset.private) == 0

        # e=True returns only public server key
        assert len(client_keyset.public) > 0
        assert len(client_keyset.private) > 0


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

        assert stat.S_IMODE(s_pub_stat.st_mode) == 0o744
        assert stat.S_IMODE(c_pub_stat.st_mode) == 0o744

        # ensure private dirs are open only to owner
        s_priv_stat = km._server_locator.private_dir.stat()
        c_priv_stat = km._client_locator.private_dir.stat()

        assert stat.S_IMODE(s_priv_stat.st_mode) == 0o700
        assert stat.S_IMODE(c_priv_stat.st_mode) == 0o700

        # ensure public files are open for reading by others
        s_pub_stat = km._server_locator.public.stat()
        c_pub_stat = km._client_locator.public.stat()

        assert stat.S_IMODE(s_pub_stat.st_mode) == 0o744
        assert stat.S_IMODE(c_pub_stat.st_mode) == 0o744

        # ensure private files are read-only for owner
        s_priv_stat = km._server_locator.private.stat()
        c_priv_stat = km._client_locator.private.stat()

        assert stat.S_IMODE(s_priv_stat.st_mode) == 0o600
        assert stat.S_IMODE(c_priv_stat.st_mode) == 0o600
