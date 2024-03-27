import pathlib
import pytest

from smartsim._core.config.config import get_config
from smartsim._core.utils.security import KeyManager


def test_key_manager_dir_preparation(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the KeyManager creates the appropriate directory
    structure required for public/private key pairs."""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg, server=True)

        key_dirs = [
            pathlib.Path(test_dir) / "server" / "pub",
            pathlib.Path(test_dir) / "server" / "priv",
            pathlib.Path(test_dir) / "client" / "pub",
            pathlib.Path(test_dir) / "client" / "priv",
        ]

        for dir in key_dirs:
            assert dir.exists()


def test_key_manager_get_existing_keys(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager loads only existing keys when
    asked not to create new keys."""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg, server=True)

        key_set = km.get_keys(no_create=True)
        assert key_set[0] is None
        assert key_set[1] is None


def test_key_manager_get_or_create_keys(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager creates keys when none can be loaded"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg)

        key_set = km.get_keys()
        assert key_set[0] is not None
        assert key_set[1] is not None


def test_key_manager_server_context(
    test_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure the key manager does not load private client
    keys when the context is set to server=True"""
    with monkeypatch.context() as ctx:
        ctx.setenv("SMARTSIM_KEY_PATH", test_dir)

        cfg = get_config()
        km = KeyManager(cfg, server=True)

        server_keyset, client_keyset = km.get_keys()

        assert len(server_keyset.public) > 0
        assert len(server_keyset.private) > 0

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
        km = KeyManager(cfg, client=True)

        server_keyset, client_keyset = km.get_keys()

        assert len(server_keyset.public) > 0
        assert len(server_keyset.private) == 0

        assert len(client_keyset.public) > 0
        assert len(client_keyset.private) > 0
