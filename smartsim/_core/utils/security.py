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


import dataclasses
import pathlib
import stat
import typing as t
from enum import IntEnum

import zmq
import zmq.auth

from smartsim._core.config.config import Config
from smartsim.log import get_logger

logger = get_logger(__name__)


class _KeyPermissions(IntEnum):
    """Permissions used by KeyManager"""

    PRIVATE_KEY = stat.S_IRUSR | stat.S_IWUSR
    """Permissions only allowing an owner to read and write the file"""
    PUBLIC_KEY = stat.S_IRUSR | stat.S_IWUSR | stat.S_IROTH | stat.S_IRGRP
    """Permissions allowing an owner, others, and the group to read a file"""

    PRIVATE_DIR = (
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IXOTH | stat.S_IXGRP
    )
    """Permissions allowing only owners to read, write and traverse a directory"""
    PUBLIC_DIR = (
        stat.S_IRUSR
        | stat.S_IWUSR
        | stat.S_IXUSR
        | stat.S_IROTH
        | stat.S_IXOTH
        | stat.S_IRGRP
        | stat.S_IXGRP
    )
    """Permissions allowing non-owners to traverse a directory"""


@dataclasses.dataclass(frozen=True)
class KeyPair:
    """A public and private key pair"""

    public: bytes = dataclasses.field(default=b"")
    """The public key"""

    private: bytes = dataclasses.field(default=b"", repr=False)
    """The private key"""

    @property
    def empty(self) -> bool:
        """Return `True` if the KeyPair has no key values set. Useful
        for faking the null object pattern"""
        return self.public == self.private and len(self.public) == 0


class _KeyLocator:
    """Determines the paths to use when persisting a `KeyPair` to disk"""

    def __init__(
        self,
        root_dir: pathlib.Path,
        filename: str,
        category: str,
    ) -> None:
        """Initiailize a `KeyLocator`

        :param root_dir: root path where keys are persisted to disk
        :param filename: the stem name of the key file
        :param category: the category or use-case for the key (e.g. server)
        :param separate_keys: flag indicating if public and private keys should
        be persisted in separate, corresponding directories
        """

        # constants for standardized paths.
        self._public_subdir = "pub"
        """The category subdirectory to use when persisting a public key"""

        self._private_subdir = "priv"
        """The category subdirectory to use when persisting a private key"""

        self._public_extension = "key"
        """The extension found on public keys"""

        self._private_extension = "key_secret"
        """The extension found on private keys"""

        self._key_root_dir = root_dir
        """Path to the root directory containing key files"""

        self._filename = filename
        """Base name for key files"""

        self._category = category
        """Category name used to further separate key locations"""

    @property
    def public_dir(self) -> pathlib.Path:
        """Target directory for the public key"""
        return self.public.parent

    @property
    def private_dir(self) -> pathlib.Path:
        """Target directory for the private key"""
        return self.private.parent

    @property
    def public_filename(self) -> str:
        """Filename (<stem>.<suffix>) of the public key file"""
        return f"{self._filename}.{self._public_extension}"

    @property
    def private_filename(self) -> str:
        """Filename (<stem>.<suffix>) of the private key file"""
        return f"{self._filename}.{self._private_extension}"

    @property
    def public(self) -> pathlib.Path:
        """Full target path of the public key file"""
        # combine the root and key type (e.g. /foo/bar + /server)
        # then combine the pub/priv key subdir (e.g. /foo/bar/server + /pub)
        path = self._key_root_dir / self._category / self._public_subdir
        return path / self.public_filename

    @property
    def private(self) -> pathlib.Path:
        """Full target path of the private key file"""
        # combine the root and key type (e.g. /foo/bar + /server)
        # then combine the pub/priv key subdir (e.g. /foo/bar/server + /pub)
        path = self._key_root_dir / self._category / self._private_subdir
        # combine the pub/priv key subdir if necessary (e.g. /foo/bar + /priv)

        return path / self.private_filename


class KeyManager:
    def __init__(
        self, config: Config, as_server: bool = False, as_client: bool = False
    ) -> None:
        """Initialize a KeyManager instance.
        :param config: SmartSim configuration
        :param as_server: flag to indicate when executing in the server context;
        set to `True` to avoid loading client secret key
        :param as_client: flag to indicate when executing in the client context;
        set to `True` to avoid loading server secret key
        """

        self._as_server = as_server
        """Set to `True` to return keys appropriate for the server context"""

        self._as_client = as_client
        """Set to `True` to return keys appropriate for the client context"""

        key_dir = pathlib.Path(config.smartsim_key_path).resolve()

        # Results in key path such as <key_root>/server/pub/smartsim.key
        self._server_locator = _KeyLocator(key_dir, "smartsim", "server")
        """The locator for producing the paths to store server key files"""

        # Results in key path such as <key_root>/client/pub/smartsim.key
        self._client_locator = _KeyLocator(key_dir, "smartsim", "client")
        """The locator for producing the paths to store client key files"""

    def create_directories(self) -> None:
        """Create the subdirectory structure necessary to hold
        the public and private key pairs for servers & clients"""
        for locator in [self._server_locator, self._client_locator]:
            if not locator.public_dir.exists():
                permission = _KeyPermissions.PUBLIC_DIR
                logger.debug(f"Creating key dir: {locator.public_dir}, {permission}")
                locator.public_dir.mkdir(parents=True, mode=permission)

            if not locator.private_dir.exists():
                permission = _KeyPermissions.PRIVATE_DIR
                logger.debug(f"Creating key dir: {locator.private_dir}, {permission}")
                locator.private_dir.mkdir(parents=True, mode=permission)

    @classmethod
    def _load_keypair(cls, locator: _KeyLocator, in_context: bool) -> KeyPair:
        """Load a specific `KeyPair` from disk

        :param locator: a `KeyLocator` that specifies the path to an existing key
        :param in_context: Boolean flag indicating if the keypair is the active
        context; ensures the public and private keys are both loaded when `True`.
        Only the public key is loaded when `False`
        :returns: a KeyPair containing the loaded public/private key
        """
        # private keys contain public & private key parts
        key_path = locator.private if in_context else locator.public

        pub_key: bytes = b""
        priv_key: t.Optional[bytes] = b""

        if key_path.exists():
            logger.debug(f"Existing key files located at {key_path}")
            pub_key, priv_key = zmq.auth.load_certificate(key_path)
        else:
            logger.debug(f"No key files found at {key_path}")

        # avoid a `None` value in the private key when it isn't loaded
        return KeyPair(pub_key, priv_key or b"")

    def _load_keys(self) -> t.Tuple[KeyPair, KeyPair]:
        """Use ZMQ auth to load public/private key pairs for the server and client
        components from the standard key paths for the associated experiment

        :returns: 2-tuple of `KeyPair` (server_keypair, client_keypair)
        ]"""
        try:
            server_keys = self._load_keypair(self._server_locator, self._as_server)
            client_keys = self._load_keypair(self._client_locator, self._as_client)

            return server_keys, client_keys
        except (ValueError, OSError):
            # expected if no keys could be loaded from disk
            logger.warning("Loading key pairs failed.", exc_info=True)

        return KeyPair(), KeyPair()

    @classmethod
    def _move_public_key(cls, locator: _KeyLocator) -> None:
        """The public and private key pair are created in the same directory. Move
        the public key out of the private subdir and into the public subdir

        :param locator: `KeyLocator` that determines the path to the
        key pair persisted in the same directory.
        """
        new_path = locator.private.with_suffix(locator.public.suffix)
        if new_path != locator.public:
            logger.debug(f"Moving key file from {locator.public} to {new_path}")
            new_path.rename(locator.public)

    def _create_keys(self) -> None:
        """Create and persist key files to disk"""
        for locator in [self._server_locator, self._client_locator]:
            # create keys in the private directory...
            zmq.auth.create_certificates(locator.private_dir, locator.private.stem)

            # ...but move the public key out of the private subdirectory
            self._move_public_key(locator)

            # and ensure correct r/w/x permissions on each file.
            locator.private.chmod(_KeyPermissions.PRIVATE_KEY)
            locator.public.chmod(_KeyPermissions.PUBLIC_KEY)

    def get_keys(self, create: bool = True) -> t.Tuple[KeyPair, KeyPair]:
        """Use ZMQ auth to generate a public/private key pair for the server
        and client components.

        :param no_create: pass `no_create=True` to ensure keys are not
        created and only pre-existing keys can be loaded
        :returns: 2-tuple of `KeyPair` (server_keypair, client_keypair)
        """
        logger.debug(f"Loading keys, creation {'is' if create else 'not'} allowed")
        server_keys, client_keys = self._load_keys()

        # check if we received "empty keys"
        if not server_keys.empty or not client_keys.empty:
            return server_keys, client_keys

        if not create:
            # if directed not to create new keys, return "empty keys"
            logger.debug("Returning empty key pairs")
            return KeyPair(), KeyPair()

        self.create_directories()
        self._create_keys()

        # load keys to ensure they were persisted
        return self._load_keys()

    @property
    def client_keys_dir(self) -> pathlib.Path:
        "Return the path to the client public keys directory"
        return self._client_locator.public_dir
