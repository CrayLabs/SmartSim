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


import dataclasses
import pathlib
import typing as t

import zmq
import zmq.auth

from smartsim._core.config.config import Config


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


class KeyLocator:
    """Determines the paths to use when persisting a `KeyPair` to disk"""

    def __init__(
        self,
        root_dir: pathlib.Path,
        filename: str,
        category: str,
        separate_keys: bool = True,
    ) -> None:
        """Initiailize a `KeyLocator`

        :param root_dir: root path where keys are persisted to disk
        :type root_dir: pathlib.Path
        :param filename: the stem name of the key file
        :type filename: str
        :param category: the category or use-case for the key (e.g. server)
        :type category: str
        :param separate_keys: flag indicating if public and private keys should
        be persisted in separate, corresponding directories
        :type separate_keys: bool

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

        self._separate_keys = separate_keys
        """Flag indicating if public and private keys are persisted separately"""

        self._category = category
        """(optional) Category name used to further separate key locations"""

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
        path = self._key_root_dir / self._category  # self._filename
        # combine the pub/priv key subdir if necessary (e.g. /foo/bar + /pub)
        if self._separate_keys:
            path = path / self._public_subdir

        return path / self.public_filename

    @property
    def private(self) -> pathlib.Path:
        """Full target path of the private key file"""
        # combine the root and key type (e.g. /foo/bar + /server)
        path = self._key_root_dir / self._category
        # combine the pub/priv key subdir if necessary (e.g. /foo/bar + /priv)
        if self._separate_keys:
            path = path / self._private_subdir

        return path / self.private_filename


class KeyManager:
    def __init__(
        self, config: Config, as_server: bool = False, as_client: bool = False
    ) -> None:
        """Initialize a KeyManager instance.
        :param config: SmartSim configuration
        :type config: Config
        :param as_server: flag to indicate when executing in the server context;
        set to `True` to avoid loading client secret key
        :type as_server: bool
        :param as_client: flag to indicate when executing in the client context;
        set to `True` to avoid loading server secret key
        :type as_client: bool"""

        self._as_server = as_server
        """Set to `True` to return keys appropriate for the server context"""

        self._as_client = as_client
        """Set to `True` to return keys appropriate for the client context"""

        self._server_base = "server"
        """The `category` directory for persisting server keys"""

        self._client_base = "client"
        """The `category` directory for persisting client keys"""

        key_dir = pathlib.Path(config.smartsim_key_path).resolve()

        self._server_locator = KeyLocator(key_dir, "smartsim", self._server_base)
        """The locator for producing the paths to store server key files"""
        self._client_locator = KeyLocator(key_dir, "smartsim", self._client_base)
        """The locator for producing the paths to store client key files"""

    def create_directories(self) -> None:
        """Create the subdirectory structure necessary to hold
        the public and private key pairs for servers & clients"""
        for locator in [self._server_locator, self._client_locator]:
            if not locator.public_dir.exists():
                locator.public_dir.mkdir(parents=True)

            if not locator.private_dir.exists():
                locator.private_dir.mkdir(parents=True)

    @classmethod
    def _load_keypair(cls, locator: KeyLocator, in_context: bool) -> KeyPair:
        """Load a specific `KeyPair` from disk

        :param locator: a `KeyLocator` that specifies the path to an existing key
        :type locator: KeyLocator
        :param in_context: Boolean flag indicating if the keypair is the active
        context; ensures the public and private keys are both loaded when `True`.
        Only the public key is loaded when `False`
        :type in_context: bool
        :returns: a KeyPair containing the loaded public/private key
        :rtype: KeyPair
        """
        # private keys contain public & private key parts
        key_path = locator.private if in_context else locator.public
        pub_key, priv_key = zmq.auth.load_certificate(key_path)

        # avoid a `None` value in the private key when it isn't loaded
        return KeyPair(pub_key, priv_key or b"")

    def _load_keys(self) -> t.Tuple[KeyPair, KeyPair]:
        """Use ZMQ auth to load public/private key pairs for the server and client
        components from the standard key paths for the associated experiment

        :returns: 2-tuple of `KeyPair` (server_keypair, client_keypair)
        :rtype: Tuple[KeyPair, KeyPair]"""
        try:
            server_keys = self._load_keypair(self._server_locator, self._as_server)
            client_keys = self._load_keypair(self._client_locator, self._as_client)

            return server_keys, client_keys
        except (ValueError, OSError):
            # expected if no keys could be loaded from disk
            ...

        return KeyPair(), KeyPair()

    @classmethod
    def _move_public_key(cls, locator: KeyLocator) -> None:
        """The public and private key pair are created in the same directory. Move
        the public key out of the private subdir and into the public subdir

        :param locator: `KeyLocator` that determines the path to the
        key pair persisted in the same directory.
        :type locator: KeyLocator"""
        new_path = locator.private.with_suffix(locator.public.suffix)
        if new_path != locator.public:
            new_path.rename(locator.public)

    def _create_keys(self) -> None:
        """Create and persist key files to disk"""
        # create server keys in the server private directory
        zmq.auth.create_certificates(
            self._server_locator.private.parent, self._server_locator.private.stem
        )
        # ...and move the server public key out of the private subdirectory
        self._move_public_key(self._server_locator)

        # create client keys in the client private directory
        zmq.auth.create_certificates(
            self._client_locator.private.parent, self._client_locator.private.stem
        )
        # ...and move the client public key out of the private subdirectory
        self._move_public_key(self._client_locator)

    def get_keys(self, create: bool = True) -> t.Tuple[KeyPair, KeyPair]:
        """Use ZMQ auth to generate a public/private key pair for the server
        and client components.

        :param no_create: pass `no_create=True` to ensure keys are not
        created and only pre-existing keys can be loaded
        :returns: 2-tuple of `KeyPair` (server_keypair, client_keypair)
        :rtype: t.Tuple[KeyPair, KeyPair]
        """
        server_keys, client_keys = self._load_keys()

        # check if we received "empty keys"
        if not server_keys.empty or not client_keys.empty:
            return server_keys, client_keys

        if not create:
            # if directed not to create new keys, return "empty keys"
            return KeyPair(), KeyPair()

        self.create_directories()
        self._create_keys()

        # load keys to ensure they were persisted
        return self._load_keys()
