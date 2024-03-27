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

    public: str
    """The public key"""
    private: str
    """The private key"""


class KeyLocator:
    """Encapsulates logic for determining the path where a
    `KeyPair` will persist on disk"""

    def __init__(
        self, root_dir: pathlib.Path, filename: str, separate_keys: bool = True
    ) -> None:
        """Initiailize a `KeyLocator`

        :param root_dir: root path where keys are persisted to disk
        :type root_dir: pathlib.Path
        :param filename: the stem name of the key file
        :type filename: str
        :param separate_keys: flag indicating if keys should be separated into
        public and private subdirectories
        :type separate_keys: bool

        """
        # constants for standardized paths.
        self._public_subdir = "pub"
        self._private_subdir = "priv"
        self._public_extension = "key"
        self._private_extension = "key_secret"

        self._key_root_dir = root_dir
        self._filename = filename
        self._separate_keys = separate_keys

    _key_root_dir: pathlib.Path
    """directory path to location of key files"""
    _filename: str
    """base name for key files"""
    _separate_keys: bool
    """flag indicating if public and private keys are persisted separately"""

    @property
    def public_dir(self) -> pathlib.Path:
        """target directory for the public key"""
        return self.public.parent

    @property
    def private_dir(self) -> pathlib.Path:
        """target directory for the private key"""
        return self.private.parent

    @property
    def public_filename(self) -> pathlib.Path:
        """The filename (<stem>.<suffix>) for the public key file"""
        return f"{self._filename}.{self._public_extension}"

    @property
    def private_filename(self) -> pathlib.Path:
        """The filename (<stem>.<suffix>) for the private key file"""
        return f"{self._filename}.{self._private_extension}"

    @property
    def public(self) -> pathlib.Path:
        """The full target path for the public key file"""
        # combine the root and key type, e.g. /foo/bar + /server
        p = self._key_root_dir / self._filename
        # combine the pub/priv key subdir if necessary
        # e.g. /foo/bar + /pub
        if self._separate_keys:
            p = p / self._public_subdir

        p = p / self.public_filename
        return p

    @property
    def private(self) -> pathlib.Path:
        """The full target path for the private key file"""
        # combine the root and key type, e.g. /foo/bar + /server
        p = self._key_root_dir / self._filename
        # combine the pub/priv key subdir if necessary
        # e.g. /foo/bar + /priv
        if self._separate_keys:
            p = p / self._private_subdir

        p = p / self.private_filename
        return p


class KeyManager:
    def __init__(
        self, config: Config, as_server: bool = False, as_client: bool = False
    ) -> None:
        """Initialize a KeyManager instance.
        :param config: SmartSim configuration
        :type config: Config
        :param server: flag indicating if executing in context of a server;
        set to `True` to avoid loading client secret key
        :type server: bool
        :param server: flag indicating if executing in context of a client;
        set to `True` to avoid loading server secret key
        :type server: bool"""
        self._key_dir = pathlib.Path(config.smartsim_key_dir).resolve()

        self._as_server = as_server
        self._as_client = as_client

        self._server_base = "server"
        self._client_base = "client"

        self._server_locator = KeyLocator(self._key_dir, self._server_base)
        self._client_locator = KeyLocator(self._key_dir, self._client_base)

    def create_directories(self) -> None:
        """Prepare the directory structure for key persistence"""
        self._prepare_file_system()

    @property
    def key_dir(self) -> pathlib.Path:
        """The root path to keys for this experiment"""
        return self._key_dir

    def _prepare_file_system(self) -> None:
        """Create the subdirectory structure necessary to hold
        the public and private key pairs for servers & clients"""
        for locator in [self._server_locator, self._client_locator]:
            if not locator.public_dir.exists():
                locator.public_dir.mkdir(parents=True)

            if not locator.private_dir.exists():
                locator.private_dir.mkdir(parents=True)

    def _load_keypair(self, locator: KeyLocator, in_context: bool) -> KeyPair:
        # private keys contain public & private key parts
        key_path = locator.public
        if in_context:
            key_path = locator.private
        pub_key, priv_key = zmq.auth.load_certificate(key_path)
        return KeyPair(pub_key, priv_key or "")

    def _load_keys(self) -> t.Tuple[KeyPair, ...]:
        """Use ZMQ auth to load public/private key pairs for the server and client
        components from the standard key paths for the associated experiment"""
        try:
            server_keys = self._load_keypair(self._server_locator, self._as_server)
            client_keys = self._load_keypair(self._client_locator, self._as_client)

            return (server_keys, client_keys)
        except (ValueError, OSError) as ex:
            # no keys could be loaded from disk
            ...
        return (None, None)

    def _move_public_key(self, locator: KeyLocator) -> None:
        """The public and private key pair are created in the same directory. Move
        the public key out of the private subdir and into the public subdir"""
        pub_path = locator.private.with_suffix(locator.public.suffix)
        pub_path.rename(locator.public)

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

    def get_keys(self, no_create: bool = False) -> t.Tuple[t.Optional[KeyPair], ...]:
        """Use ZMQ auth to generate a public/private key pair for the server and
        client components. Return paths to all four keys in a single `KeySet`"""
        keys = self._load_keys()
        if keys[0] is not None or keys[1] is not None:
            return keys

        if no_create:
            return (None, None)

        self.create_directories()
        self._create_keys()

        # ensure keys are persisted to avoid inability to connect
        return self._load_keys()
