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
    private: str


class KeyManager:
    def __init__(
        self, config: Config, server: bool = False, client: bool = False
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
        self._server = server
        self._client = client
        self._prepare_file_system()

    @property
    def key_dir(self) -> pathlib.Path:
        """The root path to keys for this experiment"""
        return self._key_dir

    @property
    def server_key_dir(self) -> pathlib.Path:
        """The path to keys for the server component of this experiment"""
        return self.key_dir / "server"

    @property
    def client_key_dir(self) -> pathlib.Path:
        """The path to keys for the client component of this experiment"""
        return self.key_dir / "client"

    def _prepare_file_system(self) -> None:
        subdirs = [
            self.server_key_dir / "pub",
            self.server_key_dir / "priv",
            self.client_key_dir / "pub",
            self.client_key_dir / "priv",
        ]

        for directory in subdirs:
            if not directory.exists():
                directory.mkdir(parents=True)

    def _load_keys(self) -> t.Tuple[KeyPair, ...]:
        """Use ZMQ auth to load public/private key pairs for the server and client
        components from the standard key paths for the associated experiment"""
        try:
            # todo: avoid hardcoding key names...
            spub, spriv = zmq.auth.load_certificate(
                self.server_key_dir / "server.key_secret"
            )
            cpub, cpriv = zmq.auth.load_certificate(
                self.client_key_dir / "client.key_secret"
            )

            # spub, spriv = server_keys
            # cpub, cpriv = client_keys

            # clear private keys from memory if not necessary
            cpriv = "" if self._server else cpriv
            spriv = "" if self._client else spriv

            return (KeyPair(spub, spriv), KeyPair(cpub, cpriv))
        except (ValueError, OSError) as ex:
            # no keys could be loaded from disk
            ...
        return (None, None)

    def _create_keys(self) -> None:
        """Create and persist key files to disk"""
        for key_dir in [self.server_key_dir, self.client_key_dir]:
            # key_dir.mkdir(mode=644)
            key_dir.mkdir(parents=True, exist_ok=True)

        # todo: unify how load/create get the base key name
        zmq.auth.create_certificates(self.server_key_dir, "server")
        zmq.auth.create_certificates(self.client_key_dir, "client")

    def get_keys(self, no_create: bool = False) -> t.Tuple[t.Optional[KeyPair], ...]:
        """Use ZMQ auth to generate a public/private key pair for the server and
        client components. Return paths to all four keys in a single `KeySet`"""
        keys = self._load_keys()
        if keys[0] is not None or keys[1] is not None:
            return keys

        if no_create:
            return (None, None)

        self._create_keys()

        # ensure keys are persisted to avoid inability to connect
        return self._load_keys()
