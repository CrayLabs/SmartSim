# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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


from .._core.utils.helpers import cat_arg_and_value, init_default
from ..error import EntityExistsError
from .entity import SmartSimEntity
from .files import EntityFiles


class Model(SmartSimEntity):
    def __init__(self, name, params, path, run_settings, params_as_args=None):
        """Initialize a model entity within Smartsim

        :param name: name of the model
        :type name: str
        :param params: model parameters for writing into configuration files or
                       to be passed as command line arguments to executable.
        :type params: dict
        :param path: path to output, error, and configuration files
        :type path: str
        :param run_settings: launcher settings specified in the experiment
        :type run_settings: RunSettings
        :param params_as_args: list of parameters which have to be
                               interpreted as command line arguments to
                               be added to run_settings
        :type params_as_args: list[str]
        """
        super().__init__(name, path, run_settings)
        self.params = params
        self.params_as_args = params_as_args
        self.incoming_entities = []
        self._key_prefixing_enabled = False
        self.files = None

    @property
    def colocated(self):
        """Return True if this Model will run with a colocated Orchestrator"""
        if self.run_settings.colocated_db_settings:
            return True
        return False

    def register_incoming_entity(self, incoming_entity):
        """Register future communication between entities.

        Registers the named data sources that this entity
        has access to by storing the key_prefix associated
        with that entity

        :param incoming_entity: The entity that data will be received from
        :type incoming_entity: SmartSimEntity
        :raises SmartSimError: if incoming entity has already been registered
        """
        if incoming_entity.name in [
            in_entity.name for in_entity in self.incoming_entities
        ]:
            raise EntityExistsError(
                f"'{incoming_entity.name}' has already "
                + "been registered as an incoming entity"
            )

        self.incoming_entities.append(incoming_entity)

    def enable_key_prefixing(self):
        """If called, the entity will prefix its keys with its own model name"""
        self._key_prefixing_enabled = True

    def disable_key_prefixing(self):
        """If called, the entity will not prefix its keys with its own model name"""
        self._key_prefixing_enabled = False

    def query_key_prefixing(self):
        """Inquire as to whether this entity will prefix its keys with its name"""
        return self._key_prefixing_enabled

    def attach_generator_files(self, to_copy=None, to_symlink=None, to_configure=None):
        """Attach files to an entity for generation

        Attach files needed for the entity that, upon generation,
        will be located in the path of the entity.

        During generation, files "to_copy" are copied into
        the path of the entity, and files "to_symlink" are
        symlinked into the path of the entity.

        Files "to_configure" are text based model input files where
        parameters for the model are set. Note that only models
        support the "to_configure" field. These files must have
        fields tagged that correspond to the values the user
        would like to change. The tag is settable but defaults
        to a semicolon e.g. THERMO = ;10;

        :param to_copy: files to copy, defaults to []
        :type to_copy: list, optional
        :param to_symlink: files to symlink, defaults to []
        :type to_symlink: list, optional
        :param to_configure: input files with tagged parameters, defaults to []
        :type to_configure: list, optional
        """
        to_copy = init_default([], to_copy, (list, str))
        to_symlink = init_default([], to_symlink, (list, str))
        to_configure = init_default([], to_configure, (list, str))
        self.files = EntityFiles(to_configure, to_copy, to_symlink)

    def colocate_db(self, port=6379, db_cpus=1, limit_app_cpus=True, ifname="lo", **kwargs):
        """Colocate an Orchestrator instance with this Model at runtime.

        This method will initialize settings which add an unsharded (not connected)
        database to this Model instance. Only this Model will be able to communicate
        with this colocated database by using the loopback TCP interface or Unix
        Domain sockets (UDS coming soon).

        Extra parameters for the db can be passed through kwargs. This includes
        many performance, caching and inference settings.

        ex. kwargs = {
            maxclients: 100000,
            threads_per_queue: 1,
            inter_op_threads: 1,
            intra_op_threads: 1,
            server-threads: 2 # keydb only
        }

        Generally these don't need to be changed.

        :param port: port to use for orchestrator database, defaults to 6379
        :type port: int, optional
        :param db_cpus: number of cpus to use for orchestrator, defaults to 1
        :type db_cpus: int, optional
        :param limit_app_cpus: whether to limit the number of cpus used by the app, defaults to True
        :type limit_app_cpus: bool, optional
        :param ifname: interface to use for orchestrator, defaults to "lo"
        :type ifname: str, optional
        :param kwargs: additional keyword arguments to pass to the orchestrator database
        :type kwargs: dict, optional

        """
        colo_db_config = {
            "port": int(port),
            "cpus": int(db_cpus),
            "interface": ifname,
            "limit_app_cpus": limit_app_cpus,

            # redisai arguments for inference settings
            "rai_args": {
                "threads_per_queue": kwargs.get("threads_per_queue", None),
                "inter_op_parallelism": kwargs.get("inter_op_parallelism", None),
                "intra_op_parallelism": kwargs.get("intra_op_parallelism", None)
            }
        }
        colo_db_config["extra_db_args"] = dict([
            (k,str(v)) for k,v in kwargs.items() if k not in colo_db_config["rai_args"]
        ])
        self.run_settings.colocated_db_settings = colo_db_config


    def params_to_args(self):
        """Convert parameters to command line arguments and update run settings."""
        for param in self.params_as_args:
            if not param in self.params:
                raise ValueError(
                    f"Tried to convert {param} to command line argument "
                    + f"for Model {self.name}, but its value was not found in model params"
                )
            if self.run_settings is None:
                raise ValueError(
                    f"Tried to configure command line parameter for Model {self.name}, "
                    + "but no RunSettings are set."
                )
            self.run_settings.add_exe_args(cat_arg_and_value(param, self.params[param]))

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

    def __repr__(self):
        return self.name

    def __str__(self):
        entity_str = "Name: " + self.name + "\n"
        entity_str += "Type: " + self.type + "\n"
        entity_str += str(self.run_settings)
        return entity_str
