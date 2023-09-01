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

from __future__ import annotations

import collections.abc
import sys
import typing as t
import warnings

from .._core.utils.helpers import cat_arg_and_value, init_default
from ..error import EntityExistsError, SSUnsupportedError
from .dbobject import DBModel, DBScript
from .entity import SmartSimEntity
from .files import EntityFiles
from ..settings.base import BatchSettings, RunSettings
from ..log import get_logger


logger = get_logger(__name__)

class Model(SmartSimEntity):
    def __init__(
        self,
        name: str,
        params: t.Dict[str, str],
        path: str,
        run_settings: RunSettings,
        params_as_args: t.Optional[t.List[str]] = None,
        batch_settings: t.Optional[BatchSettings] = None,
    ):
        """Initialize a ``Model``

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
        :param batch_settings: Launcher settings for running the individual
                               model as a batch job, defaults to None
        :type batch_settings: BatchSettings | None
        """
        super().__init__(name, path, run_settings)
        self.params = params
        self.params_as_args = params_as_args
        self.incoming_entities: t.List[SmartSimEntity] = []
        self._key_prefixing_enabled = False
        self.batch_settings = batch_settings
        self._db_models: t.List[DBModel] = []
        self._db_scripts: t.List[DBScript] = []
        self.files: t.Optional[EntityFiles] = None

    @property
    def db_models(self) -> t.Iterable[DBModel]:
        """Return an immutable collection of attached models"""
        return (model for model in self._db_models)

    @property
    def db_scripts(self) -> t.Iterable[DBScript]:
        """Return an immutable collection attached of scripts"""
        return (script for script in self._db_scripts)

    @property
    def colocated(self) -> bool:
        """Return True if this Model will run with a colocated Orchestrator"""
        return bool(self.run_settings.colocated_db_settings)

    def register_incoming_entity(self, incoming_entity: SmartSimEntity) -> None:
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

    def enable_key_prefixing(self) -> None:
        """If called, the entity will prefix its keys with its own model name"""
        self._key_prefixing_enabled = True

    def disable_key_prefixing(self) -> None:
        """If called, the entity will not prefix its keys with its own model name"""
        self._key_prefixing_enabled = False

    def query_key_prefixing(self) -> bool:
        """Inquire as to whether this entity will prefix its keys with its name"""
        return self._key_prefixing_enabled

    def attach_generator_files(
        self,
        to_copy: t.Optional[t.List[str]] = None,
        to_symlink: t.Optional[t.List[str]] = None,
        to_configure: t.Optional[t.List[str]] = None,
    ) -> None:
        """Attach files to an entity for generation

        Attach files needed for the entity that, upon generation,
        will be located in the path of the entity.  Invoking this method
        after files have already been attached will overwrite
        the previous list of entity files.

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

    @property
    def attached_files_table(self) -> str:
        """Return a list of attached files as a plain text table

        :returns: String version of table
        :rtype: str
        """
        if not self.files:
            return "No file attached to this model."
        return str(self.files)

    def print_attached_files(self) -> None:
        """Print a table of the attached files on std out
        """
        print(self.attached_files_table)

    def colocate_db(self, *args: t.Any, **kwargs: t.Any) -> None:
        """An alias for ``Model.colocate_db_tcp``"""
        warnings.warn(
            (
                "`colocate_db` has been deprecated and will be removed in a \n"
                "future release. Please use `colocate_db_tcp` or `colocate_db_uds`."
            ), FutureWarning
        )
        self.colocate_db_tcp(*args, **kwargs)

    def colocate_db_uds(
        self,
        unix_socket: str = "/tmp/redis.socket",
        socket_permissions: int = 755,
        db_cpus: int = 1,
        custom_pinning: t.Optional[t.Iterable[t.Union[int, t.Iterable[int]]]] = None,
        debug: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """Colocate an Orchestrator instance with this Model over UDS.

        This method will initialize settings which add an unsharded
        database to this Model instance. Only this Model will be able to communicate
        with this colocated database by using Unix Domain sockets.

        Extra parameters for the db can be passed through kwargs. This includes
        many performance, caching and inference settings.

        .. highlight:: python
        .. code-block:: python

            example_kwargs = {
                "maxclients": 100000,
                "threads_per_queue": 1,
                "inter_op_threads": 1,
                "intra_op_threads": 1,
                "server_threads": 2 # keydb only
            }

        Generally these don't need to be changed.

        :param unix_socket: path to where the socket file will be created
        :type unix_socket: str, optional
        :param socket_permissions: permissions for the socketfile
        :type socket_permissions: int, optional
        :param db_cpus: number of cpus to use for orchestrator, defaults to 1
        :type db_cpus: int, optional
        :param custom_pinning: CPUs to pin the orchestrator to. Passing an empty
                               iterable disables pinning
        :type custom_pinning: iterable of ints or iterable of ints, optional
        :param debug: launch Model with extra debug information about the colocated db
        :type debug: bool, optional
        :param kwargs: additional keyword arguments to pass to the orchestrator database
        :type kwargs: dict, optional
        """

        uds_options = {
            "unix_socket": unix_socket,
            "socket_permissions": socket_permissions,
            "port": 0,  # This is hardcoded to 0 as recommended by redis for UDS
        }

        common_options = {
            "cpus": db_cpus,
            "custom_pinning": custom_pinning,
            "debug": debug,
        }
        self._set_colocated_db_settings(uds_options, common_options, **kwargs)

    def colocate_db_tcp(
        self,
        port: int = 6379,
        ifname: t.Union[str, list[str]] = "lo",
        db_cpus: int = 1,
        custom_pinning: t.Optional[t.Iterable[t.Union[int, t.Iterable[int]]]] = None,
        debug: bool = False,
        **kwargs: t.Any,
    ) -> None:
        """Colocate an Orchestrator instance with this Model over TCP/IP.

        This method will initialize settings which add an unsharded
        database to this Model instance. Only this Model will be able to communicate
        with this colocated database by using the loopback TCP interface.

        Extra parameters for the db can be passed through kwargs. This includes
        many performance, caching and inference settings.

        .. highlight:: python
        .. code-block:: python

            ex. kwargs = {
                maxclients: 100000,
                threads_per_queue: 1,
                inter_op_threads: 1,
                intra_op_threads: 1,
                server_threads: 2 # keydb only
            }

        Generally these don't need to be changed.

        :param port: port to use for orchestrator database, defaults to 6379
        :type port: int, optional
        :param ifname: interface to use for orchestrator, defaults to "lo"
        :type ifname: str | list[str], optional
        :param db_cpus: number of cpus to use for orchestrator, defaults to 1
        :type db_cpus: int, optional
        :param custom_pinning: CPUs to pin the orchestrator to. Passing an empty
                               iterable disables pinning
        :type custom_pinning: iterable of ints or iterable of ints, optional
        :param debug: launch Model with extra debug information about the colocated db
        :type debug: bool, optional
        :param kwargs: additional keyword arguments to pass to the orchestrator database
        :type kwargs: dict, optional

        """

        tcp_options = {"port": port, "ifname": ifname}
        common_options = {
            "cpus": db_cpus,
            "custom_pinning": custom_pinning,
            "debug": debug,
        }
        self._set_colocated_db_settings(tcp_options, common_options, **kwargs)

    def _set_colocated_db_settings(
        self,
        connection_options: t.Dict[str, t.Any],
        common_options: t.Dict[str, t.Any],
        **kwargs: t.Any,
    ) -> None:
        """
        Ingest the connection-specific options (UDS/TCP) and set the final settings
        for the colocated database
        """

        if hasattr(self.run_settings, "mpmd") and len(self.run_settings.mpmd) > 0:
            raise SSUnsupportedError(
                "Models colocated with databases cannot be run as a mpmd workload"
            )

        if hasattr(self.run_settings, "_prep_colocated_db"):
            # pylint: disable-next=protected-access
            self.run_settings._prep_colocated_db(common_options["cpus"])

        if "limit_app_cpus" in kwargs:
            raise SSUnsupportedError(
                "Pinning app CPUs via limit_app_cpus is not supported. Modify "
                "RunSettings using the correct binding option for your launcher."
            )

        # TODO list which db settings can be extras
        common_options["custom_pinning"] = self._create_pinning_string(
            common_options["custom_pinning"],
            common_options["cpus"]
        )

        colo_db_config = {}
        colo_db_config.update(connection_options)
        colo_db_config.update(common_options)
        # redisai arguments for inference settings
        colo_db_config["rai_args"] = {
            "threads_per_queue": kwargs.get("threads_per_queue", None),
            "inter_op_parallelism": kwargs.get("inter_op_parallelism", None),
            "intra_op_parallelism": kwargs.get("intra_op_parallelism", None),
        }
        colo_db_config["extra_db_args"] = {
            k: str(v) for k, v in kwargs.items() if k not in colo_db_config["rai_args"]
        }

        self._check_db_objects_colo()
        colo_db_config["db_models"] = self._db_models
        colo_db_config["db_scripts"] = self._db_scripts

        self.run_settings.colocated_db_settings = colo_db_config

    @staticmethod
    def _create_pinning_string(
        pin_ids: t.Optional[t.Iterable[t.Union[int, t.Iterable[int]]]],
        cpus: int
        ) -> t.Optional[str]:
        """Create a comma-separated string CPU ids. By default, None returns
        0,1,...,cpus-1; an empty iterable will disable pinning altogether,
        and an iterable constructs a comma separate string (e.g. 0,2,5)
        """
        def _stringify_id(_id: int) -> str:
            """Return the cPU id as a string if an int, otherwise raise a ValueError"""
            if isinstance(_id, int):
                if _id < 0:
                    raise ValueError("CPU id must be a nonnegative number")
                return str(_id)

            raise TypeError(f"Argument is of type '{type(_id)}' not 'int'")

        _invalid_input_message = (
            "Expected a cpu pinning specification of type iterable of ints or "
            f"iterables of ints. Instead got type `{type(pin_ids)}`"
        )

        # Deal with MacOSX limitations first. The "None" (default) disables pinning
        # and is equivalent to []. The only invalid option is an iterable
        if sys.platform == "darwin":
            if pin_ids is None or not pin_ids:
                return None

            if isinstance(pin_ids, collections.abc.Iterable):
                warnings.warn(
                    "CPU pinning is not supported on MacOSX. Ignoring pinning "
                    "specification.",
                    RuntimeWarning
                )
                return None
            raise TypeError(_invalid_input_message)
        # Flatten the iterable into a list and check to make sure that the resulting
        # elements are all ints
        if pin_ids is None:
            return ','.join(_stringify_id(i) for i in range(cpus))
        if not pin_ids:
            return None
        if isinstance(pin_ids, collections.abc.Iterable):
            pin_list = []
            for pin_id in pin_ids:
                if isinstance(pin_id, collections.abc.Iterable):
                    pin_list.extend([_stringify_id(j) for j in pin_id])
                else:
                    pin_list.append(_stringify_id(pin_id))
            return ','.join(sorted(set(pin_list)))
        raise TypeError(_invalid_input_message)

    def params_to_args(self) -> None:
        """Convert parameters to command line arguments and update run settings."""
        if self.params_as_args is not None:
            for param in self.params_as_args:
                if not param in self.params:
                    raise ValueError(
                        f"Tried to convert {param} to command line argument for Model "
                        f"{self.name}, but its value was not found in model params"
                    )
                if self.run_settings is None:
                    raise ValueError(
                        "Tried to configure command line parameter for Model "
                        f"{self.name}, but no RunSettings are set."
                    )
                self.run_settings.add_exe_args(
                    cat_arg_and_value(param, self.params[param])
                )

    def add_ml_model(
        self,
        name: str,
        backend: str,
        model: t.Optional[str] = None,
        model_path: t.Optional[str] = None,
        device: t.Literal["CPU","GPU"] = "CPU",
        devices_per_node: int = 1,
        batch_size: int = 0,
        min_batch_size: int = 0,
        tag: str = "",
        inputs: t.Optional[t.List[str]] = None,
        outputs: t.Optional[t.List[str]] = None,
    ) -> None:
        """A TF, TF-lite, PT, or ONNX model to load into the DB at runtime

        Each ML Model added will be loaded into an
        orchestrator (converged or not) prior to the execution
        of this Model instance

        One of either model (in memory representation) or model_path (file)
        must be provided

        :param name: key to store model under
        :type name: str
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :type backend: str
        :param model: A model in memory (only supported for non-colocated orchestrators)
        :type model: byte string, optional
        :param model_path: serialized model
        :type model_path: file path to model
        :param device: name of device for execution, defaults to "CPU"
        :type device: str, optional
        :param devices_per_node: The number of GPU devices available on the host.
               This parameter only applies to GPU devices and will be ignored if device
               is specified as GPU.
        :type devices_per_node: int
        :param batch_size: batch size for execution, defaults to 0
        :type batch_size: int, optional
        :param min_batch_size: minimum batch size for model execution, defaults to 0
        :type min_batch_size: int, optional
        :param tag: additional tag for model information, defaults to ""
        :type tag: str, optional
        :param inputs: model inputs (TF only), defaults to None
        :type inputs: list[str], optional
        :param outputs: model outupts (TF only), defaults to None
        :type outputs: list[str], optional
        """
        db_model = DBModel(
            name=name,
            backend=backend,
            model=model,
            model_file=model_path,
            device=device,
            devices_per_node=devices_per_node,
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            tag=tag,
            inputs=inputs,
            outputs=outputs,
        )
        self.add_ml_model_object(db_model)

    def add_script(
        self,
        name: str,
        script: t.Optional[str] = None,
        script_path: t.Optional[str] = None,
        device: t.Literal["CPU","GPU"] = "CPU",
        devices_per_node: int = 1,
    ) -> None:
        """TorchScript to launch with this Model instance

        Each script added to the model will be loaded into an
        orchestrator (converged or not) prior to the execution
        of this Model instance

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the model being stored in the first N devices of type ``device``.

        One of either script (in memory string representation) or script_path (file)
        must be provided

        :param name: key to store script under
        :type name: str
        :param script: TorchScript code (only supported for non-colocated orchestrators)
        :type script: str, optional
        :param script_path: path to TorchScript code
        :type script_path: str, optional
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :param devices_per_node: The number of GPU devices available on the host.
               This parameter only applies to GPU devices and will be ignored if device
               is specified as GPU.
        :type devices_per_node: int
        """
        db_script = DBScript(
            name=name,
            script=script,
            script_path=script_path,
            device=device,
            devices_per_node=devices_per_node,
        )
        self.add_script_object(db_script)

    def add_function(
        self,
        name: str,
        function: t.Optional[str] = None,
        device: t.Literal["CPU","GPU"] = "CPU",
        devices_per_node: int = 1,
    ) -> None:
        """TorchScript function to launch with this Model instance

        Each script function to the model will be loaded into a
        non-converged orchestrator prior to the execution
        of this Model instance.

        For converged orchestrators, the :meth:`add_script` method should be used.

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the model being stored in the first N devices of type ``device``.

        :param name: key to store function under
        :type name: str
        :param function: TorchScript function code
        :type function: str, optional
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :param devices_per_node: The number of GPU devices available on the host.
               This parameter only applies to GPU devices and will be ignored if device
               is specified as GPU.
        :type devices_per_node: int
        """
        db_script = DBScript(
            name=name, script=function, device=device, devices_per_node=devices_per_node
        )
        self.add_script_object(db_script)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Model):
            return False

        if self.name == other.name:
            return True
        return False

    def __str__(self) -> str:  # pragma: no cover
        entity_str = "Name: " + self.name + "\n"
        entity_str += "Type: " + self.type + "\n"
        entity_str += str(self.run_settings) + "\n"
        if self._db_models:
            entity_str += "DB Models: \n" + str(len(self._db_models)) + "\n"
        if self._db_scripts:
            entity_str += "DB Scripts: \n" + str(len(self._db_scripts)) + "\n"
        return entity_str

    def add_ml_model_object(self, db_model: DBModel) -> None:
        if not db_model.is_file and self.colocated:
            err_msg = "ML model can not be set from memory for colocated databases.\n"
            err_msg += (
                f"Please store the ML model named {db_model.name} in binary format "
            )
            err_msg += "and add it to the SmartSim Model as file."
            raise SSUnsupportedError(err_msg)

        self._db_models.append(db_model)

    def add_script_object(self, db_script: DBScript) -> None:
        if db_script.func and self.colocated:
            if not isinstance(db_script.func, str):
                err_msg = (
                    "Functions can not be set from memory for colocated databases.\n"
                    f"Please convert the function named {db_script.name} "
                    "to a string or store it as a text file and add it to the "
                    "SmartSim Model with add_script."
                )
                raise SSUnsupportedError(err_msg)
        self._db_scripts.append(db_script)

    def _check_db_objects_colo(self) -> None:
        for db_model in self._db_models:
            if not db_model.is_file:
                err_msg = (
                    "ML model can not be set from memory for colocated databases.\n"
                    f"Please store the ML model named {db_model.name} in binary "
                    "format and add it to the SmartSim Model as file."
                )
                raise SSUnsupportedError(err_msg)

        for db_script in self._db_scripts:
            if db_script.func:
                if not isinstance(db_script.func, str):
                    err_msg = (
                        "Functions can not be set from memory for colocated "
                        "databases.\nPlease convert the function named "
                        f"{db_script.name} to a string or store it as a text"
                        "file and add it to the SmartSim Model with add_script."
                    )
                    raise SSUnsupportedError(err_msg)
