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

from __future__ import annotations

import copy
import itertools
import re
import sys
import typing as t
import warnings
from os import getcwd
from os import path as osp

from .._core._install.builder import Device
from .._core.utils.helpers import cat_arg_and_value, expand_exe_path
from ..error import EntityExistsError, SSUnsupportedError
from ..log import get_logger
from .dbobject import FSModel, FSScript
from .entity import SmartSimEntity
from .files import EntityFiles

if t.TYPE_CHECKING:
    from smartsim.types import TODO

    RunSettings = TODO
    BatchSettings = TODO


logger = get_logger(__name__)


# TODO: Remove this supression when we strip fileds/functionality
#       (run-settings/batch_settings/params_as_args/etc)!
# pylint: disable-next=too-many-public-methods
class Application(SmartSimEntity):
    def __init__(
        self,
        name: str,
        exe: str,
        run_settings: "RunSettings",
        params: t.Optional[t.Dict[str, str]] = None,
        exe_args: t.Optional[t.List[str]] = None,
        params_as_args: t.Optional[t.List[str]] = None,
        batch_settings: t.Optional["BatchSettings"] = None,
        files: t.Optional[EntityFiles] = None,
    ):
        """Initialize a ``Application``

        :param name: name of the application
        :param exe: executable to run
        :param exe_args: executable arguments
        :param params: application parameters for writing into configuration files or
                       to be passed as command line arguments to executable.
        :param path: path to output, error, and configuration files
        :param run_settings: launcher settings specified in the experiment
        :param params_as_args: list of parameters which have to be
                               interpreted as command line arguments to
                               be added to run_settings
        :param batch_settings: Launcher settings for running the individual
                               application as a batch job
        :param files: Files to have available to the application
        """
        super().__init__(name, run_settings)
        self.exe = [expand_exe_path(exe)]
        # self.exe = [exe] if run_settings.container else [expand_exe_path(exe)]
        self.exe_args = exe_args or []
        self.params = params.copy() if params else {}
        self.params_as_args = params_as_args
        self.incoming_entities: t.List[SmartSimEntity] = []
        self._key_prefixing_enabled = False
        self.batch_settings = batch_settings
        self._fs_models: t.List[FSModel] = []
        self._fs_scripts: t.List[FSScript] = []
        self.files = copy.deepcopy(files) if files else None

    @property
    def exe_args(self) -> t.Union[str, t.List[str]]:
        """Return an immutable list of attached executable arguments.

        :returns: attached executable arguments
        """
        return self._exe_args

    @exe_args.setter
    def exe_args(self, value: t.Union[str, t.List[str], None]) -> None:
        """Set the executable arguments.

        :param value: executable arguments
        """
        self._exe_args = self._build_exe_args(value)

    @property
    def fs_models(self) -> t.Iterable[FSModel]:
        """Retrieve an immutable collection of attached models

        :return: Return an immutable collection of attached models
        """
        return (model for model in self._fs_models)

    @property
    def fs_scripts(self) -> t.Iterable[FSScript]:
        """Retrieve an immutable collection attached of scripts

        :return: Return an immutable collection of attached scripts
        """
        return (script for script in self._fs_scripts)

    @property
    def colocated(self) -> bool:
        """Return True if this Model will run with a colocated FeatureStore

        :return: Return True of the Model will run with a colocated FeatureStore
        """
        if self.run_settings is None:
            return False
        return bool(self.run_settings.colocated_fs_settings)

    def add_exe_args(self, args: t.Union[str, t.List[str]]) -> None:
        """Add executable arguments to executable

        :param args: executable arguments
        """
        args = self._build_exe_args(args)
        self._exe_args.extend(args)

    def register_incoming_entity(self, incoming_entity: SmartSimEntity) -> None:
        """Register future communication between entities.

        Registers the named data sources that this entity
        has access to by storing the key_prefix associated
        with that entity

        :param incoming_entity: The entity that data will be received from
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
        """If called, the entity will prefix its keys with its own application name"""
        self._key_prefixing_enabled = True

    def disable_key_prefixing(self) -> None:
        """If called, the entity will not prefix its keys with its own
        application name
        """
        self._key_prefixing_enabled = False

    def query_key_prefixing(self) -> bool:
        """Inquire as to whether this entity will prefix its keys with its name

        :return: Return True if entity will prefix its keys with its name
        """
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

        Files "to_configure" are text based application input files where
        parameters for the application are set. Note that only applications
        support the "to_configure" field. These files must have
        fields tagged that correspond to the values the user
        would like to change. The tag is settable but defaults
        to a semicolon e.g. THERMO = ;10;

        :param to_copy: files to copy
        :param to_symlink: files to symlink
        :param to_configure: input files with tagged parameters
        """
        to_copy = to_copy or []
        to_symlink = to_symlink or []
        to_configure = to_configure or []

        # Check that no file collides with the parameter file written
        # by Generator. We check the basename, even though it is more
        # restrictive than what we need (but it avoids relative path issues)
        for strategy in [to_copy, to_symlink, to_configure]:
            if strategy is not None and any(
                osp.basename(filename) == "smartsim_params.txt" for filename in strategy
            ):
                raise ValueError(
                    "`smartsim_params.txt` is a file automatically "
                    + "generated by SmartSim and cannot be ovewritten."
                )

        self.files = EntityFiles(to_configure, to_copy, to_symlink)

    @property
    def attached_files_table(self) -> str:
        """Return a list of attached files as a plain text table

        :returns: String version of table
        """
        if not self.files:
            return "No file attached to this application."
        return str(self.files)

    def print_attached_files(self) -> None:
        """Print a table of the attached files on std out"""
        print(self.attached_files_table)

    def colocate_fs(self, *args: t.Any, **kwargs: t.Any) -> None:
        """An alias for ``Application.colocate_fs_tcp``"""
        warnings.warn(
            (
                "`colocate_fs` has been deprecated and will be removed in a \n"
                "future release. Please use `colocate_fs_tcp` or `colocate_fs_uds`."
            ),
            FutureWarning,
        )
        self.colocate_fs_tcp(*args, **kwargs)

    def colocate_fs_uds(
        self,
        unix_socket: str = "/tmp/redis.socket",
        socket_permissions: int = 755,
        fs_cpus: int = 1,
        custom_pinning: t.Optional[t.Iterable[t.Union[int, t.Iterable[int]]]] = None,
        debug: bool = False,
        fs_identifier: str = "",
        **kwargs: t.Any,
    ) -> None:
        """Colocate an FeatureStore instance with this Application over UDS.

        This method will initialize settings which add an unsharded feature
        store to this Application instance. Only this Application will be able
        to communicate with this colocated feature store by using Unix Domain
        sockets.

        Extra parameters for the fs can be passed through kwargs. This includes
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
        :param socket_permissions: permissions for the socketfile
        :param fs_cpus: number of cpus to use for FeatureStore
        :param custom_pinning: CPUs to pin the FeatureStore to. Passing an empty
                               iterable disables pinning
        :param debug: launch Application with extra debug information about the
                      colocated fs
        :param kwargs: additional keyword arguments to pass to the FeatureStore
                       feature store
        """

        if not re.match(r"^[a-zA-Z0-9.:\,_\-/]*$", unix_socket):
            raise ValueError(
                f"Invalid name for unix socket: {unix_socket}. Must only "
                "contain alphanumeric characters or . : _ - /"
            )
        uds_options: t.Dict[str, t.Union[int, str]] = {
            "unix_socket": unix_socket,
            "socket_permissions": socket_permissions,
            # This is hardcoded to 0 as recommended by redis for UDS
            "port": 0,
        }

        common_options = {
            "cpus": fs_cpus,
            "custom_pinning": custom_pinning,
            "debug": debug,
            "fs_identifier": fs_identifier,
        }
        self._set_colocated_fs_settings(uds_options, common_options, **kwargs)

    def colocate_fs_tcp(
        self,
        port: int = 6379,
        ifname: t.Union[str, list[str]] = "lo",
        fs_cpus: int = 1,
        custom_pinning: t.Optional[t.Iterable[t.Union[int, t.Iterable[int]]]] = None,
        debug: bool = False,
        fs_identifier: str = "",
        **kwargs: t.Any,
    ) -> None:
        """Colocate an FeatureStore instance with this Application over TCP/IP.

        This method will initialize settings which add an unsharded feature
        store to this Application instance. Only this Application will be able
        to communicate with this colocated feature store by using the loopback
        TCP interface.

        Extra parameters for the fs can be passed through kwargs. This includes
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

        :param port: port to use for FeatureStore feature store
        :param ifname: interface to use for FeatureStore
        :param fs_cpus: number of cpus to use for FeatureStore
        :param custom_pinning: CPUs to pin the FeatureStore to. Passing an empty
                               iterable disables pinning
        :param debug: launch Application with extra debug information about the
                      colocated fs
        :param kwargs: additional keyword arguments to pass to the FeatureStore
                       feature store
        """

        tcp_options = {"port": port, "ifname": ifname}
        common_options = {
            "cpus": fs_cpus,
            "custom_pinning": custom_pinning,
            "debug": debug,
            "fs_identifier": fs_identifier,
        }
        self._set_colocated_fs_settings(tcp_options, common_options, **kwargs)

    def _set_colocated_fs_settings(
        self,
        connection_options: t.Mapping[str, t.Union[int, t.List[str], str]],
        common_options: t.Dict[
            str,
            t.Union[
                t.Union[t.Iterable[t.Union[int, t.Iterable[int]]], None],
                bool,
                int,
                str,
                None,
            ],
        ],
        **kwargs: t.Union[int, None],
    ) -> None:
        """
        Ingest the connection-specific options (UDS/TCP) and set the final settings
        for the colocated feature store
        """

        if hasattr(self.run_settings, "mpmd") and len(self.run_settings.mpmd) > 0:
            raise SSUnsupportedError(
                "Applications colocated with feature stores cannot be run as a "
                "mpmd workload"
            )

        if hasattr(self.run_settings, "_prep_colocated_fs"):
            # pylint: disable-next=protected-access
            self.run_settings._prep_colocated_fs(common_options["cpus"])

        if "limit_app_cpus" in kwargs:
            raise SSUnsupportedError(
                "Pinning app CPUs via limit_app_cpus is not supported. Modify "
                "RunSettings using the correct binding option for your launcher."
            )

        # TODO list which fs settings can be extras
        custom_pinning_ = t.cast(
            t.Optional[t.Iterable[t.Union[int, t.Iterable[int]]]],
            common_options.get("custom_pinning"),
        )
        cpus_ = t.cast(int, common_options.get("cpus"))
        common_options["custom_pinning"] = self._create_pinning_string(
            custom_pinning_, cpus_
        )

        colo_fs_config: t.Dict[
            str,
            t.Union[
                bool,
                int,
                str,
                None,
                t.List[str],
                t.Iterable[t.Union[int, t.Iterable[int]]],
                t.List[FSModel],
                t.List[FSScript],
                t.Dict[str, t.Union[int, None]],
                t.Dict[str, str],
            ],
        ] = {}
        colo_fs_config.update(connection_options)
        colo_fs_config.update(common_options)

        redis_ai_temp = {
            "threads_per_queue": kwargs.get("threads_per_queue", None),
            "inter_op_parallelism": kwargs.get("inter_op_parallelism", None),
            "intra_op_parallelism": kwargs.get("intra_op_parallelism", None),
        }
        # redisai arguments for inference settings
        colo_fs_config["rai_args"] = redis_ai_temp
        colo_fs_config["extra_fs_args"] = {
            k: str(v) for k, v in kwargs.items() if k not in redis_ai_temp
        }

        self._check_fs_objects_colo()
        colo_fs_config["fs_models"] = self._fs_models
        colo_fs_config["fs_scripts"] = self._fs_scripts

        self.run_settings.colocated_fs_settings = colo_fs_config

    @staticmethod
    def _create_pinning_string(
        pin_ids: t.Optional[t.Iterable[t.Union[int, t.Iterable[int]]]], cpus: int
    ) -> t.Optional[str]:
        """Create a comma-separated string of CPU ids. By default, ``None``
        returns 0,1,...,cpus-1; an empty iterable will disable pinning
        altogether, and an iterable constructs a comma separated string of
        integers (e.g. ``[0, 2, 5]`` -> ``"0,2,5"``)
        """

        def _stringify_id(_id: int) -> str:
            """Return the cPU id as a string if an int, otherwise raise a ValueError"""
            if isinstance(_id, int):
                if _id < 0:
                    raise ValueError("CPU id must be a nonnegative number")
                return str(_id)

            raise TypeError(f"Argument is of type '{type(_id)}' not 'int'")

        try:
            pin_ids = tuple(pin_ids) if pin_ids is not None else None
        except TypeError:
            raise TypeError(
                "Expected a cpu pinning specification of type iterable of ints or "
                f"iterables of ints. Instead got type `{type(pin_ids)}`"
            ) from None

        # Deal with MacOSX limitations first. The "None" (default) disables pinning
        # and is equivalent to []. The only invalid option is a non-empty pinning
        if sys.platform == "darwin":
            if pin_ids:
                warnings.warn(
                    "CPU pinning is not supported on MacOSX. Ignoring pinning "
                    "specification.",
                    RuntimeWarning,
                )
            return None

        # Flatten the iterable into a list and check to make sure that the resulting
        # elements are all ints
        if pin_ids is None:
            return ",".join(_stringify_id(i) for i in range(cpus))
        if not pin_ids:
            return None
        pin_ids = ((x,) if isinstance(x, int) else x for x in pin_ids)
        to_fmt = itertools.chain.from_iterable(pin_ids)
        return ",".join(sorted({_stringify_id(x) for x in to_fmt}))

    def params_to_args(self) -> None:
        """Convert parameters to command line arguments and update run settings."""
        if self.params_as_args is not None:
            for param in self.params_as_args:
                if not param in self.params:
                    raise ValueError(
                        f"Tried to convert {param} to command line argument for "
                        f"application {self.name}, but its value was not found "
                        "in application params"
                    )
                if self.run_settings is None:
                    raise ValueError(
                        "Tried to configure command line parameter for Application "
                        f"{self.name}, but no RunSettings are set."
                    )
                self.add_exe_args(cat_arg_and_value(param, self.params[param]))

    def add_ml_model(
        self,
        name: str,
        backend: str,
        model: t.Optional[bytes] = None,
        model_path: t.Optional[str] = None,
        device: str = Device.CPU.value.upper(),
        devices_per_node: int = 1,
        first_device: int = 0,
        batch_size: int = 0,
        min_batch_size: int = 0,
        min_batch_timeout: int = 0,
        tag: str = "",
        inputs: t.Optional[t.List[str]] = None,
        outputs: t.Optional[t.List[str]] = None,
    ) -> None:
        """A TF, TF-lite, PT, or ONNX model to load into the fs at runtime

        Each ML Model added will be loaded into an
        FeatureStore (converged or not) prior to the execution
        of this Model instance

        One of either model (in memory representation) or model_path (file)
        must be provided

        :param name: key to store model under
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :param model: A model in memory (only supported for non-colocated
                      feature stores)
        :param model_path: serialized model
        :param device: name of device for execution
        :param devices_per_node: The number of GPU devices available on the host.
               This parameter only applies to GPU devices and will be ignored if device
               is specified as CPU.
        :param first_device: The first GPU device to use on the host.
               This parameter only applies to GPU devices and will be ignored if device
               is specified as CPU.
        :param batch_size: batch size for execution
        :param min_batch_size: minimum batch size for model execution
        :param min_batch_timeout: time to wait for minimum batch size
        :param tag: additional tag for model information
        :param inputs: model inputs (TF only)
        :param outputs: model outupts (TF only)
        """
        fs_model = FSModel(
            name=name,
            backend=backend,
            model=model,
            model_file=model_path,
            device=device,
            devices_per_node=devices_per_node,
            first_device=first_device,
            batch_size=batch_size,
            min_batch_size=min_batch_size,
            min_batch_timeout=min_batch_timeout,
            tag=tag,
            inputs=inputs,
            outputs=outputs,
        )
        self.add_ml_model_object(fs_model)

    def add_script(
        self,
        name: str,
        script: t.Optional[str] = None,
        script_path: t.Optional[str] = None,
        device: str = Device.CPU.value.upper(),
        devices_per_node: int = 1,
        first_device: int = 0,
    ) -> None:
        """TorchScript to launch with this Model instance

        Each script added to the application will be loaded into an
        FeatureStore (converged or not) prior to the execution
        of this Model instance

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the script being stored in the first N devices of type ``device``;
        alternatively, setting ``first_device=M`` will result in the script
        being stored on nodes M through M + N - 1.

        One of either script (in memory string representation) or script_path (file)
        must be provided

        :param name: key to store script under
        :param script: TorchScript code (only supported for non-colocated featurestores)
        :param script_path: path to TorchScript code
        :param device: device for script execution
        :param devices_per_node: The number of GPU devices available on the host.
               This parameter only applies to GPU devices and will be ignored if device
               is specified as CPU.
        :param first_device: The first GPU device to use on the host.
               This parameter only applies to GPU devices and will be ignored if device
               is specified as CPU.
        """
        fs_script = FSScript(
            name=name,
            script=script,
            script_path=script_path,
            device=device,
            devices_per_node=devices_per_node,
            first_device=first_device,
        )
        self.add_script_object(fs_script)

    def add_function(
        self,
        name: str,
        function: t.Optional[str] = None,
        device: str = Device.CPU.value.upper(),
        devices_per_node: int = 1,
        first_device: int = 0,
    ) -> None:
        """TorchScript function to launch with this Application instance

        Each script function to the application will be loaded into a
        non-converged FeatureStore prior to the execution
        of this Application instance.

        For converged featurestores, the :meth:`add_script` method should be used.

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the application being stored in the first N devices of type ``device``.

        :param name: key to store function under
        :param function: TorchScript function code
        :param device: device for script execution
        :param devices_per_node: The number of GPU devices available on the host.
               This parameter only applies to GPU devices and will be ignored if device
               is specified as CPU.
        :param first_device: The first GPU device to use on the host.
               This parameter only applies to GPU devices and will be ignored if device
               is specified as CPU.
        """
        fs_script = FSScript(
            name=name,
            script=function,
            device=device,
            devices_per_node=devices_per_node,
            first_device=first_device,
        )
        self.add_script_object(fs_script)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Application):
            return False

        if self.name == other.name:
            return True
        return False

    def __str__(self) -> str:  # pragma: no cover
        entity_str = "Name: " + self.name + "\n"
        entity_str += "Type: " + self.type + "\n"
        entity_str += str(self.run_settings) + "\n"
        if self._fs_models:
            entity_str += "FS Models: \n" + str(len(self._fs_models)) + "\n"
        if self._fs_scripts:
            entity_str += "FS Scripts: \n" + str(len(self._fs_scripts)) + "\n"
        return entity_str

    def add_ml_model_object(self, fs_model: FSModel) -> None:
        if not fs_model.is_file and self.colocated:
            err_msg = (
                "ML model can not be set from memory for colocated feature stores.\n"
            )
            err_msg += (
                f"Please store the ML model named {fs_model.name} in binary format "
            )
            err_msg += "and add it to the SmartSim Application as file."
            raise SSUnsupportedError(err_msg)

        self._fs_models.append(fs_model)

    def add_script_object(self, fs_script: FSScript) -> None:
        if fs_script.func and self.colocated:
            if not isinstance(fs_script.func, str):
                err_msg = (
                    "Functions can not be set from memory for colocated "
                    "feature stores.\n"
                    f"Please convert the function named {fs_script.name} "
                    "to a string or store it as a text file and add it to the "
                    "SmartSim Application with add_script."
                )
                raise SSUnsupportedError(err_msg)
        self._fs_scripts.append(fs_script)

    def _check_fs_objects_colo(self) -> None:
        for fs_model in self._fs_models:
            if not fs_model.is_file:
                err_msg = (
                    "ML model can not be set from memory for colocated "
                    "feature stores.\n"
                    f"Please store the ML model named {fs_model.name} in binary "
                    "format and add it to the SmartSim Application as file."
                )
                raise SSUnsupportedError(err_msg)

        for fs_script in self._fs_scripts:
            if fs_script.func:
                if not isinstance(fs_script.func, str):
                    err_msg = (
                        "Functions can not be set from memory for colocated "
                        "feature stores.\nPlease convert the function named "
                        f"{fs_script.name} to a string or store it as a text"
                        "file and add it to the SmartSim Application with add_script."
                    )
                    raise SSUnsupportedError(err_msg)

    @staticmethod
    def _build_exe_args(exe_args: t.Optional[t.Union[str, t.List[str]]]) -> t.List[str]:
        """Check and convert exe_args input to a desired collection format"""
        if not exe_args:
            return []

        if isinstance(exe_args, list):
            exe_args = copy.deepcopy(exe_args)

        if not (
            isinstance(exe_args, str)
            or (
                isinstance(exe_args, list)
                and all(isinstance(arg, str) for arg in exe_args)
            )
        ):
            raise TypeError("Executable arguments were not a list of str or a str.")

        if isinstance(exe_args, str):
            return exe_args.split()

        return exe_args
