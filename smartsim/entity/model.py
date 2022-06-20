# BSD 2-Clause License
#
# Copyright (c) 2021-2022, Hewlett Packard Enterprise
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
from ..error import EntityExistsError, SSUnsupportedError
from .dbobject import DBModel, DBScript
from .entity import SmartSimEntity
from .files import EntityFiles


class Model(SmartSimEntity):
    def __init__(self, name, params, path, run_settings, params_as_args=None):
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
        """
        super().__init__(name, path, run_settings)
        self.params = params
        self.params_as_args = params_as_args
        self.incoming_entities = []
        self._key_prefixing_enabled = False
        self._db_models = []
        self._db_scripts = []
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

    def colocate_db(
        self,
        port=6379,
        db_cpus=1,
        limit_app_cpus=True,
        ifname="lo",
        debug=False,
        **kwargs,
    ):
        """Colocate an Orchestrator instance with this Model at runtime.

        This method will initialize settings which add an unsharded (not connected)
        database to this Model instance. Only this Model will be able to communicate
        with this colocated database by using the loopback TCP interface or Unix
        Domain sockets (UDS coming soon).

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
        :param db_cpus: number of cpus to use for orchestrator, defaults to 1
        :type db_cpus: int, optional
        :param limit_app_cpus: whether to limit the number of cpus used by the app, defaults to True
        :type limit_app_cpus: bool, optional
        :param ifname: interface to use for orchestrator, defaults to "lo"
        :type ifname: str, optional
        :param debug: launch Model with extra debug information about the co-located db
        :type debug: bool, optional
        :param kwargs: additional keyword arguments to pass to the orchestrator database
        :type kwargs: dict, optional

        """
        if hasattr(self.run_settings, "mpmd") and len(self.run_settings.mpmd) > 0:
            raise SSUnsupportedError(
                "Models co-located with databases cannot be run as a mpmd workload"
            )

        if hasattr(self.run_settings, "_prep_colocated_db"):
            self.run_settings._prep_colocated_db(db_cpus)

        # TODO list which db settings can be extras
        colo_db_config = {
            "port": int(port),
            "cpus": int(db_cpus),
            "interface": ifname,
            "limit_app_cpus": limit_app_cpus,
            "debug": debug,
            # redisai arguments for inference settings
            "rai_args": {
                "threads_per_queue": kwargs.get("threads_per_queue", None),
                "inter_op_parallelism": kwargs.get("inter_op_parallelism", None),
                "intra_op_parallelism": kwargs.get("intra_op_parallelism", None),
            },
        }
        colo_db_config["extra_db_args"] = dict(
            [
                (k, str(v))
                for k, v in kwargs.items()
                if k not in colo_db_config["rai_args"]
            ]
        )

        self._check_db_objects_colo()
        colo_db_config["db_models"] = self._db_models
        colo_db_config["db_scripts"] = self._db_scripts

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

    def add_ml_model(
        self,
        name,
        backend,
        model=None,
        model_path=None,
        device="CPU",
        devices_per_node=1,
        batch_size=0,
        min_batch_size=0,
        tag="",
        inputs=None,
        outputs=None,
    ):
        """A TF, TF-lite, PT, or ONNX model to load into the DB at runtime

        Each ML Model added will be loaded into an
        orchestrator (converged or not) prior to the execution
        of this Model instance

        One of either model (in memory representation) or model_path (file)
        must be provided

        :param name: key to store model under
        :type name: str
        :param model: model in memory
        :type model: byte string, optional
        :param model_path: serialized model
        :type model_path: file path to model
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :type backend: str
        :param device: name of device for execution, defaults to "CPU"
        :type device: str, optional
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
        self._append_db_model(db_model)

    def add_script(
        self, name, script=None, script_path=None, device="CPU", devices_per_node=1
    ):
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
        :param script: TorchScript code
        :type script: str, optional
        :param script_path: path to TorchScript code
        :type script_path: str, optional
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :param devices_per_node: number of devices on each host
        :type devices_per_node: int
        """
        db_script = DBScript(
            name=name,
            script=script,
            script_path=script_path,
            device=device,
            devices_per_node=devices_per_node,
        )
        self._append_db_script(db_script)

    def add_function(self, name, function=None, device="CPU", devices_per_node=1):
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
        :param script: TorchScript code
        :type script: str or byte string, optional
        :param script_path: path to TorchScript code
        :type script_path: str, optional
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :param devices_per_node: number of devices on each host
        :type devices_per_node: int
        """
        db_script = DBScript(
            name=name, script=function, device=device, devices_per_node=devices_per_node
        )
        self._append_db_script(db_script)

    def __eq__(self, other):
        if self.name == other.name:
            return True
        return False

    def __str__(self):  # pragma: no cover
        entity_str = "Name: " + self.name + "\n"
        entity_str += "Type: " + self.type + "\n"
        entity_str += str(self.run_settings) + "\n"
        if self._db_models:
            entity_str += "DB Models: \n" + str(len(self._db_models)) + "\n"
        if self._db_scripts:
            entity_str += "DB Scripts: \n" + str(len(self._db_scripts)) + "\n"
        return entity_str

    def _append_db_model(self, db_model):
        if not db_model.is_file and self.colocated:
            err_msg = "ML model can not be set from memory for colocated databases.\n"
            err_msg += (
                f"Please store the ML model named {db_model.name} in binary format "
            )
            err_msg += "and add it to the SmartSim Model as file."
            raise SSUnsupportedError(err_msg)

        self._db_models.append(db_model)

    def _append_db_script(self, db_script):
        if db_script.func and self.colocated:
            if not isinstance(db_script.func, str):
                err_msg = (
                    "Functions can not be set from memory for colocated databases.\n"
                )
                err_msg += f"Please convert the function named {db_script.name} to a string or store "
                err_msg += "it as a text file and add it to the SmartSim Model with add_script."
                raise SSUnsupportedError(err_msg)
        self._db_scripts.append(db_script)

    def _check_db_objects_colo(self):

        for db_model in self._db_models:
            if not db_model.is_file:
                err_msg = (
                    "ML model can not be set from memory for colocated databases.\n"
                )
                err_msg += (
                    f"Please store the ML model named {db_model.name} in binary format "
                )
                err_msg += "and add it to the SmartSim Model as file."
                raise SSUnsupportedError(err_msg)

        for db_script in self._db_scripts:
            if db_script.func:
                if not isinstance(db_script.func, str):
                    err_msg = "Functions can not be set from memory for colocated databases.\n"
                    err_msg += f"Please convert the function named {db_script.name} to a string or store it "
                    err_msg += "as a text file and add it to the SmartSim Model with add_script."
                    raise SSUnsupportedError(err_msg)
