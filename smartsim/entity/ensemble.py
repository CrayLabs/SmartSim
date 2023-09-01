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

import typing as t

from copy import deepcopy
from os import getcwd

from tabulate import tabulate

from .._core.utils.helpers import init_default
from ..error import (
    EntityExistsError,
    SmartSimError,
    SSUnsupportedError,
    UserStrategyError,
)
from ..log import get_logger
from ..settings.base import BatchSettings, RunSettings
from .dbobject import DBModel, DBScript
from .entityList import EntityList
from .model import Model
from .entity import SmartSimEntity
from .strategies import create_all_permutations, random_permutations, step_values

logger = get_logger(__name__)

StrategyFunction = t.Callable[
    [t.List[str], t.List[t.List[str]], int], t.List[t.Dict[str, str]]
]


class Ensemble(EntityList):
    """``Ensemble`` is a group of ``Model`` instances that can
    be treated as a reference to a single instance.
    """

    def __init__(
        self,
        name: str,
        params: t.Dict[str, t.Any],
        params_as_args: t.Optional[t.List[str]] = None,
        batch_settings: t.Optional[BatchSettings] = None,
        run_settings: t.Optional[RunSettings] = None,
        perm_strat: str = "all_perm",
        **kwargs: t.Any,
    ) -> None:
        """Initialize an Ensemble of Model instances.

        The kwargs argument can be used to pass custom input
        parameters to the permutation strategy.

        :param name: name of the ensemble
        :type name: str
        :param params: parameters to expand into ``Model`` members
        :type params: dict[str, Any]
        :param params_as_args: list of params that should be used as command
            line arguments to the ``Model`` member executables and not written
            to generator files
        :type params_as_args: list[str]
        :param batch_settings: describes settings for ``Ensemble`` as batch workload
        :type batch_settings: BatchSettings, optional
        :param run_settings: describes how each ``Model`` should be executed
        :type run_settings: RunSettings, optional
        :param replicas: number of ``Model`` replicas to create - a keyword
            argument of kwargs
        :type replicas: int, optional
        :param perm_strategy: strategy for expanding ``params`` into
                             ``Model`` instances from params argument
                             options are "all_perm", "step", "random"
                             or a callable function. Defaults to "all_perm".
        :type perm_strategy: str
        :return: ``Ensemble`` instance
        :rtype: ``Ensemble``
        """
        self.params = init_default({}, params, dict)
        self.params_as_args = init_default({}, params_as_args, (list, str))
        self._key_prefixing_enabled = True
        self.batch_settings = init_default({}, batch_settings, BatchSettings)
        self.run_settings = init_default({}, run_settings, RunSettings)

        super().__init__(name, getcwd(), perm_strat=perm_strat, **kwargs)

    @property
    def models(self) -> t.Iterable[Model]:
        """
        Helper property to cast self.entities to Model type for type correctness
        """
        model_entities = [node for node in self.entities if isinstance(node, Model)]
        return model_entities

    def _initialize_entities(self, **kwargs: t.Any) -> None:
        """Initialize all the models within the ensemble based
        on the parameters passed to the ensemble and the permutation
        strategy given at init.

        :raises UserStrategyError: if user generation strategy fails
        """
        strategy = self._set_strategy(kwargs.pop("perm_strat"))
        replicas = kwargs.pop("replicas", None)

        # if a ensemble has parameters and run settings, create
        # the ensemble and assign run_settings to each member
        if self.params:
            if self.run_settings:
                param_names, params = self._read_model_parameters()

                # Compute all combinations of model parameters and arguments
                n_models = kwargs.get("n_models", 0)
                all_model_params = strategy(param_names, params, n_models)
                if not isinstance(all_model_params, list):
                    raise UserStrategyError(strategy)

                for i, param_set in enumerate(all_model_params):
                    if not isinstance(param_set, dict):
                        raise UserStrategyError(strategy)
                    run_settings = deepcopy(self.run_settings)
                    model_name = "_".join((self.name, str(i)))
                    model = Model(
                        model_name,
                        param_set,
                        self.path,
                        run_settings=run_settings,
                        params_as_args=self.params_as_args,
                    )
                    model.enable_key_prefixing()
                    model.params_to_args()
                    logger.debug(
                        f"Created ensemble member: {model_name} in {self.name}"
                    )
                    self.add_model(model)
            # cannot generate models without run settings
            else:
                raise SmartSimError(
                    "Ensembles without 'params' or 'replicas' argument to "
                    "expand into members cannot be given run settings"
                )
        else:
            if self.run_settings:
                if replicas:
                    for i in range(replicas):
                        model_name = "_".join((self.name, str(i)))
                        model = Model(
                            model_name,
                            {},
                            self.path,
                            run_settings=deepcopy(self.run_settings),
                        )
                        model.enable_key_prefixing()
                        logger.debug(
                            f"Created ensemble member: {model_name} in {self.name}"
                        )
                        self.add_model(model)
                else:
                    raise SmartSimError(
                        "Ensembles without 'params' or 'replicas' argument to "
                        "expand into members cannot be given run settings"
                    )
            # if no params, no run settings and no batch settings, error because we
            # don't know how to run the ensemble
            elif not self.batch_settings:
                raise SmartSimError(
                    "Ensemble must be provided batch settings or run settings"
                )
            else:
                logger.info("Empty ensemble created for batch launch")

    def add_model(self, model: Model) -> None:
        """Add a model to this ensemble

        :param model: model instance to be added
        :type model: Model
        :raises TypeError: if model is not an instance of ``Model``
        :raises EntityExistsError: if model already exists in this ensemble
        """
        if not isinstance(model, Model):
            raise TypeError(
                f"Argument to add_model was of type {type(model)}, not Model"
            )
        # "in" operator uses model name for __eq__
        if model in self.entities:
            raise EntityExistsError(
                f"Model {model.name} already exists in ensemble {self.name}"
            )

        if self._db_models:
            self._extend_entity_db_models(model, self._db_models)
        if self._db_scripts:
            self._extend_entity_db_scripts(model, self._db_scripts)

        self.entities.append(model)

    def register_incoming_entity(self, incoming_entity: SmartSimEntity) -> None:
        """Register future communication between entities.

        Registers the named data sources that this entity
        has access to by storing the key_prefix associated
        with that entity

        Only python clients can have multiple incoming connections

        :param incoming_entity: The entity that data will be received from
        :type incoming_entity: SmartSimEntity
        """
        for model in self.models:
            model.register_incoming_entity(incoming_entity)

    def enable_key_prefixing(self) -> None:
        """If called, each model within this ensemble will prefix its key with its
        own model name.
        """
        for model in self.models:
            model.enable_key_prefixing()

    def query_key_prefixing(self) -> bool:
        """Inquire as to whether each model within the ensemble will prefix their keys

        :returns: True if all models have key prefixing enabled, False otherwise
        :rtype: bool
        """
        return all(model.query_key_prefixing() for model in self.models)

    def attach_generator_files(
        self,
        to_copy: t.Optional[t.List[str]] = None,
        to_symlink: t.Optional[t.List[str]] = None,
        to_configure: t.Optional[t.List[str]] = None,
    ) -> None:
        """Attach files to each model within the ensemble for generation

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
        for model in self.models:
            model.attach_generator_files(
                to_copy=to_copy, to_symlink=to_symlink, to_configure=to_configure
            )

    @property
    def attached_files_table(self) -> str:
        """Return a plain-text table with information about files
        attached to models belonging to this ensemble.

        :returns: A table of all files attached to all models
        :rtype: str
        """
        if not self.models:
            return "The ensemble is empty, no files to show."

        table = tabulate(
            [[model.name, model.attached_files_table] for model in self.models],
            headers=["Model name", "Files"],
            tablefmt="grid",
        )

        return table

    def print_attached_files(self) -> None:
        """Print table of attached files to std out"""
        print(self.attached_files_table)

    @staticmethod
    def _set_strategy(strategy: str) -> StrategyFunction:
        """Set the permutation strategy for generating models within
        the ensemble

        :param strategy: name of the strategy or callable function
        :type strategy: str
        :raises SSUnsupportedError: if str name is not supported
        :return: strategy function
        :rtype: callable
        """
        if strategy == "all_perm":
            return create_all_permutations
        if strategy == "step":
            return step_values
        if strategy == "random":
            return random_permutations
        if callable(strategy):
            return strategy
        raise SSUnsupportedError(
            f"Permutation strategy given is not supported: {strategy}"
        )

    def _read_model_parameters(self) -> t.Tuple[t.List[str], t.List[t.List[str]]]:
        """Take in the parameters given to the ensemble and prepare to
        create models for the ensemble

        :raises TypeError: if params are of the wrong type
        :return: param names and values for permutation strategy
        :rtype: tuple[list, list]
        """

        if not isinstance(self.params, dict):
            raise TypeError(
                "Ensemble initialization argument 'params' must be of type dict"
            )

        param_names: t.List[str] = []
        parameters: t.List[t.List[str]] = []
        for name, val in self.params.items():
            param_names.append(name)

            if isinstance(val, list):
                val = [str(v) for v in val]
                parameters.append(val)
            elif isinstance(val, (int, str)):
                parameters.append([str(val)])
            else:
                raise TypeError(
                    "Incorrect type for ensemble parameters\n"
                    + "Must be list, int, or string."
                )
        return param_names, parameters

    def add_ml_model(
        self,
        name: str,
        backend: str,
        model: t.Optional[str] = None,
        model_path: t.Optional[str] = None,
        device: t.Literal["CPU", "GPU"] = "CPU",
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
        of every entity belonging to this ensemble

        One of either model (in memory representation) or model_path (file)
        must be provided

        :param name: key to store model under
        :type name: str
        :param model: model in memory
        :type model: str | bytes | None
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
        self._db_models.append(db_model)
        for entity in self.models:
            self._extend_entity_db_models(entity, [db_model])

    def add_script(
        self,
        name: str,
        script: t.Optional[str] = None,
        script_path: t.Optional[str] = None,
        device: t.Literal["CPU", "GPU"] = "CPU",
        devices_per_node: int = 1,
    ) -> None:
        """TorchScript to launch with every entity belonging to this ensemble

        Each script added to the model will be loaded into an
        orchestrator (converged or not) prior to the execution
        of every entity belonging to this ensemble

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
        self._db_scripts.append(db_script)
        for entity in self.models:
            self._extend_entity_db_scripts(entity, [db_script])

    def add_function(
        self,
        name: str,
        function: t.Optional[str] = None,
        device: t.Literal["CPU", "GPU"] = "CPU",
        devices_per_node: int = 1,
    ) -> None:
        """TorchScript function to launch with every entity belonging to this ensemble

        Each script function to the model will be loaded into a
        non-converged orchestrator prior to the execution
        of every entity belonging to this ensemble.

        For converged orchestrators, the :meth:`add_script` method should be used.

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the model being stored in the first N devices of type ``device``.

        :param name: key to store function under
        :type name: str
        :param function: TorchScript code
        :type function: str, optional
        :param device: device for script execution, defaults to "CPU"
        :type device: str, optional
        :param devices_per_node: number of devices on each host
        :type devices_per_node: int
        """
        db_script = DBScript(
            name=name, script=function, device=device, devices_per_node=devices_per_node
        )
        self._db_scripts.append(db_script)
        for entity in self.models:
            self._extend_entity_db_scripts(entity, [db_script])

    @staticmethod
    def _extend_entity_db_models(model: Model, db_models: t.List[DBModel]) -> None:
        entity_db_models = [db_model.name for db_model in model.db_models]

        for db_model in db_models:
            if db_model.name not in entity_db_models:
                model.add_ml_model_object(db_model)

    @staticmethod
    def _extend_entity_db_scripts(model: Model, db_scripts: t.List[DBScript]) -> None:
        entity_db_scripts = [db_script.name for db_script in model.db_scripts]
        for db_script in db_scripts:
            if not db_script.name in entity_db_scripts:
                model.add_script_object(db_script)
