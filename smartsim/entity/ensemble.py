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

import os.path as osp
import typing as t
from copy import deepcopy
from os import getcwd

from tabulate import tabulate

from .._core._install.builder import Device
from ..error import EntityExistsError, SmartSimError, SSUnsupportedError
from ..log import get_logger
from ..settings import BatchSettings, RunSettings
from .dbobject import FSModel, FSScript
from .entity import SmartSimEntity
from .entityList import EntityList
from .model import Application
from .strategies import PermutationStrategyType
from .strategies import resolve as resolve_strategy

logger = get_logger(__name__)


class Ensemble(EntityList[Application]):
    """``Ensemble`` is a group of ``Application`` instances that can
    be treated as a reference to a single instance.
    """

    def __init__(
        self,
        name: str,
        params: t.Optional[t.Dict[str, t.Any]] = None,
        exe: t.Optional[str] = None,
        exe_args: t.Optional[t.List[str]] = None,
        path: t.Optional[str] = getcwd(),
        params_as_args: t.Optional[t.List[str]] = None,
        batch_settings: t.Optional[BatchSettings] = None,
        run_settings: t.Optional[RunSettings] = None,
        perm_strat: t.Union[str, PermutationStrategyType] = "all_perm",
        **kwargs: t.Any,
    ) -> None:
        """Initialize an Ensemble of Application instances.

        The kwargs argument can be used to pass custom input
        parameters to the permutation strategy.

        :param name: name of the ensemble
        :param exe: executable to run
        :param exe_args: executable arguments
        :param params: parameters to expand into ``Application`` members
        :param params_as_args: list of params that should be used as command
            line arguments to the ``Application`` member executables and not written
            to generator files
        :param batch_settings: describes settings for ``Ensemble`` as batch workload
        :param run_settings: describes how each ``Application`` should be executed
        :param replicas: number of ``Application`` replicas to create - a keyword
            argument of kwargs
        :param perm_strategy: strategy for expanding ``params`` into
                             ``Application`` instances from params argument
                             options are "all_perm", "step", "random"
                             or a callable function.
        :return: ``Ensemble`` instance
        """
        self.exe = exe or ""
        self.exe_args = exe_args or []
        self.params = params or {}
        self.params_as_args = params_as_args or []
        self._key_prefixing_enabled = True
        self.batch_settings = batch_settings
        self.run_settings = run_settings
        self.replicas: str

        super().__init__(name, path=str(path), perm_strat=perm_strat, **kwargs)

    @property
    def applications(self) -> t.Collection[Application]:
        """An alias for a shallow copy of the ``entities`` attribute"""
        return list(self.entities)

    def _initialize_entities(
        self,
        *,
        perm_strat: t.Union[str, PermutationStrategyType] = "all_perm",
        **kwargs: t.Any,
    ) -> None:
        """Initialize all the applications within the ensemble based
        on the parameters passed to the ensemble and the permutation
        strategy given at init.

        :raises UserStrategyError: if user generation strategy fails
        """
        strategy = resolve_strategy(perm_strat)
        replicas = kwargs.pop("replicas", None)
        self.replicas = replicas

        # if a ensemble has parameters and run settings, create
        # the ensemble and assign run_settings to each member
        if self.params:
            if self.run_settings and self.exe:
                # Compute all combinations of application parameters and arguments
                n_applications = kwargs.get("n_applications", 0)
                all_application_params = strategy(self.params, n_applications)

                for i, param_set in enumerate(all_application_params):
                    run_settings = deepcopy(self.run_settings)
                    application_name = "_".join((self.name, str(i)))
                    application = Application(
                        name=application_name,
                        exe=self.exe,
                        exe_args=self.exe_args,
                        params=param_set,
                        path=osp.join(self.path, application_name),
                        run_settings=run_settings,
                        params_as_args=self.params_as_args,
                    )
                    application.enable_key_prefixing()
                    application.params_to_args()
                    logger.debug(
                        f"Created ensemble member: {application_name} in {self.name}"
                    )
                    self.add_application(application)
            # cannot generate applications without run settings
            else:
                raise SmartSimError(
                    "Ensembles without 'params' or 'replicas' argument to "
                    "expand into members cannot be given run settings"
                )
        else:
            if self.run_settings and self.exe:
                if replicas:
                    for i in range(replicas):
                        application_name = "_".join((self.name, str(i)))
                        application = Application(
                            name=application_name,
                            params={},
                            exe=self.exe,
                            exe_args=self.exe_args,
                            path=osp.join(self.path, application_name),
                            run_settings=deepcopy(self.run_settings),
                        )
                        application.enable_key_prefixing()
                        logger.debug(
                            "Created ensemble member: "
                            f"{application_name} in {self.name}"
                        )
                        self.add_application(application)
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

    def add_application(self, application: Application) -> None:
        """Add a application to this ensemble

        :param application: application instance to be added
        :raises TypeError: if application is not an instance of ``Application``
        :raises EntityExistsError: if application already exists in this ensemble
        """
        if not isinstance(application, Application):
            raise TypeError(
                f"Argument to add_application was of type {type(application)}, "
                " not Application"
            )
        # "in" operator uses application name for __eq__
        if application in self.entities:
            raise EntityExistsError(
                f"Application {application.name} already exists in ensemble {self.name}"
            )

        if self._fs_models:
            self._extend_entity_fs_models(application, self._fs_models)
        if self._fs_scripts:
            self._extend_entity_fs_scripts(application, self._fs_scripts)

        self.entities.append(application)

    def register_incoming_entity(self, incoming_entity: SmartSimEntity) -> None:
        """Register future communication between entities.

        Registers the named data sources that this entity
        has access to by storing the key_prefix associated
        with that entity

        Only python clients can have multiple incoming connections

        :param incoming_entity: The entity that data will be received from
        """
        for application in self.applications:
            application.register_incoming_entity(incoming_entity)

    def enable_key_prefixing(self) -> None:
        """If called, each application within this ensemble will prefix its key with its
        own application name.
        """
        for application in self.applications:
            application.enable_key_prefixing()

    def query_key_prefixing(self) -> bool:
        """Inquire as to whether each application within the ensemble will
        prefix their keys

        :returns: True if all applications have key prefixing enabled, False otherwise
        """
        return all(
            application.query_key_prefixing() for application in self.applications
        )

    def attach_generator_files(
        self,
        to_copy: t.Optional[t.List[str]] = None,
        to_symlink: t.Optional[t.List[str]] = None,
        to_configure: t.Optional[t.List[str]] = None,
    ) -> None:
        """Attach files to each application within the ensemble for generation

        Attach files needed for the entity that, upon generation,
        will be located in the path of the entity.

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
        for application in self.applications:
            application.attach_generator_files(
                to_copy=to_copy, to_symlink=to_symlink, to_configure=to_configure
            )

    @property
    def attached_files_table(self) -> str:
        """Return a plain-text table with information about files
        attached to applications belonging to this ensemble.

        :returns: A table of all files attached to all applications
        """
        if not self.applications:
            return "The ensemble is empty, no files to show."

        table = tabulate(
            [
                [application.name, application.attached_files_table]
                for application in self.applications
            ],
            headers=["Application name", "Files"],
            tablefmt="grid",
        )

        return table

    def print_attached_files(self) -> None:
        """Print table of attached files to std out"""
        print(self.attached_files_table)

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

        Each ML Model added will be loaded into a
        feature store (converged or not) prior to the execution
        of every entity belonging to this ensemble

        One of either model (in memory representation) or model_path (file)
        must be provided

        :param name: key to store model under
        :param model: model in memory
        :param model_path: serialized model
        :param backend: name of the backend (TORCH, TF, TFLITE, ONNX)
        :param device: name of device for execution
        :param devices_per_node: number of GPUs per node in multiGPU nodes
        :param first_device: first device in multi-GPU nodes to use for execution,
                             defaults to 0; ignored if devices_per_node is 1
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
        dupe = next(
            (
                fs_model.name
                for ensemble_ml_model in self._fs_models
                if ensemble_ml_model.name == fs_model.name
            ),
            None,
        )
        if dupe:
            raise SSUnsupportedError(
                f'An ML Model with name "{fs_model.name}" already exists'
            )
        self._fs_models.append(fs_model)
        for entity in self.applications:
            self._extend_entity_fs_models(entity, [fs_model])

    def add_script(
        self,
        name: str,
        script: t.Optional[str] = None,
        script_path: t.Optional[str] = None,
        device: str = Device.CPU.value.upper(),
        devices_per_node: int = 1,
        first_device: int = 0,
    ) -> None:
        """TorchScript to launch with every entity belonging to this ensemble

        Each script added to the application will be loaded into an
        feature store (converged or not) prior to the execution
        of every entity belonging to this ensemble

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the application being stored in the first N devices of type ``device``.

        One of either script (in memory string representation) or script_path (file)
        must be provided

        :param name: key to store script under
        :param script: TorchScript code
        :param script_path: path to TorchScript code
        :param device: device for script execution
        :param devices_per_node: number of devices on each host
        :param first_device: first device to use on each host
        """
        fs_script = FSScript(
            name=name,
            script=script,
            script_path=script_path,
            device=device,
            devices_per_node=devices_per_node,
            first_device=first_device,
        )
        dupe = next(
            (
                fs_script.name
                for ensemble_script in self._fs_scripts
                if ensemble_script.name == fs_script.name
            ),
            None,
        )
        if dupe:
            raise SSUnsupportedError(
                f'A Script with name "{fs_script.name}" already exists'
            )
        self._fs_scripts.append(fs_script)
        for entity in self.applications:
            self._extend_entity_fs_scripts(entity, [fs_script])

    def add_function(
        self,
        name: str,
        function: t.Optional[str] = None,
        device: str = Device.CPU.value.upper(),
        devices_per_node: int = 1,
        first_device: int = 0,
    ) -> None:
        """TorchScript function to launch with every entity belonging to this ensemble

        Each script function to the application will be loaded into a
        non-converged feature store prior to the execution
        of every entity belonging to this ensemble.

        For converged feature stores, the :meth:`add_script` method should be used.

        Device selection is either "GPU" or "CPU". If many devices are
        present, a number can be passed for specification e.g. "GPU:1".

        Setting ``devices_per_node=N``, with N greater than one will result
        in the script being stored in the first N devices of type ``device``;
        alternatively, setting ``first_device=M`` will result in the script
        being stored on nodes M through M + N - 1.

        :param name: key to store function under
        :param function: TorchScript code
        :param device: device for script execution
        :param devices_per_node: number of devices on each host
        :param first_device: first device to use on each host
        """
        fs_script = FSScript(
            name=name,
            script=function,
            device=device,
            devices_per_node=devices_per_node,
            first_device=first_device,
        )
        dupe = next(
            (
                fs_script.name
                for ensemble_script in self._fs_scripts
                if ensemble_script.name == fs_script.name
            ),
            None,
        )
        if dupe:
            raise SSUnsupportedError(
                f'A Script with name "{fs_script.name}" already exists'
            )
        self._fs_scripts.append(fs_script)
        for entity in self.applications:
            self._extend_entity_fs_scripts(entity, [fs_script])

    @staticmethod
    def _extend_entity_fs_models(
        application: Application, fs_models: t.List[FSModel]
    ) -> None:
        """
        Ensures that the Machine Learning model names being added to the Ensemble
        are unique.

        This static method checks if the provided ML model names already exist in
        the Ensemble. An SSUnsupportedError is raised if any duplicate names are
        found. Otherwise, it appends the given list of FSModel to the Ensemble.

        :param application: SmartSim Application object.
        :param fs_models: List of FSModels to append to the Ensemble.
        """
        for add_ml_model in fs_models:
            dupe = next(
                (
                    fs_model.name
                    for fs_model in application.fs_models
                    if fs_model.name == add_ml_model.name
                ),
                None,
            )
            if dupe:
                raise SSUnsupportedError(
                    f'An ML Model with name "{add_ml_model.name}" already exists'
                )
            application.add_ml_model_object(add_ml_model)

    @staticmethod
    def _extend_entity_fs_scripts(
        application: Application, fs_scripts: t.List[FSScript]
    ) -> None:
        """
        Ensures that the script/function names being added to the Ensemble are unique.

        This static method checks if the provided script/function names already exist
        in the Ensemble. An SSUnsupportedError is raised if any duplicate names
        are found. Otherwise, it appends the given list of FSScripts to the
        Ensemble.

        :param application: SmartSim Application object.
        :param fs_scripts: List of FSScripts to append to the Ensemble.
        """
        for add_script in fs_scripts:
            dupe = next(
                (
                    add_script.name
                    for fs_script in application.fs_scripts
                    if fs_script.name == add_script.name
                ),
                None,
            )
            if dupe:
                raise SSUnsupportedError(
                    f'A Script with name "{add_script.name}" already exists'
                )
            application.add_script_object(add_script)
