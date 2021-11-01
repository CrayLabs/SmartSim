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

from copy import deepcopy
from os import getcwd

from ..error import (
    EntityExistsError,
    SmartSimError,
    SSUnsupportedError,
    UserStrategyError,
)
from ..settings.base import BatchSettings, RunSettings
from ..utils import get_logger
from ..utils.helpers import cat_arg_and_value, init_default
from .entityList import EntityList
from .model import Model
from .strategies import create_all_permutations, random_permutations, step_values

logger = get_logger(__name__)


class Ensemble(EntityList):
    """``Ensemble`` is a group of ``Model`` instances that can
    be treated as a reference to a single instance.
    """

    def __init__(
        self,
        name,
        params,
        arg_params=None,
        batch_settings=None,
        run_settings=None,
        perm_strat="all_perm",
        **kwargs,
    ):
        """Initialize an Ensemble of Model instances.

        The kwargs argument can be used to pass custom input
        parameters to the permutation strategy.

        :param name: name of the ensemble
        :type name: str
        :param params: parameters to expand into ``Model`` members
        :type params: dict[str, Any]
        :param arg_params: command line parameters to expand into ``Model`` members
        :type arg_params: dict[str, Any]
        :param batch_settings: describes settings for ``Ensemble`` as batch workload
        :type batch_settings: BatchSettings, optional
        :param run_settings: describes how each ``Model`` should be executed
        :type run_settings: RunSettings, optional
        :param replicas: number of ``Model`` replicas to create - a keyword argument of kwargs
        :type replicas: int, optional
        :param perm_strategy: strategy for expanding ``params`` into
                             ``Model`` instances from params argument
                             options are "all_perm", "stepped", "random"
                             or a callable function. Defaults to "all_perm".
        :type perm_strategy: str
        :return: ``Ensemble`` instance
        :rtype: ``Ensemble``
        """
        self.params = init_default({}, params, dict)
        self.arg_params = init_default({}, arg_params, dict)
        self._key_prefixing_enabled = True
        self.batch_settings = init_default({}, batch_settings, BatchSettings)
        self.run_settings = init_default({}, run_settings, RunSettings)
        super().__init__(name, getcwd(), perm_strat=perm_strat, **kwargs)

    @property
    def models(self):
        return self.entities

    def _initialize_entities(self, **kwargs):
        """Initialize all the models within the ensemble based
        on the parameters passed to the ensemble and the permutation
        strategy given at init.

        :raises UserStrategyError: if user generation strategy fails
        """
        strategy = self._set_strategy(kwargs.pop("perm_strat"))
        replicas = kwargs.pop("replicas", None)

        # If param dictionaries are empty, empty lists will be returned
        param_names, params, arg_names, arg_params = self._read_model_parameters()

        # Pre-compute parameterized command line arguments and model
        # parameters, as we need them in different branches, if they are given
        if self.params or self.arg_params:
            all_params = strategy(
                param_names + arg_names, params + arg_params, **kwargs
            )
            if not isinstance(all_params, list):
                raise UserStrategyError(strategy)

            all_model_params = []
            all_arg_params = []

            for all_param_list in all_params:
                if not isinstance(all_param_list, dict):
                    raise UserStrategyError(strategy)
                model_param_dict = {}
                arg_param_dict = {}
                for k, v in all_param_list.items():
                    if k in param_names:
                        model_param_dict[k] = v
                    elif k in arg_names:
                        arg_param_dict[k] = v
                    else:
                        raise UserStrategyError(strategy)
                if model_param_dict:
                    all_model_params.append(model_param_dict)
                if arg_param_dict:
                    all_arg_params.append(arg_param_dict)

            if self.arg_params:
                # run settings with exe_args modified according to params
                model_run_settings = []
                for arg_param_set in all_arg_params:
                    if not isinstance(arg_param_set, dict):
                        raise UserStrategyError(strategy)
                    run_settings = deepcopy(self.run_settings)
                    for arg_param_name, arg_param_value in arg_param_set.items():
                        run_settings.add_exe_args(
                            cat_arg_and_value(arg_param_name, arg_param_value)
                        )
                    model_run_settings.append(run_settings)

        # if a ensemble has parameters and run settings, create
        # the ensemble and assign run_settings to each member
        if self.params:
            if self.run_settings:
                for i, param_set in enumerate(all_model_params):

                    if self.arg_params:
                        run_settings = model_run_settings[i]
                    else:
                        run_settings = deepcopy(self.run_settings)

                    model_name = "_".join((self.name, str(i)))
                    model = Model(
                        model_name,
                        param_set,
                        self.path,
                        run_settings=run_settings,
                    )
                    model.enable_key_prefixing()
                    logger.debug(
                        f"Created ensemble member: {model_name} in {self.name}"
                    )
                    self.add_model(model)
            # cannot generate models without run settings
            else:
                raise SmartSimError(
                    "Ensembles without 'params', 'arg_params', or 'replicas' argument to expand into members cannot be given run settings"
                )
        elif self.arg_params:
            if self.run_settings:
                for i, run_settings in enumerate(model_run_settings):
                    model_name = "_".join((self.name, str(i)))
                    model = Model(
                        model_name,
                        {},
                        self.path,
                        run_settings=run_settings,
                    )
                    model.enable_key_prefixing()
                    logger.debug(
                        f"Created ensemble member: {model_name} in {self.name}"
                    )
                    self.add_model(model)
            else:
                raise SmartSimError(
                    "Ensembles without 'params', 'arg_params', or 'replicas' argument to expand into members cannot be given run settings"
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
                        "Ensembles without 'params', 'arg_params', or 'replicas' argument to expand into members cannot be given run settings"
                    )
            # if no params, no run settings and no batch settings, error because we
            # don't know how to run the ensemble
            elif not self.batch_settings:
                raise SmartSimError(
                    "Ensemble must be provided batch settings or run settings"
                )
            else:
                logger.info("Empty ensemble created for batch launch")

    def add_model(self, model):
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
        self.entities.append(model)

    def register_incoming_entity(self, incoming_entity):
        """Register future communication between entities.

        Registers the named data sources that this entity
        has access to by storing the key_prefix associated
        with that entity

        Only python clients can have multiple incoming connections

        :param incoming_entity: The entity that data will be received from
        :type incoming_entity: SmartSimEntity
        """
        for model in self.entities:
            model.register_incoming_entity(incoming_entity)

    def enable_key_prefixing(self):
        """If called, all models within this ensemble will prefix their keys with its
        own model name.
        """
        for model in self.entities:
            model.enable_key_prefixing()

    def query_key_prefixing(self):
        """Inquire as to whether each model within the ensemble will prefix its keys

        :returns: True if all models have key prefixing enabled, False otherwise
        :rtype: bool
        """
        return all([model.query_key_prefixing() for model in self.entities])

    def attach_generator_files(self, to_copy=None, to_symlink=None, to_configure=None):
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
        for model in self.entities:
            model.attach_generator_files(
                to_copy=to_copy, to_symlink=to_symlink, to_configure=to_configure
            )

    def _set_strategy(self, strategy):
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

    def _read_model_parameters(self):
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

        if not isinstance(self.arg_params, dict):
            raise TypeError(
                "Ensemble initialization argument 'params' must be of type dict"
            )

        def list_params(params):
            param_names = []
            parameters = []
            for name, val in params.items():
                param_names.append(name)

                if isinstance(val, list):
                    parameters.append(val)
                elif isinstance(val, (int, str)):
                    parameters.append([val])
                else:
                    raise TypeError(
                        "Incorrect type for ensemble parameters\n"
                        + "Must be list, int, or string."
                    )
            return param_names, parameters

        packed_params = list_params(self.params)
        packed_arg_params = list_params(self.arg_params)
        return (
            packed_params[0],
            packed_params[1],
            packed_arg_params[0],
            packed_arg_params[1],
        )
