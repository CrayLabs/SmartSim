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
import os
import os.path
import typing as t
import random

from smartsim.builders.utils import strategies
from smartsim.builders.utils.strategies import ParamSet
from smartsim.entity import entity
from smartsim.entity.application import Application
from smartsim.launchable.job import Job
from smartsim._core.generation.builder_operations import EnsembleFileSysOperationSet, EnsembleConfigureOperation
from dataclasses import dataclass, field


if t.TYPE_CHECKING:
    from smartsim.settings.launch_settings import LaunchSettings
@dataclass(frozen=True)
class FileSet:
    """
    Represents a set of file parameters and execution arguments as parameters.
    """

    file: EnsembleConfigureOperation
    combinations: ParamSet

@dataclass(frozen=True)
class Combo:
    """
    Represents a set of file parameters and execution arguments as parameters.
    """

    file: EnsembleConfigureOperation
    combination: ParamSet

@dataclass(frozen=True)
class ComboSet:
    """
    Represents a set of file parameters and execution arguments as parameters.
    """

    combos: t.List[Combo]

class Ensemble(entity.CompoundEntity):
    """An Ensemble is a builder class that parameterizes the creation of multiple
    Applications.
    """

    def __init__(
        self,
        name: str,
        exe: str | os.PathLike[str],
        exe_args: t.Sequence[str] | None = None,
        exe_arg_parameters: t.Mapping[str, t.Sequence[t.Sequence[str]]] | None = None,
        permutation_strategy: str | strategies.PermutationStrategyType = "all_perm",
        max_permutations: int = -1,
        replicas: int = 1,
    ) -> None:
        """Initialize an ``Ensemble`` of Application instances

        An Ensemble can be tailored to align with one of the following
        creation strategies: parameter expansion or replicas.

        **Parameter Expansion**

        Parameter expansion allows users to assign different parameter values to
        multiple Applications. This is done by specifying input to `Ensemble.file_parameters`,
        `Ensemble.exe_arg_parameters` and `Ensemble.permutation_strategy`. The `permutation_strategy`
        argument accepts three options:

        1. "all_perm": Generates all possible parameter permutations for exhaustive exploration.
        2. "step": Collects identically indexed values across parameter lists to create parameter sets.
        3. "random": Enables random selection from predefined parameter spaces.

        The example below demonstrates creating an Ensemble via parameter expansion, resulting in
        the creation of two Applications:

        .. highlight:: python
        .. code-block:: python

            file_params={"SPAM": ["a", "b"], "EGGS": ["c", "d"]}
            exe_arg_parameters = {"EXE": [["a"], ["b", "c"]], "ARGS": [["d"], ["e", "f"]]}
            ensemble = Ensemble(name="name",exe="python",exe_arg_parameters=exe_arg_parameters,
                        file_parameters=file_params,permutation_strategy="step")

        This configuration will yield the following permutations:

        .. highlight:: python
        .. code-block:: python
            [ParamSet(params={'SPAM': 'a', 'EGGS': 'c'}, exe_args={'EXE': ['a'], 'ARGS': ['d']}),
             ParamSet(params={'SPAM': 'b', 'EGGS': 'd'}, exe_args={'EXE': ['b', 'c'], 'ARGS': ['e', 'f']})]

        Each ParamSet contains the parameters assigned from file_params and the corresponding executable
        arguments from exe_arg_parameters.

        **Replication**
        The replication strategy involves creating identical Applications within an Ensemble.
        This is achieved by specifying the `replicas` argument in the Ensemble.

        For example, by applying the `replicas` argument to the previous parameter expansion
        example, we can double our Application output:

        .. highlight:: python
        .. code-block:: python

            file_params={"SPAM": ["a", "b"], "EGGS": ["c", "d"]}
            exe_arg_parameters = {"EXE": [["a"], ["b", "c"]], "ARGS": [["d"], ["e", "f"]]}
            ensemble = Ensemble(name="name",exe="python",exe_arg_parameters=exe_arg_parameters,
                        file_parameters=file_params,permutation_strategy="step", replicas=2)

        This configuration will result in each ParamSet being replicated, effectively doubling
        the number of Applications created.

        :param name: name of the ensemble
        :param exe: executable to run
        :param exe_args: executable arguments
        :param exe_arg_parameters: parameters and values to be used when configuring entities
        :param files: files to be copied, symlinked, and/or configured prior to
                      execution
        :param file_parameters: parameters and values to be used when configuring
                                files
        :param permutation_strategy: strategy to control how the param values are applied to the Ensemble
        :param max_permutations: max parameter permutations to set for the ensemble
        :param replicas: number of identical entities to create within an Ensemble
        """
        self.name = name
        """The name of the ensemble"""
        self._exe = os.fspath(exe)
        """The executable to run"""
        self.exe_args = list(exe_args) if exe_args else []
        """The executable arguments"""
        self._exe_arg_parameters = (
            copy.deepcopy(exe_arg_parameters) if exe_arg_parameters else {}
        )
        """The parameters and values to be used when configuring entities"""
        self.files = EnsembleFileSysOperationSet([])
        """The files to be copied, symlinked, and/or configured prior to execution"""
        self._permutation_strategy = permutation_strategy
        """The strategy to control how the param values are applied to the Ensemble"""
        self._max_permutations = max_permutations
        """The maximum number of entities to come out of the permutation strategy"""
        self._replicas = replicas
        """How many identical entities to create within an Ensemble"""

    @property
    def exe(self) -> str:
        """Return the attached executable.

        :return: the executable
        """
        return self._exe

    @exe.setter
    def exe(self, value: str | os.PathLike[str]) -> None:
        """Set the executable.

        :param value: the executable
        """
        self._exe = os.fspath(value)

    @property
    def exe_args(self) -> t.List[str]:
        """Return attached list of executable arguments.

        :return: the executable arguments
        """
        return self._exe_args

    @exe_args.setter
    def exe_args(self, value: t.Sequence[str]) -> None:
        """Set the executable arguments.

        :param value: the executable arguments
        """
        self._exe_args = list(value)

    @property
    def exe_arg_parameters(self) -> t.Mapping[str, t.Sequence[t.Sequence[str]]]:
        """Return attached executable argument parameters.

        :return: the executable argument parameters
        """
        return self._exe_arg_parameters

    @exe_arg_parameters.setter
    def exe_arg_parameters(
        self, value: t.Mapping[str, t.Sequence[t.Sequence[str]]]
    ) -> None:
        """Set the executable argument parameters.

        :param value: the executable argument parameters
        """
        self._exe_arg_parameters = copy.deepcopy(value)

    @property
    def permutation_strategy(self) -> str | strategies.PermutationStrategyType:
        """Return the permutation strategy

        :return: the permutation strategy
        """
        return self._permutation_strategy

    @permutation_strategy.setter
    def permutation_strategy(
        self, value: str | strategies.PermutationStrategyType
    ) -> None:
        """Set the permutation strategy

        :param value: the permutation strategy
        """
        self._permutation_strategy = value

    @property
    def max_permutations(self) -> int:
        """Return the maximum permutations

        :return: the max permutations
        """
        return self._max_permutations

    @max_permutations.setter
    def max_permutations(self, value: int) -> None:
        """Set the maximum permutations

        :param value: the max permutations
        """
        self._max_permutations = value

    @property
    def replicas(self) -> int:
        """Return the number of replicas.

        :return: the number of replicas
        """
        return self._replicas

    @replicas.setter
    def replicas(self, value: int) -> None:
        """Set the number of replicas.

        :return: the number of replicas
        """
        self._replicas = value

    def _create_applications(self) -> tuple[Application, ...]:
        """Generate a collection of Application instances based on the Ensembles attributes.

        This method uses a permutation strategy to create various combinations of file
        parameters and executable arguments. Each combination is then replicated according
        to the specified number of replicas, resulting in a set of Application instances.

        :return: A tuple of Application instances
        """
        # resolve the permutation strategy
        permutation_strategy = strategies.resolve(self.permutation_strategy)
        # apply the permutation strategy to each attached config file
        perm_list: t.List[t.List[FileSet]] = [self.perm_config_file(config_file, permutation_strategy) for config_file in self.files.configure_operations]
        # group the files together
        val: t.List[tuple[FileSet]] = self._cartesian_values(perm_list)
        # duplicate if replicas
        permutations_ = itertools.chain.from_iterable(
            itertools.repeat(permutation, self.replicas) for permutation in val
        )
        all_apps = []
        i = 0
        for item in permutations_:
            i+=1
            app = Application(
                name=f"{self.name}-{i}",
                exe=self.exe,
                exe_args=self.exe_args,
            )
            # apply the config files in the tuple
            for file_set in item:
                app.files.add_configuration(src=file_set.file.src, file_parameters=file_set.combinations.params)
            all_apps.append(app)
        return tuple(all_apps)


    def perm_config_file(self, file, permutation_strategy):
        combinations = permutation_strategy(
            file.file_parameters, self.exe_arg_parameters, self.max_permutations
        ) or [ParamSet({}, {})]
        return [FileSet(file, combo) for combo in combinations]


    def _cartesian_values(self, ls):
        return list(itertools.product(*ls))


    def build_jobs(self, settings: LaunchSettings) -> tuple[Job, ...]:
        """Expand an Ensemble into a list of deployable Jobs and apply
        identical LaunchSettings to each Job.

        The number of Jobs returned is controlled by the Ensemble attributes:
            - Ensemble.exe_arg_parameters
            - Ensemble.file_parameters
            - Ensemble.permutation_strategy
            - Ensemble.max_permutations
            - Ensemble.replicas

        Consider the example below:

        .. highlight:: python
        .. code-block:: python

            # Create LaunchSettings
            my_launch_settings = LaunchSettings(...)

            # Initialize the Ensemble
            ensemble = Ensemble("my_name", "echo", "hello world", replicas=3)
            # Expand Ensemble into Jobs
            ensemble_as_jobs = ensemble.build_jobs(my_launch_settings)

        By calling `build_jobs` on `ensemble`, three Jobs are returned because
        three replicas were specified. Each Job will have the provided LaunchSettings.

        :param settings: LaunchSettings to apply to each Job
        :return: Sequence of Jobs with the provided LaunchSettings
        """
        apps = self._create_applications()
        if not apps:
            raise ValueError("There are no members as part of this ensemble")
        return tuple(Job(app, settings, app.name) for app in apps)
