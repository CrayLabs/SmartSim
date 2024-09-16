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

from smartsim.entity import entity, strategies
from smartsim.entity.application import Application
from smartsim.entity.files import EntityFiles
from smartsim.entity.strategies import ParamSet
from smartsim.launchable.job import Job

if t.TYPE_CHECKING:
    from smartsim.settings.launchSettings import LaunchSettings


class Ensemble(entity.CompoundEntity):
    """Entity to help parameterize the creation multiple application
    instances.
    """

    def __init__(
        self,
        name: str,
        exe: str | os.PathLike[str],
        exe_args: t.Sequence[str] | None = None,
        exe_arg_parameters: t.Mapping[str, t.Sequence[t.Sequence[str]]] | None = None,
        files: EntityFiles | None = None,
        file_parameters: t.Mapping[str, t.Sequence[str]] | None = None,
        permutation_strategy: str | strategies.PermutationStrategyType = "all_perm",
        max_permutations: int = -1,
        replicas: int = 1,
    ) -> None:
        """Initialize an ``Ensemble`` of application instances

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
        self._files = copy.deepcopy(files) if files else EntityFiles()
        """The files to be copied, symlinked, and/or configured prior to execution"""
        self._file_parameters = (
            copy.deepcopy(file_parameters) if file_parameters else {}
        )
        """The parameters and values to be used when configuring files"""
        self._permutation_strategy = permutation_strategy
        """The strategy to control how the param values are applied to the Ensemble"""
        self._max_permutations = max_permutations
        """The maximum number of entities to come out of the permutation strategy"""
        self._replicas = replicas
        """How many identical entities to create within an Ensemble"""

    @property
    def exe(self) -> str:
        """Return executable to run.

        :returns: application executable to run
        """
        return self._exe

    @exe.setter
    def exe(self, value: str | os.PathLike[str]) -> None:
        """Set executable to run.

        :param value: executable to run
        """
        self._exe = os.fspath(value)

    @property
    def exe_args(self) -> t.List[str]:
        """Return a list of attached executable arguments.

        :returns: application executable arguments
        """
        return self._exe_args

    @exe_args.setter
    def exe_args(self, value: t.Sequence[str]) -> None:
        """Set the executable arguments.

        :param value: executable arguments
        """
        self._exe_args = list(value)

    @property
    def exe_arg_parameters(self) -> t.Mapping[str, t.Sequence[t.Sequence[str]]]:
        """Return the executable argument parameters

        :returns: executable arguments parameters
        """
        return self._exe_arg_parameters

    @exe_arg_parameters.setter
    def exe_arg_parameters(
        self, value: t.Mapping[str, t.Sequence[t.Sequence[str]]]
    ) -> None:
        """Set the executable arguments.

        :param value: executable arguments
        """
        self._exe_arg_parameters = copy.deepcopy(value)

    @property
    def files(self) -> EntityFiles:
        """Return files to be copied, symlinked, and/or configured prior to
        execution.

        :returns: files
        """
        return self._files

    @files.setter
    def files(self, value: EntityFiles) -> None:
        """Set files to be copied, symlinked, and/or configured prior to
        execution.

        :param value: files
        """
        self._files = copy.deepcopy(value)

    @property
    def file_parameters(self) -> t.Mapping[str, t.Sequence[str]]:
        """Return file parameters.

        :returns: application file parameters
        """
        return self._file_parameters

    @file_parameters.setter
    def file_parameters(self, value: t.Mapping[str, t.Sequence[str]]) -> None:
        """Set the file parameters.

        :param value: file parameters
        """
        self._file_parameters = dict(value)

    @property
    def permutation_strategy(self) -> str | strategies.PermutationStrategyType:
        """Return the permutation strategy

        :return: permutation strategy
        """
        return self._permutation_strategy

    @permutation_strategy.setter
    def permutation_strategy(
        self, value: str | strategies.PermutationStrategyType
    ) -> None:
        """Set the permutation strategy

        :param value: permutation strategy
        """
        self._permutation_strategy = value

    @property
    def max_permutations(self) -> int:
        """Return the maximum permutations

        :return: max permutations
        """
        return self._max_permutations

    @max_permutations.setter
    def max_permutations(self, value: int) -> None:
        """Set the maximum permutations

        :param value: the maxpermutations
        """
        self._max_permutations = value

    @property
    def replicas(self) -> int:
        """Return the number of replicas

        :return: number of replicas
        """
        return self._replicas

    @replicas.setter
    def replicas(self, value: int) -> None:
        """Set the number of replicas

        :return: the number of replicas
        """
        self._replicas = value

    def _create_applications(self) -> tuple[Application, ...]:
        """Concretize the ensemble attributes into a collection of
        application instances.
        """
        permutation_strategy = strategies.resolve(self.permutation_strategy)

        combinations = permutation_strategy(
            self.file_parameters, self.exe_arg_parameters, self.max_permutations
        )
        combinations = combinations if combinations else [ParamSet({}, {})]
        permutations_ = itertools.chain.from_iterable(
            itertools.repeat(permutation, self.replicas) for permutation in combinations
        )
        return tuple(
            Application(
                name=f"{self.name}-{i}",
                exe=self.exe,
                exe_args=self.exe_args,
                files=self.files,
                file_parameters=permutation.params,
            )
            for i, permutation in enumerate(permutations_)
        )

    def as_jobs(self, settings: LaunchSettings) -> tuple[Job, ...]:
        if not settings is None:
            apps = self._create_applications()
            if not apps:
                raise ValueError("There are no members as part of this ensemble")
            return tuple(Job(app, settings, app.name) for app in apps)
        else:
            raise ValueError("The Launch Settings provided are empty")
