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
import typing as t

# TODO: We are using the mocks defined `mock` module. We should use the correct
#       module when these types are actually defined when PRs #587 and #603 are
#       merged.
# https://github.com/CrayLabs/SmartSim/pull/587
# https://github.com/CrayLabs/SmartSim/pull/603
from smartsim.entity import _mock, entity, strategies
from smartsim.entity.files import EntityFiles
from smartsim.entity.model import Application
from smartsim.entity.strategies import ParamSet


# TODO: If we like this design, we need to:
#         1) Move this to the `smartsim._core.entity.ensemble` module
#         2) Decide what to do witht the original `Ensemble` impl
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
        self.name = name
        self.exe = os.fspath(exe)
        self.exe_args = list(exe_args) if exe_args else []
        self.exe_arg_parameters = (
            copy.deepcopy(exe_arg_parameters) if exe_arg_parameters else {}
        )
        self.files = copy.deepcopy(files) if files else EntityFiles()
        self.file_parameters = dict(file_parameters) if file_parameters else {}
        self.permutation_strategy = permutation_strategy
        self.max_permutations = max_permutations
        self.replicas = replicas

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
                run_settings=_mock.Mock(),  # type: ignore[arg-type]
                # ^^^^^^^^^^^^^^^^^^^
                # FIXME: remove this constructor arg! It should not exist!!
                exe_args=self.exe_args,
                files=self.files,
                params=permutation.params,
                params_as_args=permutation.exe_args,
            )
            for i, permutation in enumerate(permutations_)
        )

    def as_jobs(self, settings: _mock.LaunchSettings) -> tuple[_mock.Job, ...]:
        apps = self._create_applications()
        if not apps:
            raise ValueError("There are no members as part of this ensemble")
        return tuple(_mock.Job(app, settings) for app in apps)
