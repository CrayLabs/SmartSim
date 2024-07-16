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

import subprocess as sp
import typing as t
import uuid

from smartsim._core.utils import helpers
from smartsim.types import LaunchedJobID

if t.TYPE_CHECKING:
    from typing_extensions import Self

    from smartsim.experiment import Experiment
    from smartsim.settings.builders import LaunchArgBuilder

_T = t.TypeVar("_T")
_T_contra = t.TypeVar("_T_contra", contravariant=True)


@t.final
class Dispatcher:
    """A class capable of deciding which launcher type should be used to launch
    a given settings builder type.
    """

    def __init__(
        self,
        *,
        dispatch_registry: (
            t.Mapping[type[LaunchArgBuilder[t.Any]], type[LauncherLike[t.Any]]] | None
        ) = None,
    ) -> None:
        self._dispatch_registry = (
            dict(dispatch_registry) if dispatch_registry is not None else {}
        )

    def copy(self) -> Self:
        """Create a shallow copy of the Dispatcher"""
        return type(self)(dispatch_registry=self._dispatch_registry)

    @t.overload
    def dispatch(
        self,
        args: None = ...,
        *,
        to_launcher: type[LauncherLike[_T]],
        allow_overwrite: bool = ...,
    ) -> t.Callable[[type[LaunchArgBuilder[_T]]], type[LaunchArgBuilder[_T]]]: ...
    @t.overload
    def dispatch(
        self,
        args: type[LaunchArgBuilder[_T]],
        *,
        to_launcher: type[LauncherLike[_T]],
        allow_overwrite: bool = ...,
    ) -> None: ...
    def dispatch(
        self,
        args: type[LaunchArgBuilder[_T]] | None = None,
        *,
        to_launcher: type[LauncherLike[_T]],
        allow_overwrite: bool = False,
    ) -> t.Callable[[type[LaunchArgBuilder[_T]]], type[LaunchArgBuilder[_T]]] | None:
        """A type safe way to add a mapping of settings builder to launcher to
        handle the settings at launch time.
        """

        def register(
            args_: type[LaunchArgBuilder[_T]], /
        ) -> type[LaunchArgBuilder[_T]]:
            if args_ in self._dispatch_registry and not allow_overwrite:
                launcher_type = self._dispatch_registry[args_]
                raise TypeError(
                    f"{args_.__name__} has already been registered to be "
                    f"launched with {launcher_type}"
                )
            self._dispatch_registry[args_] = to_launcher
            return args_

        if args is not None:
            register(args)
            return None
        return register

    def get_launcher_for(
        self, args: LaunchArgBuilder[_T] | type[LaunchArgBuilder[_T]], /
    ) -> type[LauncherLike[_T]]:
        """Find a type of launcher that is registered as being able to launch
        the output of the provided builder
        """
        if not isinstance(args, type):
            args = type(args)
        launcher_type = self._dispatch_registry.get(args, None)
        if launcher_type is None:
            raise TypeError(
                f"{type(self).__name__} {self} has no launcher type to "
                f"dispatch to for argument builder of type {args}"
            )
        # Note the sleight-of-hand here: we are secretly casting a type of
        # `LauncherLike[Any]` to `LauncherLike[_T]`. This is safe to do if all
        # entries in the mapping were added using a type safe method (e.g.
        # `Dispatcher.dispatch`), but if a user were to supply a custom
        # dispatch registry or otherwise modify the registry THIS IS NOT
        # NECESSARILY 100% TYPE SAFE!!
        return launcher_type


default_dispatcher: t.Final = Dispatcher()
dispatch: t.Final = default_dispatcher.dispatch


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TODO: move these to a common module under `smartsim._core.launcher`
# -----------------------------------------------------------------------------


def create_job_id() -> LaunchedJobID:
    return LaunchedJobID(str(uuid.uuid4()))


class LauncherLike(t.Protocol[_T_contra]):
    def start(self, launchable: _T_contra) -> LaunchedJobID: ...
    @classmethod
    def create(cls, exp: Experiment) -> Self: ...


class ShellLauncher:
    """Mock launcher for launching/tracking simple shell commands"""

    def __init__(self) -> None:
        self._launched: dict[LaunchedJobID, sp.Popen[bytes]] = {}

    def start(self, launchable: t.Sequence[str]) -> LaunchedJobID:
        id_ = create_job_id()
        exe, *rest = launchable
        self._launched[id_] = sp.Popen((helpers.expand_exe_path(exe), *rest)) # can specify a different dir for Popen
        return id_

    @classmethod
    def create(cls, exp: Experiment) -> Self:
        return cls()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
