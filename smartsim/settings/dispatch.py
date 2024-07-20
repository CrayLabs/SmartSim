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

import dataclasses
import subprocess as sp
import typing as t
import uuid

from typing_extensions import Self, TypeVarTuple, Unpack

from smartsim._core.utils import helpers
from smartsim.error import errors
from smartsim.types import LaunchedJobID

if t.TYPE_CHECKING:
    from smartsim.experiment import Experiment
    from smartsim.settings.builders import LaunchArgBuilder


_T = t.TypeVar("_T")
_Ts = TypeVarTuple("_Ts")
_T_contra = t.TypeVar("_T_contra", contravariant=True)
_TDispatchable = t.TypeVar("_TDispatchable", bound="LaunchArgBuilder")
_EnvironMappingType: t.TypeAlias = t.Mapping[str, "str | None"]
_FormatterType: t.TypeAlias = t.Callable[
    [_TDispatchable, "ExecutableLike", _EnvironMappingType], _T
]
_LaunchConfigType: t.TypeAlias = "_LauncherAdapter[ExecutableLike, _EnvironMappingType]"
_UnkownType: t.TypeAlias = t.NoReturn


@t.final
class Dispatcher:
    """A class capable of deciding which launcher type should be used to launch
    a given settings builder type.
    """

    def __init__(
        self,
        *,
        dispatch_registry: (
            t.Mapping[type[LaunchArgBuilder], _DispatchRegistration[t.Any, t.Any]]
            | None
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
        with_format: _FormatterType[_TDispatchable, _T],
        to_launcher: type[LauncherLike[_T]],
        allow_overwrite: bool = ...,
    ) -> t.Callable[[type[_TDispatchable]], type[_TDispatchable]]: ...
    @t.overload
    def dispatch(
        self,
        args: type[_TDispatchable],
        *,
        with_format: _FormatterType[_TDispatchable, _T],
        to_launcher: type[LauncherLike[_T]],
        allow_overwrite: bool = ...,
    ) -> None: ...
    def dispatch(
        self,
        args: type[_TDispatchable] | None = None,
        *,
        with_format: _FormatterType[_TDispatchable, _T],
        to_launcher: type[LauncherLike[_T]],
        allow_overwrite: bool = False,
    ) -> t.Callable[[type[_TDispatchable]], type[_TDispatchable]] | None:
        """A type safe way to add a mapping of settings builder to launcher to
        handle the settings at launch time.
        """
        err_msg: str | None = None
        if getattr(to_launcher, "_is_protocol", False):
            err_msg = f"Cannot dispatch to protocol class `{to_launcher.__name__}`"
        elif getattr(to_launcher, "__abstractmethods__", frozenset()):
            err_msg = f"Cannot dispatch to abstract class `{to_launcher.__name__}`"
        if err_msg is not None:
            raise TypeError(err_msg)

        def register(args_: type[_TDispatchable], /) -> type[_TDispatchable]:
            if args_ in self._dispatch_registry and not allow_overwrite:
                launcher_type = self._dispatch_registry[args_].launcher_type
                raise TypeError(
                    f"{args_.__name__} has already been registered to be "
                    f"launched with {launcher_type}"
                )
            self._dispatch_registry[args_] = _DispatchRegistration(
                with_format, to_launcher
            )
            return args_

        if args is not None:
            register(args)
            return None
        return register

    def get_dispatch(
        self, args: _TDispatchable | type[_TDispatchable]
    ) -> _DispatchRegistration[_TDispatchable, _UnkownType]:
        """Find a type of launcher that is registered as being able to launch
        the output of the provided builder
        """
        if not isinstance(args, type):
            args = type(args)
        dispatch = self._dispatch_registry.get(args, None)
        if dispatch is None:
            raise TypeError(
                f"No dispatch for `{type(args).__name__}` has been registered "
                f"has been registered with {type(self).__name__} `{self}`"
            )
        # Note the sleight-of-hand here: we are secretly casting a type of
        # `_DispatchRegistration[Any, Any]` ->
        #     `_DispatchRegistration[_TDispatchable, _T]`.
        #  where `_T` is unbound!
        #
        # This is safe to do if all entries in the mapping were added using a
        # type safe method (e.g.  `Dispatcher.dispatch`), but if a user were to
        # supply a custom dispatch registry or otherwise modify the registry
        # this is not necessarily 100% type safe!!
        return dispatch


@t.final
@dataclasses.dataclass(frozen=True)
class _DispatchRegistration(t.Generic[_TDispatchable, _T]):
    formatter: _FormatterType[_TDispatchable, _T]
    launcher_type: type[LauncherLike[_T]]

    def _is_compatible_launcher(self, launcher: LauncherLike[t.Any]) -> bool:
        return type(launcher) is self.launcher_type

    def create_new_launcher_configuration(
        self, for_experiment: Experiment, with_settings: _TDispatchable
    ) -> _LaunchConfigType:
        launcher = self.launcher_type.create(for_experiment)
        return self.create_adapter_from_launcher(launcher, with_settings)

    def create_adapter_from_launcher(
        self, launcher: LauncherLike[_T], settings: _TDispatchable
    ) -> _LaunchConfigType:
        if not self._is_compatible_launcher(launcher):
            raise TypeError(
                f"Cannot create launcher adapter from launcher `{launcher}` "
                f"of type `{type(launcher)}`; expected launcher of type "
                f"exactly `{self.launcher_type}`"
            )

        def format_(exe: ExecutableLike, env: _EnvironMappingType) -> _T:
            return self.formatter(settings, exe, env)

        return _LauncherAdapter(launcher, format_)

    def configure_first_compatible_launcher(
        self,
        with_settings: _TDispatchable,
        from_available_launchers: t.Iterable[LauncherLike[t.Any]],
    ) -> _LaunchConfigType:
        launcher = helpers.first(self._is_compatible_launcher, from_available_launchers)
        if launcher is None:
            raise errors.LauncherNotFoundError(
                f"No launcher of exactly type `{self.launcher_type.__name__}` "
                "could be found from provided launchers"
            )
        return self.create_adapter_from_launcher(launcher, with_settings)


@t.final
class _LauncherAdapter(t.Generic[Unpack[_Ts]]):
    def __init__(
        self, launcher: LauncherLike[_T], map_: t.Callable[[Unpack[_Ts]], _T]
    ) -> None:
        # NOTE: We need to cast off the `_T` -> `Any` in the `__init__`
        #       signature to hide the transform from users of this class. If
        #       possible, try not to expose outside of protected methods!
        self._adapt: t.Callable[[Unpack[_Ts]], t.Any] = map_
        self._adapted_launcher: LauncherLike[t.Any] = launcher

    def start(self, *args: Unpack[_Ts]) -> LaunchedJobID:
        payload = self._adapt(*args)
        return self._adapted_launcher.start(payload)


default_dispatcher: t.Final = Dispatcher()
dispatch: t.Final = default_dispatcher.dispatch


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TODO: move these to a common module under `smartsim._core.launcher`
# -----------------------------------------------------------------------------


def create_job_id() -> LaunchedJobID:
    return LaunchedJobID(str(uuid.uuid4()))


class ExecutableLike(t.Protocol):
    def as_program_arguments(self) -> t.Sequence[str]: ...


class LauncherLike(t.Protocol[_T_contra]):
    def start(self, launchable: _T_contra) -> LaunchedJobID: ...
    @classmethod
    def create(cls, exp: Experiment) -> Self: ...


# TODO: This is just a nice helper function that I am using for the time being
#       to wire everything up! In reality it might be a bit too confusing and
#       meta-program-y for production code. Check with the core team to see
#       what they think!!
def shell_format(
    run_command: str | None,
) -> _FormatterType[LaunchArgBuilder, t.Sequence[str]]:
    def impl(
        args: LaunchArgBuilder, exe: ExecutableLike, env: _EnvironMappingType
    ) -> t.Sequence[str]:
        return (
            (
                run_command,
                *(args.format_launch_args() or ()),
                "--",
                *exe.as_program_arguments(),
            )
            if run_command is not None
            else exe.as_program_arguments()
        )

    return impl


class ShellLauncher:
    """Mock launcher for launching/tracking simple shell commands"""

    def __init__(self) -> None:
        self._launched: dict[LaunchedJobID, sp.Popen[bytes]] = {}

    def start(self, launchable: t.Sequence[str]) -> LaunchedJobID:
        id_ = create_job_id()
        exe, *rest = launchable
        self._launched[id_] = sp.Popen((helpers.expand_exe_path(exe), *rest))
        return id_

    @classmethod
    def create(cls, exp: Experiment) -> Self:
        return cls()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
