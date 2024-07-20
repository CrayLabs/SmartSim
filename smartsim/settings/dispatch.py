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

from typing_extensions import Self, TypeAlias, TypeVarTuple, Unpack

from smartsim._core.utils import helpers
from smartsim.error import errors
from smartsim.types import LaunchedJobID

if t.TYPE_CHECKING:
    from smartsim.experiment import Experiment
    from smartsim.settings.builders import LaunchArgBuilder

_Ts = TypeVarTuple("_Ts")
_T_contra = t.TypeVar("_T_contra", contravariant=True)
_DispatchableT = t.TypeVar("_DispatchableT", bound="LaunchArgBuilder")
_LaunchableT = t.TypeVar("_LaunchableT")

_EnvironMappingType: TypeAlias = t.Mapping[str, "str | None"]
_FormatterType: TypeAlias = t.Callable[
    [_DispatchableT, "ExecutableLike", _EnvironMappingType], _LaunchableT
]
_LaunchConfigType: TypeAlias = "_LauncherAdapter[ExecutableLike, _EnvironMappingType]"
_UnkownType: TypeAlias = t.NoReturn


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
        with_format: _FormatterType[_DispatchableT, _LaunchableT],
        to_launcher: type[LauncherLike[_LaunchableT]],
        allow_overwrite: bool = ...,
    ) -> t.Callable[[type[_DispatchableT]], type[_DispatchableT]]: ...
    @t.overload
    def dispatch(
        self,
        args: type[_DispatchableT],
        *,
        with_format: _FormatterType[_DispatchableT, _LaunchableT],
        to_launcher: type[LauncherLike[_LaunchableT]],
        allow_overwrite: bool = ...,
    ) -> None: ...
    def dispatch(
        self,
        args: type[_DispatchableT] | None = None,
        *,
        with_format: _FormatterType[_DispatchableT, _LaunchableT],
        to_launcher: type[LauncherLike[_LaunchableT]],
        allow_overwrite: bool = False,
    ) -> t.Callable[[type[_DispatchableT]], type[_DispatchableT]] | None:
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

        def register(args_: type[_DispatchableT], /) -> type[_DispatchableT]:
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
        self, args: _DispatchableT | type[_DispatchableT]
    ) -> _DispatchRegistration[_DispatchableT, _UnkownType]:
        """Find a type of launcher that is registered as being able to launch
        the output of the provided builder
        """
        if not isinstance(args, type):
            args = type(args)
        dispatch_ = self._dispatch_registry.get(args, None)
        if dispatch_ is None:
            raise TypeError(
                f"No dispatch for `{type(args).__name__}` has been registered "
                f"has been registered with {type(self).__name__} `{self}`"
            )
        # Note the sleight-of-hand here: we are secretly casting a type of
        # `_DispatchRegistration[Any, Any]` ->
        #     `_DispatchRegistration[_DispatchableT, _LaunchableT]`.
        #  where `_LaunchableT` is unbound!
        #
        # This is safe to do if all entries in the mapping were added using a
        # type safe method (e.g.  `Dispatcher.dispatch`), but if a user were to
        # supply a custom dispatch registry or otherwise modify the registry
        # this is not necessarily 100% type safe!!
        return dispatch_


@t.final
@dataclasses.dataclass(frozen=True)
class _DispatchRegistration(t.Generic[_DispatchableT, _LaunchableT]):
    formatter: _FormatterType[_DispatchableT, _LaunchableT]
    launcher_type: type[LauncherLike[_LaunchableT]]

    def _is_compatible_launcher(self, launcher: LauncherLike[t.Any]) -> bool:
        # Disabling because we want to match the the type of the dispatch
        # *exactly* as specified by the user
        # pylint: disable-next=unidiomatic-typecheck
        return type(launcher) is self.launcher_type

    def create_new_launcher_configuration(
        self, for_experiment: Experiment, with_settings: _DispatchableT
    ) -> _LaunchConfigType:
        launcher = self.launcher_type.create(for_experiment)
        return self.create_adapter_from_launcher(launcher, with_settings)

    def create_adapter_from_launcher(
        self, launcher: LauncherLike[_LaunchableT], settings: _DispatchableT
    ) -> _LaunchConfigType:
        if not self._is_compatible_launcher(launcher):
            raise TypeError(
                f"Cannot create launcher adapter from launcher `{launcher}` "
                f"of type `{type(launcher)}`; expected launcher of type "
                f"exactly `{self.launcher_type}`"
            )

        def format_(exe: ExecutableLike, env: _EnvironMappingType) -> _LaunchableT:
            return self.formatter(settings, exe, env)

        return _LauncherAdapter(launcher, format_)

    def configure_first_compatible_launcher(
        self,
        with_settings: _DispatchableT,
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
        self,
        launcher: LauncherLike[_LaunchableT],
        map_: t.Callable[[Unpack[_Ts]], _LaunchableT],
    ) -> None:
        # NOTE: We need to cast off the `_LaunchableT` -> `Any` in the
        #       `__init__` method signature to hide the transform from users of
        #       this class. If possible, this type should not be exposed to
        #       users of this class!
        self._adapt: t.Callable[[Unpack[_Ts]], t.Any] = map_
        self._adapted_launcher: LauncherLike[t.Any] = launcher

    def start(self, *args: Unpack[_Ts]) -> LaunchedJobID:
        payload = self._adapt(*args)
        return self._adapted_launcher.start(payload)


DEFAULT_DISPATCHER: t.Final = Dispatcher()
# Disabling because we want this to look and feel like a top level function,
# but don't want to have a second copy of the nasty overloads
# pylint: disable-next=invalid-name
dispatch: t.Final = DEFAULT_DISPATCHER.dispatch


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
    def create(cls, exp: Experiment, /) -> Self: ...


def make_shell_format_fn(
    run_command: str | None,
) -> _FormatterType[LaunchArgBuilder, t.Sequence[str]]:
    def impl(
        args: LaunchArgBuilder, exe: ExecutableLike, _env: _EnvironMappingType
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
        # pylint: disable-next=consider-using-with
        self._launched[id_] = sp.Popen((helpers.expand_exe_path(exe), *rest))
        return id_

    @classmethod
    def create(cls, _: Experiment) -> Self:
        return cls()


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
