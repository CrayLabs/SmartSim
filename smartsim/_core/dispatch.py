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
import os
import pathlib
import typing as t

from typing_extensions import Self, TypeAlias, TypeVarTuple, Unpack

from smartsim._core.utils import helpers
from smartsim.error import errors
from smartsim.types import LaunchedJobID

if t.TYPE_CHECKING:
    from smartsim._core.arguments.shell import ShellLaunchArguments
    from smartsim._core.utils.launcher import LauncherProtocol
    from smartsim.experiment import Experiment
    from smartsim.settings.arguments import LaunchArguments


_Ts = TypeVarTuple("_Ts")


WorkingDirectory: TypeAlias = pathlib.Path
"""A working directory represented as a string or PathLike object"""

_DispatchableT = t.TypeVar("_DispatchableT", bound="LaunchArguments")
"""Any type of luanch arguments, typically used when the type bound by the type
argument is a key a `Dispatcher` dispatch registry
"""
_LaunchableT = t.TypeVar("_LaunchableT")
"""Any type, typically used to bind to a type accepted as the input parameter
to the to the `LauncherProtocol.start` method
"""

EnvironMappingType: TypeAlias = t.Mapping[str, "str | None"]
"""A mapping of user provided mapping of environment variables in which to run
a job
"""
FormatterType: TypeAlias = t.Callable[
    [
        _DispatchableT,
        t.Sequence[str],
        WorkingDirectory,
        EnvironMappingType,
        pathlib.Path,
        pathlib.Path,
    ],
    _LaunchableT,
]
"""A callable that is capable of formatting the components of a job into a type
capable of being launched by a launcher.
"""
_LaunchConfigType: TypeAlias = """_LauncherAdapter[
        t.Sequence[str],
        WorkingDirectory,
        EnvironMappingType,
        pathlib.Path,
        pathlib.Path]"""

"""A launcher adapater that has configured a launcher to launch the components
of a job with some pre-determined launch settings
"""
_UnkownType: TypeAlias = t.NoReturn
"""A type alias for a bottom type. Use this to inform a user that the parameter
a parameter should never be set or a callable will never return
"""


@t.final
class Dispatcher:
    """A class capable of deciding which launcher type should be used to launch
    a given settings type.

    The `Dispatcher` class maintains a type safe API for adding and retrieving
    a settings type into the underlying mapping. It does this through two main
    methods: `Dispatcher.dispatch` and `Dispatcher.get_dispatch`.

    `Dispatcher.dispatch` takes in a dispatchable type, a launcher type that is
    capable of launching a launchable type and formatting function that maps an
    instance of the dispatchable type to an instance of the launchable type.
    The dispatcher will then take these components and then enter them into its
    dispatch registry. `Dispatcher.dispatch` can also be used as a decorator,
    to automatically add a dispatchable type dispatch to a dispatcher at type
    creation time.

    `Dispatcher.get_dispatch` takes a dispatchable type or instance as a
    parameter, and will attempt to look up, in its dispatch registry, how to
    dispatch that type. It will then return an object that can configure a
    launcher of the expected launcher type. If the dispatchable type was never
    registered a `TypeError` will be raised.
    """

    def __init__(
        self,
        *,
        dispatch_registry: (
            t.Mapping[type[LaunchArguments], _DispatchRegistration[t.Any, t.Any]] | None
        ) = None,
    ) -> None:
        """Initialize a new `Dispatcher`

        :param dispatch_registry: A pre-configured dispatch registry that the
            dispatcher should use. This registry is not type checked and is
            used blindly. This registry is shallow copied, meaning that adding
            into the original registry after construction will not mutate the
            state of the registry.
        """
        self._dispatch_registry = (
            dict(dispatch_registry) if dispatch_registry is not None else {}
        )

    def copy(self) -> Self:
        """Create a shallow copy of the Dispatcher"""
        return type(self)(dispatch_registry=self._dispatch_registry)

    @t.overload
    def dispatch(  # Signature when used as a decorator
        self,
        args: None = ...,
        *,
        with_format: FormatterType[_DispatchableT, _LaunchableT],
        to_launcher: type[LauncherProtocol[_LaunchableT]],
        allow_overwrite: bool = ...,
    ) -> t.Callable[[type[_DispatchableT]], type[_DispatchableT]]: ...
    @t.overload
    def dispatch(  # Signature when used as a method
        self,
        args: type[_DispatchableT],
        *,
        with_format: FormatterType[_DispatchableT, _LaunchableT],
        to_launcher: type[LauncherProtocol[_LaunchableT]],
        allow_overwrite: bool = ...,
    ) -> None: ...
    def dispatch(  # Actual implementation
        self,
        args: type[_DispatchableT] | None = None,
        *,
        with_format: FormatterType[_DispatchableT, _LaunchableT],
        to_launcher: type[LauncherProtocol[_LaunchableT]],
        allow_overwrite: bool = False,
    ) -> t.Callable[[type[_DispatchableT]], type[_DispatchableT]] | None:
        """A type safe way to add a mapping of settings type to launcher type
        to handle a settings instance at launch time.
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
        """Find a type of launcher that is registered as being able to launch a
        settings instance of the provided type
        """
        if not isinstance(args, type):
            args = type(args)
        dispatch_ = self._dispatch_registry.get(args, None)
        if dispatch_ is None:
            raise TypeError(
                f"No dispatch for `{args.__name__}` has been registered "
                f"has been registered with {type(self).__name__} `{self}`"
            )
        # Note the sleight-of-hand here: we are secretly casting a type of
        # `_DispatchRegistration[Any, Any]` ->
        #     `_DispatchRegistration[_DispatchableT, _LaunchableT]`.
        #  where `_LaunchableT` is unbound!
        #
        # This is safe to do if all entries in the mapping were added using a
        # type safe method (e.g. `Dispatcher.dispatch`), but if a user were to
        # supply a custom dispatch registry or otherwise modify the registry
        # this is not necessarily 100% type safe!!
        return dispatch_


@t.final
@dataclasses.dataclass(frozen=True)
class _DispatchRegistration(t.Generic[_DispatchableT, _LaunchableT]):
    """An entry into the `Dispatcher`'s dispatch registry. This class is simply
    a wrapper around a launcher and how to format a `_DispatchableT` instance
    to be launched by the afore mentioned launcher.
    """

    formatter: FormatterType[_DispatchableT, _LaunchableT]
    launcher_type: type[LauncherProtocol[_LaunchableT]]

    def _is_compatible_launcher(self, launcher: LauncherProtocol[t.Any]) -> bool:
        # Disabling because we want to match the type of the dispatch
        # *exactly* as specified by the user
        # pylint: disable-next=unidiomatic-typecheck
        return type(launcher) is self.launcher_type

    def create_new_launcher_configuration(
        self, for_experiment: Experiment, with_arguments: _DispatchableT
    ) -> _LaunchConfigType:
        """Create a new instance of a launcher for an experiment that the
        provided settings were set to dispatch, and configure it with the
        provided launch settings.

        :param for_experiment: The experiment responsible creating the launcher
        :param with_settings: The settings with which to configure the newly
            created launcher
        :returns: A configured launcher
        """
        launcher = self.launcher_type.create(for_experiment)
        return self.create_adapter_from_launcher(launcher, with_arguments)

    def create_adapter_from_launcher(
        self, launcher: LauncherProtocol[_LaunchableT], arguments: _DispatchableT
    ) -> _LaunchConfigType:
        """Creates configured launcher from an existing launcher using the
        provided settings.

        :param launcher: A launcher that the type of `settings` has been
            configured to dispatch to.
        :param settings: A settings with which to configure the launcher.
        :returns: A configured launcher.
        """
        if not self._is_compatible_launcher(launcher):
            raise TypeError(
                f"Cannot create launcher adapter from launcher `{launcher}` "
                f"of type `{type(launcher)}`; expected launcher of type "
                f"exactly `{self.launcher_type}`"
            )

        def format_(
            exe: t.Sequence[str],
            path: pathlib.Path,
            env: EnvironMappingType,
            out: pathlib.Path,
            err: pathlib.Path,
        ) -> _LaunchableT:
            return self.formatter(arguments, exe, path, env, out, err)

        return _LauncherAdapter(launcher, format_)

    def configure_first_compatible_launcher(
        self,
        with_arguments: _DispatchableT,
        from_available_launchers: t.Iterable[LauncherProtocol[t.Any]],
    ) -> _LaunchConfigType:
        """Configure the first compatible adapter launch to launch with the
        provided settings. Launchers are iterated and discarded from the
        iterator until the iterator is exhausted.

        :param with_settings: The settings with which to configure the launcher
        :param from_available_launchers: An iterable that yields launcher instances
        :raises errors.LauncherNotFoundError: No compatible launcher was
            yielded from the provided iterator.
        :returns: A launcher configured with the provided settings.
        """
        launcher = helpers.first(self._is_compatible_launcher, from_available_launchers)
        if launcher is None:
            raise errors.LauncherNotFoundError(
                f"No launcher of exactly type `{self.launcher_type.__name__}` "
                "could be found from provided launchers"
            )
        return self.create_adapter_from_launcher(launcher, with_arguments)


@t.final
class _LauncherAdapter(t.Generic[Unpack[_Ts]]):
    """The launcher adapter is an adapter class takes a launcher that is
    capable of launching some type `LaunchableT` and a function with a generic
    argument list that returns a `LaunchableT`. The launcher adapter will then
    provide `start` method that will have the same argument list as the
    provided function and launch the output through the provided launcher.

    For example, the launcher adapter could be used like so:

    .. highlight:: python
    .. code-block:: python

        class SayHelloLauncher(LauncherProtocol[str]):
            ...
            def start(self, title: str):
                ...
                print(f"Hello, {title}")
                ...
            ...

        @dataclasses.dataclass
        class Person:
            name: str
            honorific: str

            def full_title(self) -> str:
                return f"{honorific}. {self.name}"

        mark = Person("Jim", "Mr")
        sally = Person("Sally", "Ms")
        matt = Person("Matt", "Dr")
        hello_person_launcher = _LauncherAdapter(SayHelloLauncher,
                                                 Person.full_title)

        hello_person_launcher.start(mark)   # prints: "Hello, Mr. Mark"
        hello_person_launcher.start(sally)  # prints: "Hello, Ms. Sally"
        hello_person_launcher.start(matt)   # prints: "Hello, Dr. Matt"
    """

    def __init__(
        self,
        launcher: LauncherProtocol[_LaunchableT],
        map_: t.Callable[[Unpack[_Ts]], _LaunchableT],
    ) -> None:
        """Initialize a launcher adapter

        :param launcher: The launcher instance this class should wrap
        :param map_: A callable with arguments for the new `start` method that
            can translate them into the expected launching type for the wrapped
            launcher.
        """
        # NOTE: We need to cast off the `_LaunchableT` -> `Any` in the
        #       `__init__` method signature to hide the transform from users of
        #       this class. If possible, this type should not be exposed to
        #       users of this class!
        self._adapt: t.Callable[[Unpack[_Ts]], t.Any] = map_
        self._adapted_launcher: LauncherProtocol[t.Any] = launcher

    def start(self, *args: Unpack[_Ts]) -> LaunchedJobID:
        """Start a new job through the wrapped launcher using the custom
        `start` signature

        :param args: The custom start arguments
        :returns: The launched job id provided by the wrapped launcher
        """
        payload = self._adapt(*args)
        return self._adapted_launcher.start(payload)


DEFAULT_DISPATCHER: t.Final = Dispatcher()
"""A global `Dispatcher` instance that SmartSim automatically configures to
launch its built in launchables
"""

# Disabling because we want this to look and feel like a top level function,
# but don't want to have a second copy of the nasty overloads
# pylint: disable-next=invalid-name
dispatch: t.Final = DEFAULT_DISPATCHER.dispatch
"""Function that can be used as a decorator to add a dispatch registration into
`DEFAULT_DISPATCHER`.
"""
