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

import abc
import contextlib
import dataclasses
import io
import sys

import pytest

from smartsim._core import dispatch
from smartsim._core.utils.launcher import LauncherProtocol, create_job_id
from smartsim.error import errors

pytestmark = pytest.mark.group_a

FORMATTED = object()


def format_fn(args, exe, env):
    return FORMATTED


@pytest.fixture
def expected_dispatch_registry(mock_launcher, mock_launch_args):
    yield {
        type(mock_launch_args): dispatch._DispatchRegistration(
            format_fn, type(mock_launcher)
        )
    }


def test_declaritive_form_dispatch_declaration(
    mock_launcher, mock_launch_args, expected_dispatch_registry
):
    d = dispatch.Dispatcher()
    assert type(mock_launch_args) == d.dispatch(
        with_format=format_fn, to_launcher=type(mock_launcher)
    )(type(mock_launch_args))
    assert d._dispatch_registry == expected_dispatch_registry


def test_imperative_form_dispatch_declaration(
    mock_launcher, mock_launch_args, expected_dispatch_registry
):
    d = dispatch.Dispatcher()
    assert None == d.dispatch(
        type(mock_launch_args), to_launcher=type(mock_launcher), with_format=format_fn
    )
    assert d._dispatch_registry == expected_dispatch_registry


def test_dispatchers_from_same_registry_do_not_cross_polute(
    mock_launcher, mock_launch_args, expected_dispatch_registry
):
    some_starting_registry = {}
    d1 = dispatch.Dispatcher(dispatch_registry=some_starting_registry)
    d2 = dispatch.Dispatcher(dispatch_registry=some_starting_registry)
    assert (
        d1._dispatch_registry == d2._dispatch_registry == some_starting_registry == {}
    )
    assert (
        d1._dispatch_registry is not d2._dispatch_registry is not some_starting_registry
    )

    d2.dispatch(
        type(mock_launch_args), with_format=format_fn, to_launcher=type(mock_launcher)
    )
    assert d1._dispatch_registry == {}
    assert d2._dispatch_registry == expected_dispatch_registry


def test_copied_dispatchers_do_not_cross_pollute(
    mock_launcher, mock_launch_args, expected_dispatch_registry
):
    some_starting_registry = {}
    d1 = dispatch.Dispatcher(dispatch_registry=some_starting_registry)
    d2 = d1.copy()
    assert (
        d1._dispatch_registry == d2._dispatch_registry == some_starting_registry == {}
    )
    assert (
        d1._dispatch_registry is not d2._dispatch_registry is not some_starting_registry
    )

    d2.dispatch(
        type(mock_launch_args), to_launcher=type(mock_launcher), with_format=format_fn
    )
    assert d1._dispatch_registry == {}
    assert d2._dispatch_registry == expected_dispatch_registry


@pytest.mark.parametrize(
    "add_dispatch, expected_ctx",
    (
        pytest.param(
            lambda d, s, l: d.dispatch(s, to_launcher=l, with_format=format_fn),
            pytest.raises(TypeError, match="has already been registered"),
            id="Imperative -- Disallowed implicitly",
        ),
        pytest.param(
            lambda d, s, l: d.dispatch(
                s, to_launcher=l, with_format=format_fn, allow_overwrite=True
            ),
            contextlib.nullcontext(),
            id="Imperative -- Allowed with flag",
        ),
        pytest.param(
            lambda d, s, l: d.dispatch(to_launcher=l, with_format=format_fn)(s),
            pytest.raises(TypeError, match="has already been registered"),
            id="Declarative -- Disallowed implicitly",
        ),
        pytest.param(
            lambda d, s, l: d.dispatch(
                to_launcher=l, with_format=format_fn, allow_overwrite=True
            )(s),
            contextlib.nullcontext(),
            id="Declarative -- Allowed with flag",
        ),
    ),
)
def test_dispatch_overwriting(
    add_dispatch,
    expected_ctx,
    mock_launcher,
    mock_launch_args,
    expected_dispatch_registry,
):
    d = dispatch.Dispatcher(dispatch_registry=expected_dispatch_registry)
    with expected_ctx:
        add_dispatch(d, type(mock_launch_args), type(mock_launcher))


@pytest.mark.parametrize(
    "type_or_instance",
    (
        pytest.param(type, id="type"),
        pytest.param(lambda x: x, id="instance"),
    ),
)
def test_dispatch_can_retrieve_dispatch_info_from_dispatch_registry(
    expected_dispatch_registry, mock_launcher, mock_launch_args, type_or_instance
):
    d = dispatch.Dispatcher(dispatch_registry=expected_dispatch_registry)
    assert dispatch._DispatchRegistration(
        format_fn, type(mock_launcher)
    ) == d.get_dispatch(type_or_instance(mock_launch_args))


@pytest.mark.parametrize(
    "type_or_instance",
    (
        pytest.param(type, id="type"),
        pytest.param(lambda x: x, id="instance"),
    ),
)
def test_dispatch_raises_if_settings_type_not_registered(
    mock_launch_args, type_or_instance
):
    d = dispatch.Dispatcher(dispatch_registry={})
    with pytest.raises(
        TypeError, match="No dispatch for `.+?(?=`)` has been registered"
    ):
        d.get_dispatch(type_or_instance(mock_launch_args))


class LauncherABC(abc.ABC):
    @abc.abstractmethod
    def start(self, launchable): ...
    @classmethod
    @abc.abstractmethod
    def create(cls, exp): ...


class PartImplLauncherABC(LauncherABC):
    def start(self, launchable):
        return create_job_id()


class FullImplLauncherABC(PartImplLauncherABC):
    @classmethod
    def create(cls, exp):
        return cls()


@pytest.mark.parametrize(
    "cls, ctx",
    (
        pytest.param(
            LauncherProtocol,
            pytest.raises(TypeError, match="Cannot dispatch to protocol"),
            id="Cannot dispatch to protocol class",
        ),
        pytest.param(
            "mock_launcher",
            contextlib.nullcontext(None),
            id="Can dispatch to protocol implementation",
        ),
        pytest.param(
            LauncherABC,
            pytest.raises(TypeError, match="Cannot dispatch to abstract class"),
            id="Cannot dispatch to abstract class",
        ),
        pytest.param(
            PartImplLauncherABC,
            pytest.raises(TypeError, match="Cannot dispatch to abstract class"),
            id="Cannot dispatch to partially implemented abstract class",
        ),
        pytest.param(
            FullImplLauncherABC,
            contextlib.nullcontext(None),
            id="Can dispatch to fully implemented abstract class",
        ),
    ),
)
def test_register_dispatch_to_launcher_types(request, cls, ctx):
    if isinstance(cls, str):
        cls = request.getfixturevalue(cls)
    d = dispatch.Dispatcher()
    with ctx:
        d.dispatch(to_launcher=cls, with_format=format_fn)


@dataclasses.dataclass(frozen=True)
class BufferWriterLauncher(LauncherProtocol[list[str]]):
    buf: io.StringIO

    if sys.version_info < (3, 10):
        __hash__ = object.__hash__

    @classmethod
    def create(cls, exp):
        return cls(io.StringIO())

    def start(self, strs):
        self.buf.writelines(f"{s}\n" for s in strs)
        return create_job_id()

    def get_status(self, *ids):
        raise NotImplementedError

    def stop_jobs(self, *ids):
        raise NotImplementedError


class BufferWriterLauncherSubclass(BufferWriterLauncher): ...


@pytest.fixture
def buffer_writer_dispatch():
    stub_format_fn = lambda *a, **kw: ["some", "strings"]
    return dispatch._DispatchRegistration(stub_format_fn, BufferWriterLauncher)


@pytest.mark.parametrize(
    "input_, map_,  expected",
    (
        pytest.param(
            ["list", "of", "strings"],
            lambda xs: xs,
            ["list\n", "of\n", "strings\n"],
            id="[str] -> [str]",
        ),
        pytest.param(
            "words on new lines",
            lambda x: x.split(),
            ["words\n", "on\n", "new\n", "lines\n"],
            id="str -> [str]",
        ),
        pytest.param(
            range(1, 4),
            lambda xs: [str(x) for x in xs],
            ["1\n", "2\n", "3\n"],
            id="[int] -> [str]",
        ),
    ),
)
def test_launcher_adapter_correctly_adapts_input_to_launcher(input_, map_, expected):
    buf = io.StringIO()
    adapter = dispatch._LauncherAdapter(BufferWriterLauncher(buf), map_)
    adapter.start(input_)
    buf.seek(0)
    assert buf.readlines() == expected


@pytest.mark.parametrize(
    "launcher_instance, ctx",
    (
        pytest.param(
            BufferWriterLauncher(io.StringIO()),
            contextlib.nullcontext(None),
            id="Correctly configures expected launcher",
        ),
        pytest.param(
            BufferWriterLauncherSubclass(io.StringIO()),
            pytest.raises(
                TypeError,
                match="^Cannot create launcher adapter.*expected launcher of type .+$",
            ),
            id="Errors if launcher types are disparate",
        ),
        pytest.param(
            "mock_launcher",
            pytest.raises(
                TypeError,
                match="^Cannot create launcher adapter.*expected launcher of type .+$",
            ),
            id="Errors if types are not an exact match",
        ),
    ),
)
def test_dispatch_registration_can_configure_adapter_for_existing_launcher_instance(
    request, mock_launch_args, buffer_writer_dispatch, launcher_instance, ctx
):
    if isinstance(launcher_instance, str):
        launcher_instance = request.getfixturevalue(launcher_instance)
    with ctx:
        adapter = buffer_writer_dispatch.create_adapter_from_launcher(
            launcher_instance, mock_launch_args
        )
        assert adapter._adapted_launcher is launcher_instance


@pytest.mark.parametrize(
    "launcher_instances, ctx",
    (
        pytest.param(
            (BufferWriterLauncher(io.StringIO()),),
            contextlib.nullcontext(None),
            id="Correctly configures expected launcher",
        ),
        pytest.param(
            (
                "mock_launcher",
                "mock_launcher",
                BufferWriterLauncher(io.StringIO()),
                "mock_launcher",
            ),
            contextlib.nullcontext(None),
            id="Correctly ignores incompatible launchers instances",
        ),
        pytest.param(
            (),
            pytest.raises(
                errors.LauncherNotFoundError,
                match="^No launcher of exactly type.+could be found from provided launchers$",
            ),
            id="Errors if no launcher could be found",
        ),
        pytest.param(
            (
                "mock_launcher",
                BufferWriterLauncherSubclass(io.StringIO),
                "mock_launcher",
            ),
            pytest.raises(
                errors.LauncherNotFoundError,
                match="^No launcher of exactly type.+could be found from provided launchers$",
            ),
            id="Errors if no launcher matches expected type exactly",
        ),
    ),
)
def test_dispatch_registration_configures_first_compatible_launcher_from_sequence_of_launchers(
    request, mock_launch_args, buffer_writer_dispatch, launcher_instances, ctx
):
    def resolve_instance(inst):
        return request.getfixturevalue(inst) if isinstance(inst, str) else inst

    launcher_instances = tuple(map(resolve_instance, launcher_instances))

    with ctx:
        adapter = buffer_writer_dispatch.configure_first_compatible_launcher(
            with_arguments=mock_launch_args, from_available_launchers=launcher_instances
        )


def test_dispatch_registration_can_create_a_laucher_for_an_experiment_and_can_reconfigure_it_later(
    mock_launch_args, buffer_writer_dispatch
):
    class MockExperiment: ...

    exp = MockExperiment()
    adapter_1 = buffer_writer_dispatch.create_new_launcher_configuration(
        for_experiment=exp, with_arguments=mock_launch_args
    )
    assert type(adapter_1._adapted_launcher) == buffer_writer_dispatch.launcher_type
    existing_launcher = adapter_1._adapted_launcher

    adapter_2 = buffer_writer_dispatch.create_adapter_from_launcher(
        existing_launcher, mock_launch_args
    )
    assert type(adapter_2._adapted_launcher) == buffer_writer_dispatch.launcher_type
    assert adapter_1._adapted_launcher is adapter_2._adapted_launcher
    assert adapter_1 is not adapter_2
