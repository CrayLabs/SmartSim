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

import pytest
from smartsim.settings import dispatch

import contextlib


def test_declaritive_form_dispatch_declaration(launcher_like, settings_builder):
    d = dispatch.Dispatcher()
    assert type(settings_builder) == d.dispatch(to_launcher=type(launcher_like))(
        type(settings_builder)
    )
    assert d._dispatch_registry == {type(settings_builder): type(launcher_like)}


def test_imperative_form_dispatch_declaration(launcher_like, settings_builder):
    d = dispatch.Dispatcher()
    assert None == d.dispatch(type(settings_builder), to_launcher=type(launcher_like))
    assert d._dispatch_registry == {type(settings_builder): type(launcher_like)}


def test_dispatchers_from_same_registry_do_not_cross_polute(
    launcher_like, settings_builder
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

    d2.dispatch(type(settings_builder), to_launcher=type(launcher_like))
    assert d1._dispatch_registry == {}
    assert d2._dispatch_registry == {type(settings_builder): type(launcher_like)}


def test_copied_dispatchers_do_not_cross_pollute(launcher_like, settings_builder):
    some_starting_registry = {}
    d1 = dispatch.Dispatcher(dispatch_registry=some_starting_registry)
    d2 = d1.copy()
    assert (
        d1._dispatch_registry == d2._dispatch_registry == some_starting_registry == {}
    )
    assert (
        d1._dispatch_registry is not d2._dispatch_registry is not some_starting_registry
    )

    d2.dispatch(type(settings_builder), to_launcher=type(launcher_like))
    assert d1._dispatch_registry == {}
    assert d2._dispatch_registry == {type(settings_builder): type(launcher_like)}


@pytest.mark.parametrize(
    "add_dispatch, expected_ctx",
    (
        pytest.param(
            lambda d, s, l: d.dispatch(s, to_launcher=l),
            pytest.raises(TypeError, match="has already been registered"),
            id="Imperative -- Disallowed implicitly",
        ),
        pytest.param(
            lambda d, s, l: d.dispatch(s, to_launcher=l, allow_overwrite=True),
            contextlib.nullcontext(),
            id="Imperative -- Allowed with flag",
        ),
        pytest.param(
            lambda d, s, l: d.dispatch(to_launcher=l)(s),
            pytest.raises(TypeError, match="has already been registered"),
            id="Declarative -- Disallowed implicitly",
        ),
        pytest.param(
            lambda d, s, l: d.dispatch(to_launcher=l, allow_overwrite=True)(s),
            contextlib.nullcontext(),
            id="Declarative -- Allowed with flag",
        ),
    ),
)
def test_dispatch_overwriting(
    add_dispatch, expected_ctx, launcher_like, settings_builder
):
    registry = {type(settings_builder): type(launcher_like)}
    d = dispatch.Dispatcher(dispatch_registry=registry)
    with expected_ctx:
        add_dispatch(d, type(settings_builder), type(launcher_like))


@pytest.mark.parametrize(
    "map_settings",
    (
        pytest.param(type, id="From settings type"),
        pytest.param(lambda s: s, id="From settings instance"),
    ),
)
def test_dispatch_can_retrieve_launcher_to_dispatch_to(
    map_settings, launcher_like, settings_builder
):
    registry = {type(settings_builder): type(launcher_like)}
    d = dispatch.Dispatcher(dispatch_registry=registry)
    assert type(launcher_like) == d.get_launcher_for(map_settings(settings_builder))


@pytest.mark.parametrize(
    "map_settings",
    (
        pytest.param(type, id="From settings type"),
        pytest.param(lambda s: s, id="From settings instance"),
    ),
)
def test_dispatch_raises_if_settings_type_not_registered(
    map_settings, launcher_like, settings_builder
):
    d = dispatch.Dispatcher(dispatch_registry={})
    with pytest.raises(TypeError, match="no launcher type to dispatch to"):
        d.get_launcher_for(map_settings(settings_builder))
