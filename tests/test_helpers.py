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

import collections
import signal

import pytest

from smartsim._core.utils import helpers
from smartsim._core.utils.helpers import cat_arg_and_value

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_double_dash_concat():
    result = cat_arg_and_value("--foo", "FOO")
    assert result == "--foo=FOO"


def test_single_dash_concat():
    result = cat_arg_and_value("-foo", "FOO")
    assert result == "-foo FOO"


def test_single_char_concat():
    result = cat_arg_and_value("x", "FOO")
    assert result == "-x FOO"


def test_fallthrough_concat():
    result = cat_arg_and_value("xx", "FOO")  # <-- no dashes, > 1 char
    assert result == "--xx=FOO"


def test_encode_decode_cmd_round_trip():
    orig_cmd = ["this", "is", "a", "cmd"]
    decoded_cmd = helpers.decode_cmd(helpers.encode_cmd(orig_cmd))
    assert orig_cmd == decoded_cmd
    assert orig_cmd is not decoded_cmd


def test_encode_raises_on_empty():
    with pytest.raises(ValueError):
        helpers.encode_cmd([])


def test_decode_raises_on_empty():
    with pytest.raises(ValueError):
        helpers.decode_cmd("")


class MockSignal:
    def __init__(self):
        self.signal_handlers = collections.defaultdict(lambda: signal.SIG_IGN)

    def signal(self, signalnum, handler):
        orig = self.getsignal(signalnum)
        self.signal_handlers[signalnum] = handler
        return orig

    def getsignal(self, signalnum):
        return self.signal_handlers[signalnum]


@pytest.fixture
def mock_signal(monkeypatch):
    mock_signal = MockSignal()
    monkeypatch.setattr(helpers, "signal", mock_signal)
    yield mock_signal


def test_signal_intercept_stack_will_register_itself_with_callback_fn(mock_signal):
    callback = lambda num, frame: ...
    stack = helpers.SignalInterceptionStack.push(signal.NSIG, callback)
    assert isinstance(stack, helpers.SignalInterceptionStack)
    assert stack is mock_signal.signal_handlers[signal.NSIG]
    assert len(stack._callbacks) == 1
    assert stack._callbacks[0] == callback


def test_signal_intercept_stack_keeps_track_of_previous_handlers(mock_signal):
    default_handler = lambda num, frame: ...
    mock_signal.signal_handlers[signal.NSIG] = default_handler
    stack = helpers.SignalInterceptionStack.push(signal.NSIG, lambda n, f: ...)
    assert stack._original is default_handler


def test_signal_intercept_stacks_are_registered_per_signal_number(mock_signal):
    handler = lambda num, frame: ...
    stack_1 = helpers.SignalInterceptionStack.push(signal.NSIG, handler)
    stack_2 = helpers.SignalInterceptionStack.push(signal.NSIG + 1, handler)

    assert mock_signal.signal_handlers[signal.NSIG] is stack_1
    assert mock_signal.signal_handlers[signal.NSIG + 1] is stack_2
    assert stack_1 is not stack_2
    assert stack_1._callbacks == stack_2._callbacks == [handler]


def test_signal_intercept_handlers_will_not_overwrite_if_handler_already_exists(
    mock_signal,
):
    handler_1 = lambda num, frame: ...
    handler_2 = lambda num, frame: ...
    stack_1 = helpers.SignalInterceptionStack.push(signal.NSIG, handler_1)
    stack_2 = helpers.SignalInterceptionStack.push(signal.NSIG, handler_2)
    assert stack_1 is stack_2 is mock_signal.signal_handlers[signal.NSIG]
    assert list(stack_1._callbacks) == [handler_1, handler_2]


def test_signal_intercept_stack_enforces_that_handlers_are_unique(mock_signal):
    handler = lambda num, frame: ...
    stack = helpers.SignalInterceptionStack.push(signal.NSIG, handler)
    helpers.SignalInterceptionStack.push(signal.NSIG, handler)
    assert list(stack._callbacks) == [handler]


def test_singnal_intercept_stack_enforces_that_method_handlers_are_unique(mock_signal):
    class C:
        def fn(num, frame): ...

    c = C()
    stack = helpers.SignalInterceptionStack.push(signal.NSIG, c.fn)
    helpers.SignalInterceptionStack.push(signal.NSIG, c.fn)
    assert list(stack._callbacks) == [c.fn]


def test_signal_handler_calls_functions_in_reverse_order(mock_signal):
    called_list = []
    default = lambda num, frame: called_list.append("default")
    handler_1 = lambda num, frame: called_list.append("handler_1")
    handler_2 = lambda num, frame: called_list.append("handler_2")

    mock_signal.signal_handlers[signal.NSIG] = default
    helpers.SignalInterceptionStack.push(signal.NSIG, handler_1)
    helpers.SignalInterceptionStack.push(signal.NSIG, handler_2)
    mock_signal.signal_handlers[signal.NSIG](signal.NSIG, None)
    assert called_list == ["handler_2", "handler_1", "default"]
