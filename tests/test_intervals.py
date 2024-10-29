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

import contextlib
import operator
import time

import pytest

from smartsim._core.control.interval import SynchronousTimeInterval

pytestmark = pytest.mark.group_a


@pytest.mark.parametrize(
    "timeout", [pytest.param(i, id=f"{i} second(s)") for i in range(10)]
)
def test_sync_timeout_finite(timeout, monkeypatch):
    """Test that the sync timeout intervals are correctly calculated"""
    monkeypatch.setattr(time, "perf_counter", lambda *_, **__: 0)
    t = SynchronousTimeInterval(timeout)
    assert t.delta == timeout
    assert t.elapsed == 0
    assert t.remaining == timeout
    assert (operator.not_ if timeout > 0 else bool)(t.expired)
    assert not t.infinite
    future = timeout + 2
    monkeypatch.setattr(time, "perf_counter", lambda *_, **__: future)
    assert t.elapsed == future
    assert t.remaining == 0
    assert t.expired
    assert not t.infinite
    new_t = t.new_interval()
    assert new_t.delta == timeout
    assert new_t.elapsed == 0
    assert new_t.remaining == timeout
    assert (operator.not_ if timeout > 0 else bool)(new_t.expired)
    assert not new_t.infinite


def test_sync_timeout_can_block_thread():
    """Test that the sync timeout can block the calling thread"""
    timeout = 1
    now = time.perf_counter()
    SynchronousTimeInterval(timeout).block()
    later = time.perf_counter()
    assert abs(later - now - timeout) <= 0.25


def test_sync_timeout_infinte():
    """Passing in `None` to a sync timeout creates a timeout with an infinite
    delta time
    """
    t = SynchronousTimeInterval(None)
    assert t.remaining == float("inf")
    assert t.infinite
    with pytest.raises(RuntimeError, match="block thread forever"):
        t.block()


def test_sync_timeout_raises_on_invalid_value(monkeypatch):
    """Cannot make a sync time interval with a negative time delta"""
    with pytest.raises(ValueError):
        SynchronousTimeInterval(-1)
