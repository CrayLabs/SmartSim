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
