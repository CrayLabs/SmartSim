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


import packaging
import pytest

from smartsim._core._install.buildenv import Version_

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


def test_version_hash_eq():
    """Ensure hashes are equal if data is equal"""
    v1 = Version_("1.2.3")
    v2 = Version_("1.2.3")

    hash1 = hash(v1)
    hash2 = hash(v2)

    assert hash1 == hash2


def test_version_hash_ne():
    """Ensure hashes are NOT equal if data is different"""
    v1 = Version_("1.2.3")
    v2 = Version_("1.2.4")

    hash1 = hash(v1)
    hash2 = hash(v2)

    assert hash1 != hash2


def test_version_equality_eq():
    """Test equality operator on items expected to be equal"""
    v1 = Version_("1.2.3")
    v2 = Version_("1.2.3")

    assert v1 == v2


def test_version_equality_ne():
    """Test equality operator on items expected to be unequal"""
    v1 = Version_("1.2.3")
    v2 = Version_("1.2.4")

    assert v1 != v2

    # def test_version_bad_input():
    """Test behavior when passing an invalid version string"""
    version = Version_("1")
    assert version.major == 1
    with pytest.raises((IndexError, packaging.version.InvalidVersion)) as ex:
        version.minor

    version = Version_("2.")
    with pytest.raises((IndexError, packaging.version.InvalidVersion)) as ex:
        version.major

    version = Version_("3.0.")

    with pytest.raises((IndexError, packaging.version.InvalidVersion)) as ex:
        version.major

    version = Version_("3.1.a")
    assert version.major == 3
    assert version.minor == 1
    with pytest.raises((IndexError, packaging.version.InvalidVersion)) as ex:
        version.patch


def test_version_bad_parse_fail():
    """Test behavior when trying to parse with an invalid input string"""
    # todo: ensure we can't take invalid input and have this IndexError occur.
    version = Version_("abcdefg")
    with pytest.raises((IndexError, packaging.version.InvalidVersion)) as ex:
        version.major
