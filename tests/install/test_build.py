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

import operator

import pytest

from smartsim._core._cli.build import parse_requirement
from smartsim._core._install.buildenv import Version_

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


_SUPPORTED_OPERATORS = ("==", ">=", ">", "<=", "<")


@pytest.mark.parametrize(
    "spec, name, version",
    (
        pytest.param("foo", "foo", None, id="Just Name"),
        pytest.param("foo==1", "foo", Version_("1"), id="With Major"),
        pytest.param("foo==1.2", "foo", Version_("1.2"), id="With Minor"),
        pytest.param("foo==1.2.3", "foo", Version_("1.2.3"), id="With Patch"),
        *(
            pytest.param(
                f"foo{symbol}1.2.3{tag}",
                "foo",
                Version_(f"1.2.3{tag}"),
                id=f"{symbol=} | {tag=}",
            )
            for symbol in _SUPPORTED_OPERATORS
            for tag in ("", "+cuda", "+rocm", "+cpu")
        ),
    ),
)
def test_parse_requirement_name_and_version(spec, name, version):
    p_name, p_version, _ = parse_requirement(spec)
    assert p_name == name
    assert p_version == version


# fmt: off
@pytest.mark.parametrize(
    "spec, ver, should_pass",
    (
        pytest.param("foo"            , Version_("1.2.3")     ,  True, id="No spec"),
        # EQ --------------------------------------------------------------------------
        pytest.param("foo==1.2.3"     , Version_("1.2.3")     ,  True, id="EQ Spec, EQ Version"),
        pytest.param("foo==1.2.3"     , Version_("1.2.5")     , False, id="EQ Spec, GT Version"),
        pytest.param("foo==1.2.3"     , Version_("1.2.2")     , False, id="EQ Spec, LT Version"),
        pytest.param("foo==1.2.3+rocm", Version_("1.2.3+rocm"),  True, id="EQ Spec, Compatable Version with suffix"),
        pytest.param("foo==1.2.3"     , Version_("1.2.3+cuda"), False, id="EQ Spec, Compatable Version, Extra Suffix"),
        pytest.param("foo==1.2.3+cuda", Version_("1.2.3")     , False, id="EQ Spec, Compatable Version, Missing Suffix"),
        pytest.param("foo==1.2.3+cuda", Version_("1.2.3+rocm"), False, id="EQ Spec, Compatable Version, Mismatched Suffix"),
        # LT --------------------------------------------------------------------------
        pytest.param("foo<1.2.3"      , Version_("1.2.3")     , False, id="LT Spec, EQ Version"),
        pytest.param("foo<1.2.3"      , Version_("1.2.5")     , False, id="LT Spec, GT Version"),
        pytest.param("foo<1.2.3"      , Version_("1.2.2")     ,  True, id="LT Spec, LT Version"),
        pytest.param("foo<1.2.3+rocm" , Version_("1.2.2+rocm"),  True, id="LT Spec, Compatable Version with suffix"),
        pytest.param("foo<1.2.3"      , Version_("1.2.2+cuda"), False, id="LT Spec, Compatable Version, Extra Suffix"),
        pytest.param("foo<1.2.3+cuda" , Version_("1.2.2")     , False, id="LT Spec, Compatable Version, Missing Suffix"),
        pytest.param("foo<1.2.3+cuda" , Version_("1.2.2+rocm"), False, id="LT Spec, Compatable Version, Mismatched Suffix"),
        # LE --------------------------------------------------------------------------
        pytest.param("foo<=1.2.3"     , Version_("1.2.3")     ,  True, id="LE Spec, EQ Version"),
        pytest.param("foo<=1.2.3"     , Version_("1.2.5")     , False, id="LE Spec, GT Version"),
        pytest.param("foo<=1.2.3"     , Version_("1.2.2")     ,  True, id="LE Spec, LT Version"),
        pytest.param("foo<=1.2.3+rocm", Version_("1.2.3+rocm"),  True, id="LE Spec, Compatable Version with suffix"),
        pytest.param("foo<=1.2.3"     , Version_("1.2.3+cuda"), False, id="LE Spec, Compatable Version, Extra Suffix"),
        pytest.param("foo<=1.2.3+cuda", Version_("1.2.3")     , False, id="LE Spec, Compatable Version, Missing Suffix"),
        pytest.param("foo<=1.2.3+cuda", Version_("1.2.3+rocm"), False, id="LE Spec, Compatable Version, Mismatched Suffix"),
        # GT --------------------------------------------------------------------------
        pytest.param("foo>1.2.3"      , Version_("1.2.3")     , False, id="GT Spec, EQ Version"),
        pytest.param("foo>1.2.3"      , Version_("1.2.5")     ,  True, id="GT Spec, GT Version"),
        pytest.param("foo>1.2.3"      , Version_("1.2.2")     , False, id="GT Spec, LT Version"),
        pytest.param("foo>1.2.3+rocm" , Version_("1.2.4+rocm"),  True, id="GT Spec, Compatable Version with suffix"),
        pytest.param("foo>1.2.3"      , Version_("1.2.4+cuda"), False, id="GT Spec, Compatable Version, Extra Suffix"),
        pytest.param("foo>1.2.3+cuda" , Version_("1.2.4")     , False, id="GT Spec, Compatable Version, Missing Suffix"),
        pytest.param("foo>1.2.3+cuda" , Version_("1.2.4+rocm"), False, id="GT Spec, Compatable Version, Mismatched Suffix"),
        # GE --------------------------------------------------------------------------
        pytest.param("foo>=1.2.3"     , Version_("1.2.3")     ,  True, id="GE Spec, EQ Version"),
        pytest.param("foo>=1.2.3"     , Version_("1.2.5")     ,  True, id="GE Spec, GT Version"),
        pytest.param("foo>=1.2.3"     , Version_("1.2.2")     , False, id="GE Spec, LT Version"),
        pytest.param("foo>=1.2.3+rocm", Version_("1.2.3+rocm"),  True, id="GE Spec, Compatable Version with suffix"),
        pytest.param("foo>=1.2.3"     , Version_("1.2.3+cuda"), False, id="GE Spec, Compatable Version, Extra Suffix"),
        pytest.param("foo>=1.2.3+cuda", Version_("1.2.3")     , False, id="GE Spec, Compatable Version, Missing Suffix"),
        pytest.param("foo>=1.2.3+cuda", Version_("1.2.3+rocm"), False, id="GE Spec, Compatable Version, Mismatched Suffix"),
    )
)
# fmt: on
def test_parse_requirement_comparison_fn(spec, ver, should_pass):
    _, _, cmp = parse_requirement(spec)
    assert cmp(ver) == should_pass


@pytest.mark.parametrize(
    "spec, ctx",
    (
        *(
            pytest.param(
                f"thing{symbol}",
                pytest.raises(ValueError, match="Invalid requirement string:"),
                id=f"No version w/ operator {symbol}",
            )
            for symbol in _SUPPORTED_OPERATORS
        ),
        pytest.param(
            "thing>=>1.2.3",
            pytest.raises(ValueError, match="Invalid requirement string:"),
            id="Operator too long",
        ),
        pytest.param(
            "thing<>1.2.3",
            pytest.raises(ValueError, match="Unrecognized comparison operator: <>"),
            id="Nonsense operator",
        ),
    ),
)
def test_parse_requirement_errors_on_invalid_spec(spec, ctx):
    with ctx:
        parse_requirement(spec)
