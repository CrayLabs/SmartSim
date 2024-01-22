# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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


import functools
import pathlib
import platform
import threading
import time

import pytest

import smartsim._core._install.builder as build

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


for_each_device = pytest.mark.parametrize("device", ["cpu", "gpu"])

_toggle_build_optional_backend = lambda backend: pytest.mark.parametrize(
    f"build_{backend}",
    [
        pytest.param(switch, id=f"with{'' if switch else 'out'}-{backend}")
        for switch in (True, False)
    ],
)
toggle_build_tf = _toggle_build_optional_backend("tf")
toggle_build_pt = _toggle_build_optional_backend("pt")
toggle_build_ort = _toggle_build_optional_backend("ort")


@pytest.mark.parametrize(
    "mock_os", [pytest.param(os_, id=f"os='{os_}'") for os_ in ("Windows", "Java", "")]
)
def test_rai_builder_raises_on_unsupported_op_sys(monkeypatch, mock_os):
    monkeypatch.setattr(platform, "system", lambda: mock_os)
    with pytest.raises(build.BuildError, match="operating system") as err_info:
        build.RedisAIBuilder()


@pytest.mark.parametrize(
    "mock_arch",
    [
        pytest.param(arch_, id=f"arch='{arch_}'")
        for arch_ in ("i386", "i686", "i86pc", "aarch64", "arm64", "armv7l", "")
    ],
)
def test_rai_builder_raises_on_unsupported_architecture(monkeypatch, mock_arch):
    monkeypatch.setattr(platform, "machine", lambda: mock_arch)
    with pytest.raises(build.BuildError, match="architecture"):
        build.RedisAIBuilder()


@pytest.fixture
def p_test_dir(test_dir):
    yield pathlib.Path(test_dir).resolve()


@for_each_device
def test_rai_builder_raises_if_attempting_to_place_deps_when_build_dir_dne(
    monkeypatch, p_test_dir, device
):
    monkeypatch.setattr(
        build.RedisAIBuilder,
        "rai_build_path",
        property(lambda self: p_test_dir / "path/to/dir/that/dne"),
    )
    rai_builder = build.RedisAIBuilder()
    with pytest.raises(build.BuildError, match=r"build directory not found"):
        rai_builder._fetch_deps_for(device)


@for_each_device
def test_rai_builder_raises_if_attempting_to_place_deps_in_nonempty_dir(
    monkeypatch, p_test_dir, device
):
    (p_test_dir / "some_file.txt").touch()
    monkeypatch.setattr(
        build.RedisAIBuilder, "rai_build_path", property(lambda self: p_test_dir)
    )
    monkeypatch.setattr(
        build.RedisAIBuilder, "get_deps_dir_path_for", lambda *a, **kw: p_test_dir
    )
    rai_builder = build.RedisAIBuilder()

    with pytest.raises(build.BuildError, match=r"is not empty"):
        rai_builder._fetch_deps_for(device)


def _confirm_inst_presence(type_, should_be_present, seq):
    expected_num_occurrences = 1 if should_be_present else 0
    occurrences = filter(lambda item: isinstance(item, type_), seq)
    return expected_num_occurrences == len(tuple(occurrences))


# Helper functions to check for the presence (or absence) of a
# ``_RAIBuildDependency`` dependency in a list of dependencies that need to be
# fetched by a ``RedisAIBuilder`` instance
dlpack_dep_presence = functools.partial(
    _confirm_inst_presence, build._DLPackRepository, True
)
pt_dep_presence = functools.partial(_confirm_inst_presence, build._PTArchive)
tf_dep_presence = functools.partial(_confirm_inst_presence, build._TFArchive)
ort_dep_presence = functools.partial(_confirm_inst_presence, build._ORTArchive)


@for_each_device
@toggle_build_tf
@toggle_build_pt
@toggle_build_ort
def test_rai_builder_will_add_dep_if_backend_requested_wo_duplicates(
    device, build_tf, build_pt, build_ort
):
    rai_builder = build.RedisAIBuilder(
        build_tf=build_tf, build_torch=build_pt, build_onnx=build_ort
    )
    requested_backends = rai_builder._get_deps_to_fetch_for(device)
    assert dlpack_dep_presence(requested_backends)
    assert tf_dep_presence(build_tf, requested_backends)
    assert pt_dep_presence(build_pt, requested_backends)
    assert ort_dep_presence(build_ort, requested_backends)


@for_each_device
@toggle_build_tf
@toggle_build_pt
def test_rai_builder_will_not_add_dep_if_custom_dep_path_provided(
    device, p_test_dir, build_tf, build_pt
):
    mock_ml_lib = p_test_dir / "some/ml/lib"
    mock_ml_lib.mkdir(parents=True)
    rai_builder = build.RedisAIBuilder(
        build_tf=build_tf,
        build_torch=build_pt,
        build_onnx=False,
        libtf_dir=str(mock_ml_lib if build_tf else ""),
        torch_dir=str(mock_ml_lib if build_pt else ""),
    )
    requested_backends = rai_builder._get_deps_to_fetch_for(device)
    assert dlpack_dep_presence(requested_backends)
    assert tf_dep_presence(False, requested_backends)
    assert pt_dep_presence(False, requested_backends)
    assert ort_dep_presence(False, requested_backends)
    assert len(requested_backends) == 1


def test_rai_builder_raises_if_it_fetches_an_unexpected_number_of_ml_deps(
    monkeypatch, p_test_dir
):
    monkeypatch.setattr(
        build.RedisAIBuilder, "rai_build_path", property(lambda self: p_test_dir)
    )
    monkeypatch.setattr(
        build,
        "_place_rai_dep_at",
        lambda target, verbose: lambda dep: target
        / "whoops_all_ml_deps_extract_to_a_dir_with_this_name",
    )
    rai_builder = build.RedisAIBuilder(build_tf=True, build_torch=True, build_onnx=True)
    with pytest.raises(
        build.BuildError,
        match=r"Expected to place \d+ dependencies, but only found \d+",
    ):
        rai_builder._fetch_deps_for("cpu")


def test_threaded_map():
    def _some_io_op(x):
        return x * x

    assert (0, 1, 4, 9, 16) == tuple(build._threaded_map(_some_io_op, range(5)))


def test_threaded_map_returns_early_if_nothing_to_map():
    sleep_duration = 60

    def _some_long_io_op(_):
        time.sleep(sleep_duration)

    start = time.time()
    build._threaded_map(_some_long_io_op, [])
    end = time.time()
    assert end - start < sleep_duration
