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


import pytest

import pathlib
import platform
import threading
import time

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


# fmt: off
# Hacky nonsense to allow for currying w/ pythons weird lambda binding
def _confirm_dep_count(t):
    _eager_find_all_instances_of = lambda    t, xs: tuple(filter(lambda x: isinstance(x, t), xs))
    _count_instances_of          = lambda    t, xs: len(_eager_find_all_instances_of(t, xs))
    _is_num_instances_of         = lambda n, t, xs: _count_instances_of(t, xs) == n
    _is_one_instance_of          = lambda    t, xs: _is_num_instances_of(1, t,xs)
    _is_no_instance_of           = lambda    t, xs: _is_num_instances_of(0, t,xs)

    def _partial(should_build):
        def _partial_partial(xs):
            return (_is_one_instance_of if should_build else _is_no_instance_of)(t, xs)
        return _partial_partial
    return _partial

_confirm_optional_dep_count      = lambda t: lambda should_build, xs: _confirm_dep_count(t)(should_build)(xs)
confirm_dlpack_dep_count         = _confirm_dep_count(build._DLPackRepository)(True)
confirm_pt_dep_count             = _confirm_optional_dep_count(build._PTArchive)
confirm_tf_dep_count             = _confirm_optional_dep_count(build._TFArchive)
confirm_ort_dep_count            = _confirm_optional_dep_count(build._ORTArchive)
# fmt: on


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
    assert confirm_dlpack_dep_count(requested_backends)
    assert confirm_tf_dep_count(build_tf, requested_backends)
    assert confirm_pt_dep_count(build_pt, requested_backends)
    assert confirm_ort_dep_count(build_ort, requested_backends)


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
    needed_deps = rai_builder._get_deps_to_fetch_for(device)
    assert len(needed_deps) == 1
    (dl_pack,) = needed_deps
    assert isinstance(dl_pack, build._DLPackRepository)


def test_rai_builder_raises_if_it_fetches_an_unepected_number_of_ml_deps(
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
    delay = 0.25

    def _some_long_io_op(_):
        time.sleep(delay)

    start = time.time()
    build._threaded_map(_some_long_io_op, range(40))
    end = time.time()
    assert abs(delay - (end - start)) < 0.1


def test_threaded_map_returns_early_if_nothing_to_map():
    def _some_long_io_op(_):
        time.sleep(10)

    start = time.time()
    build._threaded_map(_some_long_io_op, [])
    end = time.time()
    assert end - start < 0.1
