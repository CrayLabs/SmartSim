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


import functools
import pathlib
import textwrap
import time

import pytest

import smartsim._core._install.builder as build
from smartsim._core._install.buildenv import RedisAIVersion

# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a

RAI_VERSIONS = RedisAIVersion("1.2.7")

for_each_device = pytest.mark.parametrize(
    "device", [build.Device.CPU, build.Device.GPU]
)

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
def test_os_enum_raises_on_unsupported(mock_os):
    with pytest.raises(build.BuildError, match="operating system") as err_info:
        build.OperatingSystem.from_str(mock_os)


@pytest.mark.parametrize(
    "mock_arch",
    [
        pytest.param(arch_, id=f"arch='{arch_}'")
        for arch_ in ("i386", "i686", "i86pc", "aarch64", "armv7l", "")
    ],
)
def test_arch_enum_raises_on_unsupported(mock_arch):
    with pytest.raises(build.BuildError, match="architecture"):
        build.Architecture.from_str(mock_arch)


@pytest.fixture
def p_test_dir(test_dir):
    yield pathlib.Path(test_dir).resolve()


@for_each_device
def test_rai_builder_raises_if_attempting_to_place_deps_when_build_dir_dne(
    monkeypatch, p_test_dir, device
):
    monkeypatch.setattr(build.RedisAIBuilder, "_validate_platform", lambda a: None)
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
    monkeypatch.setattr(build.RedisAIBuilder, "_validate_platform", lambda a: None)
    monkeypatch.setattr(
        build.RedisAIBuilder, "rai_build_path", property(lambda self: p_test_dir)
    )
    monkeypatch.setattr(
        build.RedisAIBuilder, "get_deps_dir_path_for", lambda *a, **kw: p_test_dir
    )
    rai_builder = build.RedisAIBuilder()

    with pytest.raises(build.BuildError, match=r"is not empty"):
        rai_builder._fetch_deps_for(device)


invalid_build_arm64 = [
    dict(build_tf=True, build_onnx=True),
    dict(build_tf=False, build_onnx=True),
    dict(build_tf=True, build_onnx=False),
]
invalid_build_ids = [
    ",".join([f"{key}={value}" for key, value in d.items()])
    for d in invalid_build_arm64
]


@pytest.mark.parametrize("build_options", invalid_build_arm64, ids=invalid_build_ids)
def test_rai_builder_raises_if_unsupported_deps_on_arm64(build_options):
    with pytest.raises(build.BuildError, match=r"are not supported on.*ARM64"):
        build.RedisAIBuilder(
            _os=build.OperatingSystem.DARWIN,
            architecture=build.Architecture.ARM64,
            **build_options,
        )


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
    monkeypatch, device, build_tf, build_pt, build_ort
):
    monkeypatch.setattr(build.RedisAIBuilder, "_validate_platform", lambda a: None)

    rai_builder = build.RedisAIBuilder(
        build_tf=build_tf, build_torch=build_pt, build_onnx=build_ort
    )
    requested_backends = rai_builder._get_deps_to_fetch_for(build.Device(device))
    assert dlpack_dep_presence(requested_backends)
    assert tf_dep_presence(build_tf, requested_backends)
    assert pt_dep_presence(build_pt, requested_backends)
    assert ort_dep_presence(build_ort, requested_backends)


@for_each_device
@toggle_build_tf
@toggle_build_pt
def test_rai_builder_will_not_add_dep_if_custom_dep_path_provided(
    monkeypatch, device, p_test_dir, build_tf, build_pt
):
    monkeypatch.setattr(build.RedisAIBuilder, "_validate_platform", lambda a: None)
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
    monkeypatch.setattr(build.RedisAIBuilder, "_validate_platform", lambda a: None)
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
        rai_builder._fetch_deps_for(build.Device.CPU)


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


def test_correct_pt_variant_os():
    # Check that all Linux variants return Linux
    for linux_variant in build.OperatingSystem.LINUX.value:
        os_ = build.OperatingSystem.from_str(linux_variant)
        assert build._choose_pt_variant(os_) == build._PTArchiveLinux

    # Check that ARM64 and X86_64 Mac OSX return the Mac variant
    all_archs = (build.Architecture.ARM64, build.Architecture.X64)
    for arch in all_archs:
        os_ = build.OperatingSystem.DARWIN
        assert build._choose_pt_variant(os_) == build._PTArchiveMacOSX


def test_PTArchiveMacOSX_url():
    arch = build.Architecture.X64
    pt_version = RAI_VERSIONS.torch

    pt_linux_cpu = build._PTArchiveLinux(
        build.Architecture.X64, build.Device.CPU, pt_version, False
    )
    x64_prefix = "https://download.pytorch.org/libtorch/"
    assert x64_prefix in pt_linux_cpu.url

    pt_macosx_cpu = build._PTArchiveMacOSX(
        build.Architecture.ARM64, build.Device.CPU, pt_version, False
    )
    arm64_prefix = "https://github.com/CrayLabs/ml_lib_builder/releases/download/"
    assert arm64_prefix in pt_macosx_cpu.url


def test_PTArchiveMacOSX_gpu_error():
    with pytest.raises(build.BuildError, match="support GPU on Mac OSX"):
        build._PTArchiveMacOSX(
            build.Architecture.ARM64, build.Device.GPU, RAI_VERSIONS.torch, False
        ).url


def test_valid_platforms():
    assert build.RedisAIBuilder(
        _os=build.OperatingSystem.LINUX,
        architecture=build.Architecture.X64,
        build_tf=True,
        build_torch=True,
        build_onnx=True,
    )
    assert build.RedisAIBuilder(
        _os=build.OperatingSystem.DARWIN,
        architecture=build.Architecture.X64,
        build_tf=True,
        build_torch=True,
        build_onnx=False,
    )
    assert build.RedisAIBuilder(
        _os=build.OperatingSystem.DARWIN,
        architecture=build.Architecture.X64,
        build_tf=False,
        build_torch=True,
        build_onnx=False,
    )


@pytest.mark.parametrize(
    "plat,cmd,expected_cmd",
    [
        # Bare Word
        pytest.param(
            build.Platform(build.OperatingSystem.LINUX, build.Architecture.X64),
            ["git", "clone", "my-repo"],
            ["git", "clone", "my-repo"],
            id="git-Linux-X64",
        ),
        pytest.param(
            build.Platform(build.OperatingSystem.LINUX, build.Architecture.ARM64),
            ["git", "clone", "my-repo"],
            ["git", "clone", "my-repo"],
            id="git-Linux-Arm64",
        ),
        pytest.param(
            build.Platform(build.OperatingSystem.DARWIN, build.Architecture.X64),
            ["git", "clone", "my-repo"],
            ["git", "clone", "my-repo"],
            id="git-Darwin-X64",
        ),
        pytest.param(
            build.Platform(build.OperatingSystem.DARWIN, build.Architecture.ARM64),
            ["git", "clone", "my-repo"],
            [
                "git",
                "clone",
                "--config",
                "core.autocrlf=false",
                "--config",
                "core.eol=lf",
                "my-repo",
            ],
            id="git-Darwin-Arm64",
        ),
        # Abs path
        pytest.param(
            build.Platform(build.OperatingSystem.LINUX, build.Architecture.X64),
            ["/path/to/git", "clone", "my-repo"],
            ["/path/to/git", "clone", "my-repo"],
            id="Abs-Linux-X64",
        ),
        pytest.param(
            build.Platform(build.OperatingSystem.LINUX, build.Architecture.ARM64),
            ["/path/to/git", "clone", "my-repo"],
            ["/path/to/git", "clone", "my-repo"],
            id="Abs-Linux-Arm64",
        ),
        pytest.param(
            build.Platform(build.OperatingSystem.DARWIN, build.Architecture.X64),
            ["/path/to/git", "clone", "my-repo"],
            ["/path/to/git", "clone", "my-repo"],
            id="Abs-Darwin-X64",
        ),
        pytest.param(
            build.Platform(build.OperatingSystem.DARWIN, build.Architecture.ARM64),
            ["/path/to/git", "clone", "my-repo"],
            [
                "/path/to/git",
                "clone",
                "--config",
                "core.autocrlf=false",
                "--config",
                "core.eol=lf",
                "my-repo",
            ],
            id="Abs-Darwin-Arm64",
        ),
    ],
)
def test_git_commands_are_configered_correctly_for_platforms(plat, cmd, expected_cmd):
    assert build.config_git_command(plat, cmd) == expected_cmd


def test_modify_source_files(p_test_dir):
    def make_text_blurb(food):
        return textwrap.dedent(f"""\
            My favorite food is {food}
            {food} is an important part of a healthy breakfast
            {food} {food} {food} {food}
            This line should be unchanged!
            --> {food} <--
            """)

    original_word = "SPAM"
    mutated_word = "EGGS"

    source_files = []
    for i in range(3):
        source_file = p_test_dir / f"test_{i}"
        source_file.touch()
        source_file.write_text(make_text_blurb(original_word))
        source_files.append(source_file)
    # Modify a single file
    build._modify_source_files(source_files[0], original_word, mutated_word)
    assert source_files[0].read_text() == make_text_blurb(mutated_word)
    assert source_files[1].read_text() == make_text_blurb(original_word)
    assert source_files[2].read_text() == make_text_blurb(original_word)

    # Modify multiple files
    build._modify_source_files(
        (source_files[1], source_files[2]), original_word, mutated_word
    )
    assert source_files[1].read_text() == make_text_blurb(mutated_word)
    assert source_files[2].read_text() == make_text_blurb(mutated_word)
