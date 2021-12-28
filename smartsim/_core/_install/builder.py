import os
import sys
import stat
import site
import time
import shutil
import subprocess
import multiprocessing as mp
from pathlib import Path


# NOTE: This will be imported by setup.py and hence no
#       smartsim related items should be imported into
#       this file.

# TODO:
#   - check cmake version and use system if possible to avoid conflicts

class SetupError(Exception):
    pass


class Builder():

    def __init__(self, jobs=None, verbose=False):
        _core_dir = Path(os.path.abspath(__file__)).parent.parent
        self.build_dir = _core_dir.joinpath(".third-party")
        self.bin_path = _core_dir.joinpath("bin/")
        self.lib_path = _core_dir.joinpath("lib/")
        self.out = subprocess.DEVNULL
        self.verbose = verbose
        if self.verbose:
            self.out = None

        # make build directory "SmartSim/smartsim/_core/.third-party"
        if not self.build_dir.is_dir():
            self.build_dir.mkdir()
        if not self.bin_path.is_dir():
            self.bin_path.mkdir()
        if not self.lib_path.is_dir():
            self.lib_path.mkdir()

        # set jobs to 1 if not specified
        self.jobs = jobs
        if not jobs:
            self.jobs = 1

    # implemented in base classes
    @property
    def is_built(self):
        raise NotImplementedError

    def build_from_git(self):
        raise NotImplementedError


    @property
    def make(self):
        make_bin = shutil.which("make")
        if make_bin:
            return make_bin
        raise SetupError("Could not find Make binary")

    @property
    def cmake(self):
        """Find and use pip installed cmake if possible"""
        try:
            import cmake
            return os.path.join(cmake.CMAKE_BIN_DIR, "cmake")
        except ImportError:
            cmake_bin = shutil.which('cmake')
            if cmake_bin:
                return cmake_bin
            else:
                raise SetupError("Could not locate cmake in env or python packages")


    def copy_file(self, src, dst, set_exe=False):
        shutil.copyfile(src, dst)
        if set_exe:
            Path(dst).chmod(stat.S_IXUSR | stat.S_IWUSR | stat.S_IRUSR)

    def copy_dir(self, src, dst, set_exe=False):
        src = Path(src)
        dst = Path(dst)
        dst.mkdir(exist_ok=True)
        # copy directory contents
        for content in src.glob("*"):
            if content.is_dir():
                self.copy_dir(content, dst / content.name, set_exe=set_exe)
            else:
                self.copy_file(content, dst / content.name, set_exe=set_exe)

    # get setup path
    def copy_to_bin(self, files):
        for file in files:
            binary_dest = self.bin_path.joinpath(file.name)
            self.copy_file(file, binary_dest, set_exe=True)

    def copy_to_lib(self, files):
        for file in files:
            lib_dest = self.lib_path.joinpath(file.name)
            self.copy_file(file, lib_dest)

    def cleanup(self):
        if self.build_dir.is_dir():
            shutil.rmtree(str(self.build_dir))


class RedisBuilder(Builder):

    def __init__(self, c_compiler, cpp_compiler, malloc, jobs=None, verbose=False):
        super().__init__(jobs=jobs, verbose=verbose)
        self.c = c_compiler
        self.cpp = cpp_compiler
        self.malloc = malloc

    @property
    def is_built(self):
        server = self.bin_path.joinpath("redis-server").is_file()
        cli = self.bin_path.joinpath("redis-cli").is_file()
        return server and cli

    def build_from_git(self, git_url, branch):
        redis_build_path = Path(self.build_dir, "redis")

        # remove git directory if it exists as it should
        # really never exist as we delete after build
        if redis_build_path.is_dir():
            shutil.rmtree(str(redis_build_path))

        # get the source code
        subprocess.check_call(["git",
                               "clone",
                               git_url,
                               "--branch", branch,
                               "--depth", "1", "redis"],
                                stdout=self.out,
                                stderr=self.out,
                                cwd=self.build_dir)

        print(f"Building Redis {branch} ...")
        cmd = [f"CC={self.c}",
               f"CXX={self.cpp}",
               self.make,
               f"-j {self.jobs}",
               f"MALLOC={self.malloc}"]
        subprocess.check_call(" ".join(cmd),
                                stdout=self.out,
                                stderr=self.out,
                                cwd=str(redis_build_path),
                                shell=True)

        # move redis binaries to smartsim/smartsim/_core/bin
        src_dir = redis_build_path.joinpath("src")
        binaries = ["redis-server", "redis-cli"]
        to_export = [src_dir.joinpath(_bin) for _bin in binaries]
        self.copy_to_bin(to_export)


class RedisAIBuilder(Builder):

    def __init__(self,
                 c_compiler,
                 cpp_compiler,
                 torch_dir=None,
                 torch=True,
                 tf=True,
                 onnx=False,
                 jobs=None,
                 verbose=False):
        super().__init__(jobs=jobs, verbose=verbose)
        self.c = c_compiler
        self.cpp = cpp_compiler
        self.rai_build_path = Path(self.build_dir, "RedisAI")

        # convert to int for RAI build scipt
        self.torch = 1 if torch else 0
        self.tf = 1 if tf else 0
        self.onnx = 1 if onnx else 0
        self.torch_dir = torch_dir


    @property
    def is_built(self):
        server = self.lib_path.joinpath("backends").is_dir()
        cli = self.lib_path.joinpath("redisai.so").is_file()
        return server and cli

    def copy_tf_cmake(self):
        # remove the previous version
        rai_tf_cmake = self.rai_build_path.joinpath(
            "opt/cmake/modules/FindTensorFlow.cmake").resolve()
        rai_tf_cmake.unlink()
        # copy ours in
        self.copy_file(self.bin_path / "modules/FindTensorFlow.cmake",
                       rai_tf_cmake,
                       set_exe=False)

    def build_from_git(self, git_url, branch, device):

        # delete previous build dir (should never be there)
        if self.rai_build_path.is_dir():
            shutil.rmtree(self.rai_build_path)

        print("Downloading ML Backend dependencies...")
        # clone the repo
        clone_cmd = ["GIT_LFS_SKIP_SMUDGE=1",
            "git", "clone",
            "--recursive",
            git_url,
            f"--branch v{branch}",
            "--depth=1",
            "RedisAI"]
        clone_cmd = " ".join(clone_cmd)
        subprocess.check_call(clone_cmd,
                              stdout=self.out,
                              stderr=self.out,
                              cwd=self.build_dir,
                              shell=True)

        # copy FindTensorFlow.cmake to RAI cmake dir
        self.copy_tf_cmake()

        # get RedisAI dependencies
        dep_cmd = [
            f"CC={self.c}",
            f"CXX={self.cpp}",
            f"WITH_PT=0", # torch is always 0 because we never use the torch from RAI
            f"WITH_TF={self.tf}",
            f"WITH_TFLITE=0", # never build with TF lite
            f"WITH_ORT={self.onnx}",
            "bash",
            "get_deps.sh",
            device
        ]
        dep_cmd = " ".join(dep_cmd)
        subprocess.check_call(dep_cmd,
                              cwd=self.rai_build_path,
                              stdout=self.out,
                              stderr=self.out,
                              shell=True)

        print(f"Building ML backends and RedisAI {branch}")
        build_cmd = [
            f"CC={self.c}",
            f"CXX={self.cpp}",
            f"WITH_PT={self.torch}", # but we built it in if the user specified it
            f"WITH_TF={self.tf}",
            f"WITH_TFLITE=0", # never build TF Lite
            f"WITH_ORT={self.onnx}",
            "WITH_UNIT_TESTS=0"
        ]

        if device == "gpu":
            build_cmd.append("GPU=1")
        else:
            build_cmd.append("GPU=0")

        if self.torch_dir:
            build_cmd.append(f"Torch_DIR={str(self.torch_dir)}")

        build_cmd.extend(["make", "-C", "opt", "-j",  f"{self.jobs}" , "build"])
        build_cmd = " ".join(build_cmd)
        subprocess.check_call(build_cmd,
                              cwd=self.rai_build_path,
                              shell=True,
                              stdout=self.out,
                              stderr=self.out,
                              env=os.environ.copy().update({"Torch_DIR":self.torch_dir}))
        time.sleep(2)
        self.install_backends(device)
        if self.torch:
            self.move_torch_libs()
        #self.cleanup()

    def install_backends(self, device):
        self.rai_install_path = self.rai_build_path.joinpath(f"install-{device}").resolve()
        rai_lib = self.rai_install_path / "redisai.so"
        rai_backends = self.rai_install_path / "backends"

        if rai_lib.is_file() and rai_backends.is_dir():
            self.copy_dir(rai_backends,
                          self.lib_path / "backends",
                          set_exe=True)
            self.copy_file(rai_lib,
                           self.lib_path / "redisai.so",
                           set_exe=True)


    def move_torch_libs(self):

        ss_rai_torch_path = self.lib_path / "backends"/ "redisai_torch"
        ss_rai_torch_lib_path = ss_rai_torch_path / "lib"

        # retrieve torch shared libraries and copy to the
        # smartsim/_core/lib/backends/redisai_torch/lib dir
        pip_torch_path = Path(site.getsitepackages()[0]) / "torch"
        pip_torch_lib_path = pip_torch_path / "lib"

        self.copy_dir(
            pip_torch_lib_path,
            ss_rai_torch_lib_path,
            set_exe=True
        )

        # also move the openmp files if on a mac
        if sys.platform == "darwin":
            dylibs = pip_torch_path / ".dylibs"
            self.copy_dir(dylibs,
                          ss_rai_torch_path / ".dylibs",
                          set_exe=True)
