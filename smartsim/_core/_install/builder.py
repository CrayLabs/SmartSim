import os
import re
import shutil
import site
import stat
import subprocess
import sys
from pathlib import Path

# NOTE: This will be imported by setup.py and hence no
#       smartsim related items should be imported into
#       this file.

# TODO:
#   - check cmake version and use system if possible to avoid conflicts


class SetupError(Exception):
    pass


class Builder:
    """Base class for building third-party libraries"""

    url_regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )


    def __init__(self, env, jobs=1, verbose=False):

        # build environment from buildenv
        self.env = env

        # Find _core directory and set up paths
        _core_dir = Path(os.path.abspath(__file__)).parent.parent
        self.build_dir = _core_dir / ".third-party"
        self.bin_path = _core_dir / "bin"
        self.lib_path = _core_dir / "lib"

        # Set wether build process will output to std output
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

        self.jobs = jobs

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

    def is_valid_url(self, url):
        return re.match(self.url_regex, url) is not None

    def cleanup(self):
        if self.build_dir.is_dir():
            shutil.rmtree(str(self.build_dir))


class RedisBuilder(Builder):
    """Class to build Redis from Source

    Supported build methods:
     - from git

    See buildenv.py for buildtime configuration of Redis
    version and url.
    """

    def __init__(self,
                 build_env={},
                 malloc="libc",
                 jobs=None,
                 verbose=False):
        super().__init__(build_env, jobs=jobs, verbose=verbose)
        self.malloc = malloc

    @property
    def is_built(self):
        """Check if Redis is built"""
        server = self.bin_path.joinpath("redis-server").is_file()
        cli = self.bin_path.joinpath("redis-cli").is_file()
        return server and cli

    def build_from_git(self, git_url, branch):
        """Build Redis from git

        :param git_url: url from which to retrieve Redis
        :type git_url: str
        :param branch: branch to checkout
        :type branch: str
        """
        redis_build_path = Path(self.build_dir, "redis")

        # remove git directory if it exists as it should
        # really never exist as we delete after build
        if redis_build_path.is_dir():
            shutil.rmtree(str(redis_build_path))

        # Check Redis URL
        if not self.is_valid_url(git_url):
            raise SetupError(f"Malformed Redis URL: {git_url}")


        # get the source code
        subprocess.check_call(
            ["git", "clone", git_url, "--branch", branch, "--depth", "1", "redis"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.build_dir,
        )

        cmd = [
            self.make,
            f"-j {self.jobs}",
            f"MALLOC={self.malloc}",
        ]
        subprocess.check_call(
            " ".join(cmd),
            stdout=self.out,
            stderr=self.out,
            cwd=str(redis_build_path),
            shell=True,
            env=self.env
        )

        # move redis binaries to smartsim/smartsim/_core/bin
        redis_src_dir = redis_build_path / "src"
        self.copy_file(redis_src_dir / "redis-server",
                       self.bin_path / "redis-server", set_exe=True)
        self.copy_file(redis_src_dir / "redis-cli",
                       self.bin_path / "redis-cli", set_exe=True)


class RedisAIBuilder(Builder):
    """Class to build RedisAI from Source

    Supported build method:
     - from git

    See buildenv.py for buildtime configuration of RedisAI
    version and url.
    """

    def __init__(
        self,
        build_env={},
        torch_dir="",
        build_torch=True,
        build_tf=True,
        build_onnx=False,
        jobs=None,
        verbose=False,
    ):
        super().__init__(build_env, jobs=jobs, verbose=verbose)
        self.rai_build_path = Path(self.build_dir, "RedisAI")

        # convert to int for RAI build script
        self.torch = 1 if build_torch else 0
        self.tf = 1 if build_tf else 0
        self.onnx = 1 if build_onnx else 0
        self.torch_dir = torch_dir

    @property
    def is_built(self):
        server = self.lib_path.joinpath("backends").is_dir()
        cli = self.lib_path.joinpath("redisai.so").is_file()
        return server and cli

    def copy_tf_cmake(self):
        """Copy the FindTensorFlow.cmake file to the build directory
        as the version included in RedisAI is out of date for us.
        """
        # remove the previous version
        rai_tf_cmake = self.rai_build_path.joinpath(
            "opt/cmake/modules/FindTensorFlow.cmake"
        ).resolve()
        rai_tf_cmake.unlink()
        # copy ours in
        self.copy_file(
            self.bin_path / "modules/FindTensorFlow.cmake", rai_tf_cmake, set_exe=False
        )

    def build_from_git(self, git_url, branch, device):
        """Build RedisAI from git

        :param git_url: url from which to retrieve RedisAI
        :type git_url: str
        :param branch: branch to checkout
        :type branch: str
        :param device: cpu or gpu
        :type device: str
        """

        # delete previous build dir (should never be there)
        if self.rai_build_path.is_dir():
            shutil.rmtree(self.rai_build_path)

        # Check RedisAI URL
        if not self.is_valid_url(git_url):
            raise SetupError(f"Malformed RedisAI URL: {git_url}")


        # clone the repo
        clone_cmd = [
            "GIT_LFS_SKIP_SMUDGE=1",
            "git",
            "clone",
            "--recursive",
            git_url,
            f"--branch v{branch}",
            "--depth=1",
            "RedisAI",
        ]
        clone_cmd = " ".join(clone_cmd)
        subprocess.check_call(
            clone_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.build_dir,
            shell=True
        )

        # copy FindTensorFlow.cmake to RAI cmake dir
        self.copy_tf_cmake()

        # get RedisAI dependencies
        dep_cmd = [
            f"WITH_PT=0",  # torch is always 0 because we never use the torch from RAI
            f"WITH_TF={self.tf}",
            f"WITH_TFLITE=0",  # never build with TF lite
            f"WITH_ORT={self.onnx}",
            "bash",
            "get_deps.sh",
            device,
        ]
        dep_cmd = " ".join(dep_cmd)
        subprocess.check_call(
            dep_cmd,
            cwd=self.rai_build_path,
            stdout=subprocess.DEVNULL, # always mute this as it not helpful
            stderr=subprocess.DEVNULL,
            shell=True,
        )

        build_cmd = [
            f"WITH_PT={self.torch}",  # but we built it in if the user specified it
            f"WITH_TF={self.tf}",
            f"WITH_TFLITE=0",  # never build TF Lite
            f"WITH_ORT={self.onnx}",
            "WITH_UNIT_TESTS=0",
        ]

        if device == "gpu":
            build_cmd.append("GPU=1")
        else:
            build_cmd.append("GPU=0")

        if self.torch_dir:
            self.env["Torch_DIR"] = str(self.torch_dir)

        build_cmd.extend([self.make, "-C", "opt", "-j", f"{self.jobs}", "build"])
        subprocess.check_call(
            " ".join(build_cmd),
            cwd=self.rai_build_path,
            shell=True,
            stdout=self.out,
            stderr=self.out,
            env=self.env
        )

        self._install_backends(device)
        if self.torch:
            self._move_torch_libs()
        self.cleanup()

    def _install_backends(self, device):
        """Move backend libraries to smartsim/_core/lib/

        :param device: cpu or cpu
        :type device: str
        """
        self.rai_install_path = self.rai_build_path.joinpath(
            f"install-{device}"
        ).resolve()
        rai_lib = self.rai_install_path / "redisai.so"
        rai_backends = self.rai_install_path / "backends"

        if rai_lib.is_file() and rai_backends.is_dir():
            self.copy_dir(rai_backends, self.lib_path / "backends", set_exe=True)
            self.copy_file(rai_lib, self.lib_path / "redisai.so", set_exe=True)

    def _move_torch_libs(self):
        """Move pip install torch libraries

        Since we use pip installed torch libraries for building
        RedisAI, we need to move them into the LD_runpath of redisai.so
        in the smartsim/_core/lib directory.
        """

        ss_rai_torch_path = self.lib_path / "backends" / "redisai_torch"
        ss_rai_torch_lib_path = ss_rai_torch_path / "lib"

        # retrieve torch shared libraries and copy to the
        # smartsim/_core/lib/backends/redisai_torch/lib dir
        pip_torch_path = Path(site.getsitepackages()[0]) / "torch"
        pip_torch_lib_path = pip_torch_path / "lib"

        self.copy_dir(pip_torch_lib_path, ss_rai_torch_lib_path, set_exe=True)

        # also move the openmp files if on a mac
        if sys.platform == "darwin":
            dylibs = pip_torch_path / ".dylibs"
            self.copy_dir(dylibs, ss_rai_torch_path / ".dylibs", set_exe=True)
