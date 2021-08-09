import os
import stat
import subprocess
import shutil
from pathlib import Path
import multiprocessing as mp

import cmake
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.install import install
from setuptools.command.build_py import build_py

# get number of processors
NPROC = mp.cpu_count()
class Builder():

    def __init__(self):
        self.setup_path = Path(os.path.abspath(os.path.dirname(__file__)))

    @property
    def make(self):
        make_cmd = shutil.which("make")
        return make_cmd

    @property
    def cmake(self):
        """Find and use pip installed cmake"""
        cmake_cmd = os.path.join(cmake.CMAKE_BIN_DIR, "cmake")
        return cmake_cmd

    @property
    def build_env(self):
        _env = {"CC": "gcc",
               "CXX": "g++"}
        env = os.environ.copy()
        env.update(_env)
        return env

    def copy_to_bin(self, files):
        bin_path = self.setup_path.joinpath("smartsim/bin/")
        for file in files:
            binary_dest = bin_path.joinpath(file.name)
            shutil.copyfile(file, binary_dest)
            binary_dest.chmod(stat.S_IXUSR | stat.S_IWUSR | stat.S_IRUSR)

    def copy_to_lib(self, files):
        lib_path = self.setup_path.joinpath("smartsim/lib/")
        if not lib_path.is_dir():
            lib_path.mkdir()
        for file in files:
            lib_dest = lib_path.joinpath(file.name)
            shutil.copyfile(file, lib_dest)
            # shared library also needs to be executable
            lib_dest.chmod(stat.S_IXUSR | stat.S_IWUSR | stat.S_IRUSR)

class Redis(Builder):

    def build(self, build_dir):
        # get the source code
        subprocess.check_call(["git", "clone", "https://github.com/redis/redis.git",
                               "--branch", "6.0.8" , "--depth", "1", "redis"], cwd=build_dir)
        # build dependencies
        redis_build_path = Path(build_dir, "redis")
        subprocess.check_call([f"{self.make} MALLOC=libc"],
                              cwd=redis_build_path,
                              env=self.build_env,
                              shell=True)

        src_dir = redis_build_path.joinpath("src")
        binaries = ["redis-server", "redis-cli"]
        to_export = [src_dir.joinpath(_bin) for _bin in binaries]
        self.copy_to_bin(to_export)


# Hacky workaround for solving CI build "purelib" issue
# see https://github.com/google/or-tools/issues/616
class InstallPlatlib(install):
    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib

class SmartSimBuild(build_py):

    @staticmethod
    def check_build_environment():
        check_prereq("make")
        check_prereq("gcc")
        check_prereq("g++")

    @staticmethod
    def get_build_dir():
        setup_path = Path(os.path.abspath(os.path.dirname(__file__)))
        build_dir = setup_path.joinpath("smartsim/.third-party")

        # make build directory ".third-party"
        if not build_dir.is_dir():
            build_dir.mkdir()

        return build_dir

    def run(self):
        self.check_build_environment()
        build_dir = self.get_build_dir()

        redis_builder = Redis()
        redis_builder.build(build_dir)

        # remove build directory
        shutil.rmtree(build_dir)

        # run original build_py command
        build_py.run(self)

# check that certain dependencies are installed
# TODO: Check versions for compatible versions
def check_prereq(command):
    try:
        out = subprocess.check_output([command, '--version'])
    except OSError as e:
        raise RuntimeError(
            f"{command} must be installed to build SmartSim") from e

# Tested with wheel v0.29.0
class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name

       We use this because we want to pre-package Redis for certain
       platforms to use.
    """
    def has_ext_modules(_placeholder):
        return True


setup(
# ... in setup.cfg
    packages=["smartsim"],
    package_data={"smartsim": [
        "bin/*"
    ]},
    cmdclass={
        "build_py": SmartSimBuild,
        "install": InstallPlatlib
    },
    scripts=["./smart"],
    zip_safe=False,
    distclass=BinaryDistribution
)