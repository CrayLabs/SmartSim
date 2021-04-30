import os
import re
import sys
import glob
import sysconfig
import platform
import subprocess
import shutil
import site
from pathlib import Path
import multiprocessing as mp

import cmake
from setuptools import setup, find_packages
from distutils.version import LooseVersion

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
            shutil.copyfile(file, bin_path)

    def copy_to_lib(self, files):
        lib_path = self.setup_path.joinpath("smartsim/lib/")
        for file in files:
            shutil.copyfile(file, lib_path)


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

class RedisIP(Builder):


    def build(self, build_dir):
        subprocess.check_call(["git", "clone", "https://github.com/Spartee/RedisIP.git",
                               "--branch", "0.1.0" , "--depth", "1", "RedisIP"], cwd=build_dir)

        cfg = 'Release'
        build_args = ['--config', cfg]
        build_args += ['--', f'-j{str(NPROC)}']

        cmake_path = Path(os.path.abspath(build_dir), "RedisIP")
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={cmake_path}']
        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']

        # run cmake prep step
        subprocess.check_call([self.cmake] + cmake_args,
                              cwd=cmake_path,
                              env=self.build_env)


        cmake_cmd = [self.cmake, '--build', '.'] + build_args
        subprocess.check_call(cmake_cmd,
                              cwd=cmake_path)

        to_export = cmake_path.joinpath("libredisip.so")
        self.copy_to_lib([to_export])

class SmartSimBuild():

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

        build_dir = self.get_build_dir()

        redis_builder = Redis()
        redis_builder.build(build_dir)

        redisip_builder = RedisIP()
        redisip_builder.build(build_dir)


# check that certain dependencies are installed
# TODO: Check versions for compatible versions
def check_prereq(command):
    try:
        out = subprocess.check_output([command, '--version'])
    except OSError:
        raise RuntimeError(
            f"{command} must be installed to build SmartSim")


if __name__ == "__main__":

    # builder
    build = SmartSimBuild()

    # check tools needed for installation
    build.check_build_environment()

    # build
    build.run()


    setup(
    # ... in setup.cfg
        packages=find_packages(),
        package_data={"smartsim": [
            "/bin/*"
        ]},
        libraries=["/lib/*",],
        zip_safe=False,
    )