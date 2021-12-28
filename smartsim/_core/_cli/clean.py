import argparse
import shutil
import sys
from pathlib import Path

from smartsim._core._cli.utils import get_install_path


class Clean:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Remove previous ML runtime installation"
        )
        parser.add_argument(
            "--clobber",
            action="store_true",
            default=False,
            help="Remove all SmartSim non-python dependencies as well",
        )
        args = parser.parse_args(sys.argv[2:])

        self._core_path = get_install_path() / "_core"
        self.clean(_all=args.clobber)

    def clean(self, _all=False):
        """Remove pre existing installations of ML runtimes

        :param _all: Remove all non-python dependencies
        :type _all: bool, optional
        """

        build_temp = self._core_path / ".third-party"
        if build_temp.is_dir():
            shutil.rmtree(build_temp, ignore_errors=True)

        lib_path = self._core_path / "lib"
        if lib_path.is_dir():

            # remove RedisAI
            rai_path = lib_path / "redisai.so"
            if rai_path.is_file():
                rai_path.unlink()
                print("Successfully removed existing RedisAI installation")

            backend_path = lib_path / "backends"
            if backend_path.is_dir():
                shutil.rmtree(backend_path, ignore_errors=True)
                print("Successfully removed ML runtimes")

        bin_path = self._core_path / "bin"
        if bin_path.is_dir() and _all:
            files_to_remove = ["redis-server", "redis-cli"]
            removed = False
            for _file in files_to_remove:
                file_path = bin_path.joinpath(_file)

                if file_path.is_file():
                    removed = True
                    file_path.unlink()
            if removed:
                print("Successfully removed SmartSim Redis installation")
