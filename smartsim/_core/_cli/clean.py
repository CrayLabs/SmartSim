import argparse
import shutil
import sys

from smartsim._core._cli.utils import get_install_path
from smartsim.log import get_logger

smart_logger_format = "[%(name)s] %(levelname)s %(message)s"
logger = get_logger("Smart", fmt=smart_logger_format)


class Clean:
    def __init__(self, clean_all=False):
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
        clobber = args.clobber or clean_all
        self.clean(_all=clobber)

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
                logger.info("Successfully removed existing RedisAI installation")

            backend_path = lib_path / "backends"
            if backend_path.is_dir():
                shutil.rmtree(backend_path, ignore_errors=True)
                logger.info("Successfully removed ML runtimes")

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
                logger.info("Successfully removed SmartSim Redis installation")
