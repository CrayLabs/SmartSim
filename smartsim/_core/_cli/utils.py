import subprocess
from pathlib import Path

from smartsim._core._install.buildenv import SetupError
from smartsim._core._install.builder import BuildError
from smartsim._core.utils import colorize
from smartsim.log import get_logger

smart_logger_format = "[%(name)s] %(levelname)s %(message)s"
logger = get_logger("Smart", fmt=smart_logger_format)


def get_install_path():
    try:
        import smartsim as _
    except (ImportError, ModuleNotFoundError):
        raise SetupError("Could not import SmartSim") from None

    # find the path to the setup script
    package_path = Path(_.__path__[0]).resolve()
    if not package_path.is_dir():
        raise SetupError("Could not find SmartSim installation site")

    return package_path


def color_bool(trigger=True):
    _color = "green" if trigger else "red"
    return colorize(str(trigger), color=_color)


def pip_install(packages, end_point=None, verbose=False):
    """Install a pip package to be used in the SmartSim build
    Currently only Torch shared libraries are re-used for the build
    """
    if end_point:
        packages.append(f"-f {end_point}")
    packages = " ".join(packages)

    # form pip install command
    cmd = ["python", "-m", "pip", "install", packages]
    cmd = " ".join(cmd)

    if verbose:
        logger.info(f"Installing packages {packages}...")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    _, err = proc.communicate()
    returncode = int(proc.returncode)
    if returncode != 0:
        error = f"{packages} installation failed with exitcode {returncode}\n"
        error += err.decode("utf-8")
        raise BuildError(error)
    if verbose:
        logger.info(f"{packages} installed successfully")
