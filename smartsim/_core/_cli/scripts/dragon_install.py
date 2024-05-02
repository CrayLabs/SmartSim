import os
import pathlib
import subprocess
import sys
import typing as t

from github import Github
from github.GitReleaseAsset import GitReleaseAsset

from smartsim._core._cli.utils import pip
from smartsim._core._install.builder import WebTGZ
from smartsim._core.utils.helpers import expand_exe_path
from smartsim.error.errors import SmartSimCLIActionCancelled
from smartsim.log import get_logger

logger = get_logger(__name__)


def config_dir() -> pathlib.Path:
    """Return a directory path where SmartSim configuration files can be written

    :returns: path to the configuration directory"""
    return pathlib.Path(__file__).parents[2] / "config"


def create_dotenv(dragon_root_dir: pathlib.Path) -> None:
    """Create a .env file with required environment variables for the Dragon runtime"""
    dotenv_path = config_dir() / ".env"
    dragon_vars = {"DRAGON_ROOT_DIR": str(dragon_root_dir)}

    lines = [f"{k}={v}" for k, v in dragon_vars.items()]
    lines.append("\n")

    with dotenv_path.open("w") as dotenv:
        dotenv.writelines(lines)


def check_for_utility(util_name: str) -> str:
    """Check for existence of the provided CLI utility.

    :param util_name: CLI utility to locate
    :returns: Full path to executable if found. Otherwise, empty string"""
    utility = ""

    try:
        utility = expand_exe_path(util_name)
    except FileNotFoundError:
        logger.debug(f"{util_name} not available for Cray EX platform check.")

    return utility or ""


def _execute_platform_cmd(cmd: str) -> t.Tuple[str, int]:
    """Execute the platform check command as a subprocess

    :param cmd: the command to execute
    :returns: True if platform is cray ex, False otherwise"""
    process = subprocess.run(
        cmd.split(),
        capture_output=True,
        check=False,
    )
    return process.stdout.decode("utf-8"), process.returncode


def is_crayex_platform() -> bool:
    """Returns True if the current platform is identified as Cray EX and
    HSTA-aware dragon package can be installed, False otherwise.

    :returns: True if current platform is Cray EX, False otherwise"""

    # ldconfig -p | grep cray | grep pmi.so &&
    # ldconfig -p | grep cray | grep pmi2.so &&
    # fi_info | grep cxi\
    ldconfig = check_for_utility("ldconfig")
    fi_info = check_for_utility("fi_info")
    if not all((ldconfig, fi_info)):
        logger.warning(
            "Unable to validate Cray EX platform. Installing standard version"
        )
        return False

    locate_msg = "Unable to locate %s. Installing standard version"

    ldconfig1 = f"{ldconfig} -p"
    ldc_out1, _ = _execute_platform_cmd(ldconfig1)
    target = "pmi.so"
    candidates = [x for x in ldc_out1.split("\n") if "cray" in x]
    pmi1 = any(x for x in candidates if target in x)
    if not pmi1:
        logger.warning(locate_msg, target)
        return False

    ldconfig2 = f"{ldconfig} -p"
    ldc_out2, _ = _execute_platform_cmd(ldconfig2)
    target = "pmi2.so"
    candidates = [x for x in ldc_out2.split("\n") if "cray" in x]
    pmi2 = any(x for x in candidates if target in x)
    if not pmi2:
        logger.warning(locate_msg, target)
        return False

    fi_info_out, _ = _execute_platform_cmd(fi_info)
    target = "cxi"
    cxi = any(x for x in fi_info_out.split("\n") if target in x)
    if not cxi:
        logger.warning(locate_msg, target)
        return False

    return True


def python_version() -> str:
    """Return a formatted string used to filter release assets
    for the current python version"""
    return f"py{sys.version_info.major}.{sys.version_info.minor}"


def dragon_pin() -> str:
    """Return a string indicating the pinned major/minor version of the dragon
    package to install"""
    return "0.9"


def _platform_filter(asset_name: str) -> bool:
    """Return True if the asset name matches naming standard for current
    platform (Cray or non-Cray). Otherwise, returns False.

    :param asset_name: A value to inspect for keywords indicating a Cray EX asset
    :returns: True if supplied value is correct for current platform"""
    key = "crayex"
    is_cray = key in asset_name.lower()
    if is_crayex_platform():
        return is_cray
    return not is_cray


def _version_filter(asset_name: str) -> bool:
    """Return true if the supplied value contains a python version match

    :param asset_name: A value to inspect for keywords indicating a python version
    :returns: True if supplied value is correct for current python version"""
    return python_version() in asset_name


def _pin_filter(asset_name: str) -> bool:
    """Return true if the supplied value contains a dragon version pin match

    :param asset_name: A value to inspect for keywords indicating a dragon version
    :returns: True if supplied value is correct for current dragon version"""
    return f"dragon-{dragon_pin()}" in asset_name


def _get_release_assets() -> t.Collection[GitReleaseAsset]:
    """Retrieve a collection of available assets for all releases that satisfy
    the dragon version pin

    :returns: A collection of release assets"""
    git = Github()

    dragon_repo = git.get_repo("DragonHPC/dragon")

    if dragon_repo is None:
        raise SmartSimCLIActionCancelled("Unable to locate dragon repo")

    # find any releases matching our pinned version requirement
    tags = [tag for tag in dragon_repo.get_tags() if dragon_pin() in tag.name]
    # repo.get_latest_release fails if only pre-release results are returned
    pin_releases = list(dragon_repo.get_release(tag.name) for tag in tags)
    releases = sorted(pin_releases, key=lambda r: r.published_at, reverse=True)

    # take the most recent release for the given pin
    assets = releases[0].assets

    return assets


def filter_assets(assets: t.Collection[GitReleaseAsset]) -> t.Optional[GitReleaseAsset]:
    """Filter the available release assets so that HSTA agents are used
    when run on a Cray EX platform

    :param assets: The collection of dragon release assets to filter
    :returns: An asset meeting platform & version filtering requirements"""
    # Expect cray & non-cray assets that require a filter, e.g.
    # 'dragon-0.8-py3.9.4.1-bafaa887f.tar.gz',
    # 'dragon-0.8-py3.9.4.1-CRAYEX-ac132fe95.tar.gz'
    asset = next(
        (
            asset
            for asset in assets
            if _version_filter(asset.name)
            and _platform_filter(asset.name)
            and _pin_filter(asset.name)
        ),
        None,
    )
    return asset


def retrieve_asset_info() -> GitReleaseAsset:
    """Find a release asset that meets all necessary filtering criteria

    :param dragon_pin: identify the dragon version to install (e.g. dragon-0.8)
    :returns: A GitHub release asset"""
    assets = _get_release_assets()
    asset = filter_assets(assets)
    if asset is None:
        raise SmartSimCLIActionCancelled("No dragon runtime asset available to install")

    logger.debug(f"Retrieved asset metadata: {asset}")
    return asset


def retrieve_asset(working_dir: pathlib.Path, asset: GitReleaseAsset) -> pathlib.Path:
    """Retrieve the physical file associated to a given GitHub release asset

    :param working_dir: location in file system where assets should be written
    :param asset: GitHub release asset to retrieve
    :returns: path to the downloaded asset"""
    if working_dir.exists() and list(working_dir.rglob("*.whl")):
        return working_dir

    archive = WebTGZ(asset.browser_download_url)
    archive.extract(working_dir)

    logger.debug(f"Retrieved {asset.browser_download_url} to {working_dir}")
    return working_dir


def install_package(asset_dir: pathlib.Path) -> int:
    """Install the package found in `asset_dir` into the current python environment

    :param asset_dir: path to a decompressed archive contents for a release asset"""
    wheels = asset_dir.rglob("*.whl")
    wheel_path = next(wheels, None)
    if not wheel_path:
        logger.error(f"No wheel found for package in {asset_dir}")
        return 1

    create_dotenv(wheel_path.parent)

    while wheel_path is not None:
        logger.info(f"Installing package: {wheel_path.absolute()}")

        try:
            pip("install", "--force-reinstall", str(wheel_path))
            wheel_path = next(wheels, None)
        except Exception:
            logger.error(f"Unable to install from {asset_dir}")
            return 1

    return 0


def cleanup(
    archive_path: t.Optional[pathlib.Path] = None,
) -> None:
    """Delete the downloaded asset and any files extracted during installation

    :param archive_path: path to a downloaded archive for a release asset"""
    if archive_path:
        archive_path.unlink(missing_ok=True)
        logger.debug(f"Deleted archive: {archive_path}")


def install_dragon(extraction_dir: t.Union[str, os.PathLike[str]]) -> int:
    """Retrieve a dragon runtime appropriate for the current platform
    and install to the current python environment
    :param extraction_dir: path for download and extraction of assets
    :returns: Integer return code, 0 for success, non-zero on failures"""
    if sys.platform == "darwin":
        logger.debug(f"Dragon not supported on platform: {sys.platform}")
        return 1

    extraction_dir = pathlib.Path(extraction_dir)
    filename: t.Optional[pathlib.Path] = None
    asset_dir: t.Optional[pathlib.Path] = None

    try:
        asset_info = retrieve_asset_info()
        asset_dir = retrieve_asset(extraction_dir, asset_info)

        return install_package(asset_dir)
    except Exception as ex:
        logger.error("Unable to install dragon runtime", exc_info=ex)
    finally:
        cleanup(filename)

    return 2


if __name__ == "__main__":
    sys.exit(install_dragon(pathlib.Path().cwd()))
