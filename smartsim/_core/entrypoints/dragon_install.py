import logging
import pathlib
import shutil
import subprocess
import sys
import tarfile
import typing as t

import requests
from github import Github
from github.GitReleaseAsset import GitReleaseAsset

from smartsim.error.errors import SmartSimCLIActionCancelled

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def check_for_utility(util_name: str) -> str:
    """Check for existence of the provided CLI utility.

    :param util_name: CLI utility to locate
    :returns: Full path to executable if found. Otherwise, empty string"""
    utility = shutil.which(util_name)
    if not utility:
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
        logger.warning("Unable to validate Cray EX platform. Aborting dragon install")
        return False

    locate_msg = "Unable to locate %s. Aborting dragon install"

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
    return "dragon-0.8"


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
    return dragon_pin() in asset_name


def _get_release_assets() -> t.Collection[GitReleaseAsset]:
    """Retrieve a dictionary mapping asset names to asset files from the
    latest Dragon release

    :returns: A dictionary containing latest assets matching the supplied pin"""
    git = Github()

    dragon_repo = git.get_repo("DragonHPC/dragon")

    if dragon_repo is None:
        raise SmartSimCLIActionCancelled("Unable to locate dragon repo")

    # repo.get_latest_release fails if only pre-release results are returned
    all_releases = list(dragon_repo.get_releases())
    all_releases = sorted(all_releases, key=lambda r: r.published_at, reverse=True)

    release = all_releases[0]
    assets = release.assets

    return assets


def filter_assets(assets: t.Collection[GitReleaseAsset]) -> t.Optional[GitReleaseAsset]:
    """Filter the available release assets so that HSTA agents are used
    when run on a Cray EX platform

    :param assets: The collection of dragon release assets to filter
    :returns: An asset meeting platform & version filtering requirements"""
    # Expect cray & non-cray assets that require a filter, e.g.
    # 'dragon-0.8-py3.9.4.1-bafaa887f.tar.gz',
    # 'dragon-0.8-py3.9.4.1-CRAYEX-ac132fe95.tar.gz'
    return next(
        (
            asset
            for asset in assets
            if _version_filter(asset.name)
            and _platform_filter(asset.name)
            and _pin_filter(asset.name)
        ),
        None,
    )


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
    output_path = working_dir / asset.name
    if output_path.exists():
        return output_path

    request = requests.get(asset.browser_download_url, timeout=60)
    status_code = request.status_code

    if status_code != 200:
        raise SmartSimCLIActionCancelled(
            f"Unable to retrieve asset. Request status {status_code}"
        )

    with open(output_path, "wb") as asset_file:
        asset_file.write(request.content)

    logger.debug(f"Retrieving {asset.browser_download_url} to {output_path}")
    return output_path


def expand_archive(archive_path: pathlib.Path) -> pathlib.Path:
    """Expand the archive file from the asset

    :param archive_path: path to a downloaded archive for a release asset"""
    if not archive_path.exists():
        raise ValueError(f"Archive {archive_path} does not exist")

    # create a target dir that excludes the .tar.gz suffixes
    expand_to = str(archive_path.absolute()).replace(".tar.gz", "")

    with tarfile.TarFile.open(archive_path, "r") as archive:
        archive.extractall(expand_to)

    logger.debug(f"Asset expanded into: {expand_to}")
    return pathlib.Path(expand_to)


def install_package(asset_dir: pathlib.Path) -> int:
    """Install the package found in `asset_dir` into the current python environment

    :param asset_dir: path to a decompressed archive contents for a release asset"""
    package_path = next(asset_dir.rglob("*.whl"), None)
    if not package_path:
        logger.error(f"No wheel found for package in {asset_dir}")
        return 1

    cmd = f"python -m pip install --force-reinstall {package_path}"
    logger.info(f"Executing installation: {cmd}")

    process = subprocess.run(cmd.split(), check=False)
    logger.debug(f"Installation completed with return code: {process.returncode}")
    return process.returncode


def cleanup(
    archive_path: t.Optional[pathlib.Path] = None,
    asset_dir: t.Optional[pathlib.Path] = None,
) -> None:
    """Delete the downloaded asset and any files extracted during installation

    :param archive_path: path to a downloaded archive for a release asset
    :param asset_dir: path to a decompressed archive contents for a release asset"""
    if archive_path:
        archive_path.unlink(missing_ok=True)
        logger.debug(f"Deleted archive: {archive_path}")
    if asset_dir:
        shutil.rmtree(asset_dir, ignore_errors=True)
        logger.debug(f"Deleted asset directory: {asset_dir}")


def install_dragon() -> int:
    """Retrieve a dragon runtime appropriate for the current platform
    and install to the current python environment
    :returns: Integer return code, 0 for success, non-zero on failures"""
    filename: t.Optional[pathlib.Path] = None
    asset_dir: t.Optional[pathlib.Path] = None

    try:
        asset_info = retrieve_asset_info()
        filename = retrieve_asset(pathlib.Path().cwd(), asset_info)
        asset_dir = expand_archive(filename)

        return install_package(asset_dir)
    except Exception as ex:
        logger.error("Unable to install dragon runtime", exc_info=ex)
    finally:
        cleanup(filename, asset_dir)

    return 1


if __name__ == "__main__":
    return_code = install_dragon()
    sys.exit(return_code)
