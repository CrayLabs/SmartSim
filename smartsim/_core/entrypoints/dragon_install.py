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
    :type util_name: str
    :returns: Full path to executable if found. Otherwise, empty string
    :rtype: str"""
    utility = shutil.which(util_name)
    if not utility:
        logger.debug(f"{util_name} not available for Cray EX platform check.")
    return utility or ""


def _execute_platform_cmd(cmd: str) -> bool:
    """Execute the platform check command as a subprocess

    :param cmd: the command to execute
    :type cmd: str:
    :returns: True if platform is cray ex, False otherwise
    :rtype: bool"""
    with subprocess.Popen(
        cmd.split(),
        stdout=sys.stdout,
        stderr=sys.stderr,
    ) as popen:

        stdout, stderr = popen.communicate()
        outputs = stdout.decode("utf-8")
        errors = stderr.decode("utf-8")

        if len(outputs) == 0 or len(errors) > 0:
            return False

        return True


def is_crayex_platform() -> bool:
    """Returns True if the current platform is identified as Cray EX and
    HSTA-aware dragon package can be installed, False otherwise.

    :returns: True if current platform is Cray EX, False otherwise
    :rtype: bool"""

    # ldconfig -p | grep cray | grep pmi.so &&
    # ldconfig -p | grep cray | grep pmi2.so &&
    # fi_info | grep cxi\
    ldconfig = check_for_utility("ldconfig")
    fi_info = check_for_utility("fi_info")
    if not all((ldconfig, fi_info)):
        return False

    cray_pmi1 = f"{ldconfig} -p | grep cray | grep pmi.so"
    cray_pmi2 = f"{ldconfig} -p | grep cray | grep pmi2.so"
    cxi = f"{fi_info} | grep cxi"
    cmd = f"{cray_pmi1} && {cray_pmi2} && {cxi}"

    logger.debug(f"Checking for Cray EX platform: {cmd}")
    return _execute_platform_cmd(cmd)


def python_version() -> str:
    """Return a formatted string used to filter release assets
    for the current python version"""
    return f"py{sys.version_info.major}.{sys.version_info.minor}"


def dragon_pin() -> str:
    """Return a string indicating the pinned major/minor version of the dragon
    package to install"""
    return "dragon-0.8"


def _platform_filter(value: str) -> bool:
    """Return True if the supplied value contains a `Cray EX` keyword and the
    current platform is Cray, False otherwise.

    :param value: A value to inspect for keywords indicating a Cray EX asset
    :type value: str
    :returns: True if supplied value is correct for current platform
    :rtype: bool"""
    key = "crayex"
    is_cray = key in value.lower()
    if is_crayex_platform():
        return is_cray
    return not is_cray


def _version_filter(value: str) -> bool:
    """Return true if the supplied value contains a python version match

    :param value: A value to inspect for keywords indicating a python version
    :type value: str
    :returns: True if supplied value is correct for current python version
    :rtype: bool"""
    return python_version() in value


def _pin_filter(value: str) -> bool:
    """Return true if the supplied value contains a dragon version pin match

    :param value: A value to inspect for keywords indicating a dragon version
    :type value: str
    :returns: True if supplied value is correct for current dragon version
    :rtype: bool"""
    return dragon_pin() in value


def _get_release_assets() -> t.Dict[str, GitReleaseAsset]:
    """Retrieve a dictionary mapping asset names to asset files from the
    latest Dragon release

    :returns: A dictionary containing latest assets matching the supplied pin
    :rtype: Dict[str, GitReleaseAsset]"""
    git = Github()

    dragon_repo = git.get_repo("DragonHPC/dragon")

    if dragon_repo is None:
        raise SmartSimCLIActionCancelled("Unable to locate dragon repo")

    # repo.get_latest_release fails if only pre-release results are returned
    all_releases = list(dragon_repo.get_releases())
    pinned_releases = [release for release in all_releases]
    all_releases = sorted(pinned_releases, key=lambda r: r.published_at, reverse=True)

    release = all_releases[0]
    assets = release.assets

    asset_map = {asset.name: asset for asset in assets if dragon_pin in asset.name}
    return asset_map


def filter_assets(assets: t.Dict[str, GitReleaseAsset]) -> t.Optional[GitReleaseAsset]:
    """Filter the available release assets so that HSTA agents are used
    when run on a Cray EX platform

    :param assets: The collection of dragon release assets to filter
    :type assets: t.Dict[str, GitReleaseAsset]
    :returns: An asset meeting platform & version filtering requirements
    :rtype: Optional[GitReleaseAsset]"""
    # We'll have a cray & non-cray assets to filter, e.g.
    # 'dragon-0.8-py3.9.4.1-bafaa887f.tar.gz',
    # 'dragon-0.8-py3.9.4.1-CRAYEX-ac132fe95.tar.gz'

    iterable = iter(
        k
        for k in assets
        if _version_filter(k) and _platform_filter(k) and _pin_filter(k)
    )

    asset_key = next(iterable, None)
    if not asset_key:
        logger.error(f"Unable to find a release asset for {python_version()}")
        return None

    return assets.get(asset_key, None)


def retrieve_asset_info() -> GitReleaseAsset:
    """Find a release asset that meets all necessary filtering criteria

    :param dragon_pin: A string identifying the dragon version to install (e.g. dragon-0.8)
    :type dragon_pin: str
    :returns: A GitHub release asset
    :rtype: GitReleaseAsset"""
    asset_map = _get_release_assets()
    asset = filter_assets(asset_map)
    if asset is None:
        raise SmartSimCLIActionCancelled("No dragon runtime asset available to install")

    logger.debug(f"Retrieved asset metadata: {asset}")
    return asset


def retrieve_asset(working_dir: pathlib.Path, asset: GitReleaseAsset) -> pathlib.Path:
    """Retrieve the physical file associated to a given GitHub release asset

    :param asset: GitHub release asset to retrieve
    :type asset: GitReleaseAsset
    :returns: path to the downloaded asset
    :rtype: pathlib.Path"""
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

    logger.debug("Selected asset: {filename}")
    return output_path


def expand_archive(archive_path: pathlib.Path) -> pathlib.Path:
    """Expand the archive file from the asset

    :param archive_path: path to a downloaded archive for a release asset
    :type archive_path: Optional[pathlib.Path]"""
    if not archive_path.exists():
        raise ValueError(f"Archive {archive_path} does not exist")

    # create a target dir that excludes the .tar.gz suffixes
    expand_to = str(archive_path.absolute()).replace(".tar.gz", "")

    with tarfile.TarFile.open(archive_path, "r") as archive:
        archive.extractall(expand_to)

    logger.debug("Asset expanded into: {expand_to}")
    return pathlib.Path(expand_to)


def install_package(asset_dir: pathlib.Path) -> int:
    """Install the package found in `asset_dir` into the current python environment

    :param asset_dir: path to a decompressed archive contents for a release asset
    :type asset_dir: Optional[pathlib.Path]"""
    package_path = next(asset_dir.rglob("*.whl"), None)
    if not package_path:
        logger.error(f"No wheel found for package in {asset_dir}")
        return 1

    cmd = f"python -m pip install --force-reinstall {package_path}"
    logger.info(f"Executing installation: {cmd}")

    with subprocess.Popen(cmd.split()) as installer:
        result = installer.wait()
        logger.debug(f"Installation completed with return code: {result}")
        return result


def cleanup(
    archive_path: t.Optional[pathlib.Path] = None,
    asset_dir: t.Optional[pathlib.Path] = None,
) -> None:
    """Delete the downloaded asset and any files extracted during installation

    :param archive_path: path to a downloaded archive for a release asset
    :type archive_path: Optional[pathlib.Path]
    :param asset_dir: path to a decompressed archive contents for a release asset
    :type asset_dir: Optional[pathlib.Path]
    """
    if archive_path:
        archive_path.unlink(missing_ok=True)
        logger.debug(f"Deleted archive: {archive_path}")
    if asset_dir:
        shutil.rmtree(asset_dir, ignore_errors=True)
        logger.debug(f"Deleted asset directory: {asset_dir}")


def install_dragon(dragon_pin: str) -> int:
    """Retrieve a dragon runtime appropriate for the current platform
    and install to the current python environment"""
    filename: t.Optional[pathlib.Path] = None
    asset_dir: t.Optional[pathlib.Path] = None

    try:
        asset_info = retrieve_asset_info(pathlib.Path.cwd(), dragon_pin)
        filename = retrieve_asset(asset_info)
        asset_dir = expand_archive(filename)

        return install_package(asset_dir)
    except Exception as ex:
        logger.error("Unable to install dragon runtime", exc_info=ex)
    finally:
        cleanup(filename, asset_dir)

    return 1


if __name__ == "__main__":
    install_dragon(dragon_pin())
