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


def is_crayex_platform() -> bool:
    """Returns True if the current platform is identified as Cray EX and
    HSTA-aware dragon package can be installed, False otherwise."""

    # ldconfig -p | grep cray | grep pmi.so &&
    # ldconfig -p | grep cray | grep pmi2.so &&
    # fi_info | grep cxi
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

    ldconfig = check_for_utility("ldconfig")
    fi_info = check_for_utility("fi_info")
    if not all((ldconfig, fi_info)):
        return False

    cray_pmi1 = f"{ldconfig} -p | grep cray | grep pmi.so"
    cray_pmi2 = f"{ldconfig} -p | grep cray | grep pmi2.so"
    cxi = f"{fi_info} | grep cxi"
    cmd = f"{cray_pmi1} && {cray_pmi2} && {cxi}"

    logger.debug(f"Checking for Cray EX platform: {cmd}")
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


def pyv() -> str:
    """Return a formatted string used to filter release assets
    for the current python version"""
    return f"py{sys.version_info.major}.{sys.version_info.minor}"


def retrieve_asset_info() -> GitReleaseAsset:
    def is_version_match(key: str) -> bool:
        "Return true if the supplied value contains a python version match"
        return pyv() in key

    def check_cray(key: str, match_cray: bool) -> bool:
        "Return true if the supplied value contains a `Cray` keyword"
        if match_cray:
            return "cray" in key.lower()

        return "cray" not in key.lower()

    def get_latest_assets() -> t.Dict[str, GitReleaseAsset]:
        """Retrieve a dictionary mapping asset names to asset files from the
        latest Dragon release"""
        git = Github()

        dragon_repo = git.get_repo("DragonHPC/dragon")

        if dragon_repo is None:
            raise SmartSimCLIActionCancelled("Unable to locate dragon repo")

        # repo.get_latest_release fails if only pre-release results are returned
        all_releases = list(dragon_repo.get_releases())
        all_releases = sorted(all_releases, key=lambda r: r.published_at, reverse=True)

        release = all_releases[0]
        assets = release.assets

        asset_map = {asset.name: asset for asset in assets}
        return asset_map

    def filter_assets(
        assets: t.Dict[str, GitReleaseAsset]
    ) -> t.Optional[GitReleaseAsset]:
        """Filter the available release assets so that HSTA agents are used
        when run on a Cray EX platform"""
        # We'll have a cray & non-cray assets to filter, e.g.
        # 'dragon-0.8-py3.9.4.1-bafaa887f.tar.gz',
        # 'dragon-0.8-py3.9.4.1-CRAYEX-ac132fe95.tar.gz'

        # add a filter looking for CRAYEX asset on the Cray EX platform
        match_cray = False
        if is_crayex_platform():
            match_cray = True
            logger.debug("Installer targeting Dragon for Cray EX platform")

        def _filter(name: str, match_cray: bool) -> bool:
            return is_version_match(name) and check_cray(name, match_cray)

        iterable = iter(k for k in asset_map if _filter(k, match_cray))

        asset_key = next(iterable, None)
        if not asset_key:
            logger.error(f"Unable to find a release asset for {pyv()}")
            return None

        return assets.get(asset_key, None)

    asset_map = get_latest_assets()
    asset = filter_assets(asset_map)
    if asset is None:
        raise SmartSimCLIActionCancelled("No dragon runtime asset available to install")

    logger.debug(f"Retrieved asset metadata: {asset}")
    return asset


def retrieve_asset(asset: GitReleaseAsset) -> pathlib.Path:
    """Retrieve the physical file associated to a given GitHub release asset"""
    output_path = pathlib.Path().cwd() / asset.name
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
    """Expand the archive file from the asset."""
    # create a target dir that excludes the .tar.gz suffixes
    expand_to = str(archive_path.absolute()).replace(".tar.gz", "")

    with tarfile.TarFile.open(archive_path, "r") as archive:
        archive.extractall(expand_to)

    logger.debug("Asset expanded into: {expand_to}")
    return pathlib.Path(expand_to)


def install_package(asset_dir: pathlib.Path) -> int:
    package_path = next(asset_dir.rglob("*.whl"), None)
    if not package_path:
        logger.error(f"No wheel found for package in {asset_dir}")

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
    """Delete the downloaded asset and any files extracted during installation"""
    if archive_path:
        archive_path.unlink(missing_ok=True)
        logger.debug(f"Deleted archive: {archive_path}")
    if asset_dir:
        shutil.rmtree(asset_dir, ignore_errors=True)
        logger.debug(f"Deleted asset directory: {asset_dir}")


def install_dragon() -> int:
    """Retrieve a dragon runtime appropriate for the current platform
    and install to the current python environment"""
    filename: t.Optional[pathlib.Path] = None
    asset_dir: t.Optional[pathlib.Path] = None

    try:
        asset_info = retrieve_asset_info()
        filename = retrieve_asset(asset_info)
        asset_dir = expand_archive(filename)

        install_package(asset_dir)
        return 0
    except Exception as ex:
        logger.error("Unable to install dragon runtime", exc_info=ex)
    finally:
        cleanup(filename, asset_dir)

    return 1


if __name__ == "__main__":
    install_dragon()
