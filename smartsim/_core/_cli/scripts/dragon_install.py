import os
import pathlib
import sys
import typing as t

from github import Github
from github.GitReleaseAsset import GitReleaseAsset

from smartsim._core._cli.utils import pip
from smartsim._core._install.builder import WebTGZ
from smartsim._core.config import CONFIG
from smartsim._core.utils.helpers import check_platform, is_crayex_platform
from smartsim.error.errors import SmartSimCLIActionCancelled
from smartsim.log import get_logger

logger = get_logger(__name__)


def create_dotenv(dragon_root_dir: pathlib.Path) -> None:
    """Create a .env file with required environment variables for the Dragon runtime"""
    dragon_root = str(dragon_root_dir)
    dragon_inc_dir = str(dragon_root_dir / "include")
    dragon_lib_dir = str(dragon_root_dir / "lib")
    dragon_bin_dir = str(dragon_root_dir / "bin")

    dragon_vars = {
        "DRAGON_BASE_DIR": dragon_root,
        "DRAGON_ROOT_DIR": dragon_root,  # note: same as base_dir
        "DRAGON_INCLUDE_DIR": dragon_inc_dir,
        "DRAGON_LIB_DIR": dragon_lib_dir,
        "DRAGON_VERSION": dragon_pin(),
        "PATH": dragon_bin_dir,
        "LD_LIBRARY_PATH": dragon_lib_dir,
    }

    lines = [f"{k}={v}\n" for k, v in dragon_vars.items()]

    if not CONFIG.dragon_dotenv.parent.exists():
        CONFIG.dragon_dotenv.parent.mkdir(parents=True)

    with CONFIG.dragon_dotenv.open("w", encoding="utf-8") as dotenv:
        dotenv.writelines(lines)


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

    platform_result = check_platform()
    if not platform_result.is_cray:
        logger.warning("Installing Dragon without HSTA support")
        for msg in platform_result.failures:
            logger.warning(msg)

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
    sys.exit(install_dragon(CONFIG.core_path / ".dragon"))
