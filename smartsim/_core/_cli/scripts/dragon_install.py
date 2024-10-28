import os
import pathlib
import re
import shutil
import sys
import typing as t
from urllib.request import Request, urlopen

from github import Github
from github.Auth import Token
from github.GitRelease import GitRelease
from github.GitReleaseAsset import GitReleaseAsset
from github.Repository import Repository

from smartsim._core._cli.utils import pip
from smartsim._core._install.utils import retrieve
from smartsim._core.config import CONFIG
from smartsim._core.utils.helpers import check_platform, is_crayex_platform
from smartsim.error.errors import SmartSimCLIActionCancelled
from smartsim.log import get_logger

logger = get_logger(__name__)

DEFAULT_DRAGON_REPO = "DragonHPC/dragon"
DEFAULT_DRAGON_VERSION = "0.10"
DEFAULT_DRAGON_VERSION_TAG = f"v{DEFAULT_DRAGON_VERSION}"
_GH_TOKEN = "SMARTSIM_DRAGON_TOKEN"


class DragonInstallRequest:
    """Encapsulates a request to install the dragon package"""

    def __init__(
        self,
        working_dir: pathlib.Path,
        repo_name: t.Optional[str] = None,
        version: t.Optional[str] = None,
    ) -> None:
        """Initialize an install request.

        :param working_dir: A path to store temporary files used during installation
        :param repo_name: The name of a repository to install from, e.g. DragonHPC/dragon
        :param version: The version to install, e.g. v0.10
        """

        self.working_dir = working_dir
        """A path to store temporary files used during installation"""

        self.repo_name = repo_name or DEFAULT_DRAGON_REPO
        """The name of a repository to install from, e.g. DragonHPC/dragon"""

        self.pkg_version = version or DEFAULT_DRAGON_VERSION
        """The version to install, e.g. 0.10"""

        self._check()

    def _check(self) -> None:
        """Perform validation of this instance

        :raises ValueError: if any value fails validation"""
        if not self.repo_name or len(self.repo_name.split("/")) != 2:
            raise ValueError(
                f"Invalid dragon repository name. Example: `dragonhpc/dragon`"
            )

        # version must match standard dragon tag & filename format `vX.YZ`
        match = re.match(r"^\d\.\d+$", self.pkg_version)
        if not self.pkg_version or not match:
            raise ValueError("Invalid dragon version. Examples: `0.9, 0.91, 0.10`")

        # attempting to retrieve from a non-default repository requires an auth token
        if self.repo_name.lower() != DEFAULT_DRAGON_REPO.lower() and not self.raw_token:
            raise ValueError(
                f"An access token must be available to access {self.repo_name}. "
                f"Set the `{_GH_TOKEN}` env var to pass your access token."
            )

    @property
    def raw_token(self) -> t.Optional[str]:
        """Returns the raw access token from the environment, if available"""
        return os.environ.get(_GH_TOKEN, None)


def get_auth_token(request: DragonInstallRequest) -> t.Optional[Token]:
    """Create a Github.Auth.Token if an access token can be found
    in the environment

    :param request: details of a request for the installation of the dragon package
    :returns: an auth token if one can be built, otherwise `None`"""
    if gh_token := request.raw_token:
        return Token(gh_token)
    return None


def create_dotenv(dragon_root_dir: pathlib.Path, dragon_version: str) -> None:
    """Create a .env file with required environment variables for the Dragon runtime"""
    dragon_root = str(dragon_root_dir)
    dragon_inc_dir = dragon_root + "/include"
    dragon_lib_dir = dragon_root + "/lib"
    dragon_bin_dir = dragon_root + "/bin"

    dragon_vars = {
        "DRAGON_BASE_DIR": dragon_root,
        "DRAGON_ROOT_DIR": dragon_root,
        "DRAGON_INCLUDE_DIR": dragon_inc_dir,
        "DRAGON_LIB_DIR": dragon_lib_dir,
        "DRAGON_VERSION": dragon_version,
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


def _pin_filter(asset_name: str, dragon_version: str) -> bool:
    """Return true if the supplied value contains a dragon version pin match

    :param asset_name: the asset name to inspect for keywords indicating a dragon version
    :param dragon_version: the dragon version to match
    :returns: True if supplied value is correct for current dragon version"""
    return f"dragon-{dragon_version}" in asset_name


def _get_all_releases(dragon_repo: Repository) -> t.Collection[GitRelease]:
    """Retrieve all available releases for the configured dragon repository

    :param dragon_repo: A GitHub repository object for the dragon package
    :returns: A list of GitRelease"""
    all_releases = [release for release in list(dragon_repo.get_releases())]
    return all_releases


def _get_release_assets(request: DragonInstallRequest) -> t.Collection[GitReleaseAsset]:
    """Retrieve a collection of available assets for all releases that satisfy
    the dragon version pin

    :param request: details of a request for the installation of the dragon package
    :returns: A collection of release assets"""
    auth = get_auth_token(request)
    git = Github(auth=auth)
    dragon_repo = git.get_repo(request.repo_name)

    if dragon_repo is None:
        raise SmartSimCLIActionCancelled("Unable to locate dragon repo")

    all_releases = sorted(
        _get_all_releases(dragon_repo), key=lambda r: r.published_at, reverse=True
    )

    # filter the list of releases to include only the target version
    releases = [
        release
        for release in all_releases
        if request.pkg_version in release.title or release.tag_name
    ]

    releases = sorted(releases, key=lambda r: r.published_at, reverse=True)

    if not releases:
        release_titles = ", ".join(release.title for release in all_releases)
        raise SmartSimCLIActionCancelled(
            f"Unable to find a release for dragon version {request.pkg_version}. "
            f"Available releases: {release_titles}"
        )

    assets: t.List[GitReleaseAsset] = []

    # install the latest release of the target version (including pre-release)
    for release in releases:
        # delay in attaching release assets may leave us with an empty list, retry
        # with the next available release
        if assets := list(release.get_assets()):
            logger.debug(f"Found assets for dragon release {release.title}")
            break
        else:
            logger.debug(f"No assets for dragon release {release.title}. Retrying.")

    if not assets:
        raise SmartSimCLIActionCancelled(
            f"Unable to find assets for dragon release {release.title}"
        )

    return assets


def filter_assets(
    request: DragonInstallRequest, assets: t.Collection[GitReleaseAsset]
) -> t.Optional[GitReleaseAsset]:
    """Filter the available release assets so that HSTA agents are used
    when run on a Cray EX platform

    :param request: details of a request for the installation of the dragon package
    :param assets: The collection of dragon release assets to filter
    :returns: An asset meeting platform & version filtering requirements"""
    # Expect cray & non-cray assets that require a filter, e.g.
    # 'dragon-0.8-py3.9.4.1-bafaa887f.tar.gz',
    # 'dragon-0.8-py3.9.4.1-CRAYEX-ac132fe95.tar.gz'
    all_assets = [asset.name for asset in assets]

    assets = list(
        asset
        for asset in assets
        if _version_filter(asset.name) and _pin_filter(asset.name, request.pkg_version)
    )

    if len(assets) == 0:
        available = "\n\t".join(all_assets)
        logger.warning(
            f"Please specify a dragon version (e.g. {DEFAULT_DRAGON_VERSION}) "
            f"of an asset available in the repository:\n\t{available}"
        )
        return None

    asset: t.Optional[GitReleaseAsset] = None

    # Apply platform filter if we have multiple matches for python/dragon version
    if len(assets) > 0:
        asset = next((asset for asset in assets if _platform_filter(asset.name)), None)

    if not asset:
        asset = assets[0]
        logger.warning(f"Platform-specific package not found. Using {asset.name}")

    return asset


def retrieve_asset_info(request: DragonInstallRequest) -> GitReleaseAsset:
    """Find a release asset that meets all necessary filtering criteria

    :param request: details of a request for the installation of the dragon package
    :returns: A GitHub release asset"""
    assets = _get_release_assets(request)
    asset = filter_assets(request, assets)

    platform_result = check_platform()
    if not platform_result.is_cray:
        logger.warning("Installing Dragon without HSTA support")
        for msg in platform_result.failures:
            logger.warning(msg)

    if asset is None:
        raise SmartSimCLIActionCancelled("No dragon runtime asset available to install")

    logger.debug(f"Retrieved asset metadata: {asset}")
    return asset


def retrieve_asset(
    request: DragonInstallRequest, asset: GitReleaseAsset
) -> pathlib.Path:
    """Retrieve the physical file associated to a given GitHub release asset

    :param request: details of a request for the installation of the dragon package
    :param asset: GitHub release asset to retrieve
    :returns: path to the directory containing the extracted release asset
    :raises SmartSimCLIActionCancelled: if the asset cannot be downloaded or extracted
    """
    download_dir = request.working_dir / str(asset.id)

    # if we've previously downloaded the release and still have
    # wheels laying around, use that cached version instead
    cleanup(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    # grab a copy of the complete asset
    asset_path = download_dir / str(asset.name)

    # use the asset URL instead of the browser_download_url to enable
    # using auth for private repositories
    headers: t.Dict[str, str] = {"Accept": "application/octet-stream"}

    if request.raw_token:
        headers["Authorization"] = f"Bearer {request.raw_token}"

    try:
        # a github asset endpoint causes a redirect. the first request
        # receives a pre-signed URL to the asset to pass on to retrieve
        dl_request = Request(asset.url, headers=headers)
        response = urlopen(dl_request)
        presigned_url = response.url

        logger.debug(f"Retrieved asset {asset.name} metadata from {asset.url}")
    except Exception:
        logger.exception(f"Unable to download {asset.name} from: {asset.url}")
        presigned_url = asset.url

    # extract the asset
    try:
        retrieve(presigned_url, asset_path)

        logger.debug(f"Extracted {asset.name} to {download_dir}")
    except Exception as ex:
        raise SmartSimCLIActionCancelled(
            f"Unable to extract {asset.name} from {download_dir}"
        ) from ex

    return download_dir


def install_package(request: DragonInstallRequest, asset_dir: pathlib.Path) -> int:
    """Install the package found in `asset_dir` into the current python environment

    :param request: details of a request for the installation of the dragon package
    :param asset_dir: path to a decompressed archive contents for a release asset
    :returns: Integer return code, 0 for success, non-zero on failures"""
    found_wheels = list(asset_dir.rglob("*.whl"))
    if not found_wheels:
        logger.error(f"No wheel(s) found for package in {asset_dir}")
        return 1

    create_dotenv(found_wheels[0].parent, request.pkg_version)

    try:
        wheels = list(map(str, found_wheels))
        for wheel_path in wheels:
            logger.info(f"Installing package: {wheel_path}")
            pip("install", wheel_path)
    except Exception:
        logger.error(f"Unable to install from {asset_dir}")
        return 1

    return 0


def cleanup(
    archive_path: t.Optional[pathlib.Path] = None,
) -> None:
    """Delete the downloaded asset and any files extracted during installation

    :param archive_path: path to a downloaded archive for a release asset"""
    if not archive_path:
        return

    if archive_path.exists() and archive_path.is_file():
        archive_path.unlink()
        archive_path = archive_path.parent

    if archive_path.exists() and archive_path.is_dir():
        shutil.rmtree(archive_path, ignore_errors=True)
        logger.debug(f"Deleted temporary files in: {archive_path}")


def install_dragon(request: DragonInstallRequest) -> int:
    """Retrieve a dragon runtime appropriate for the current platform
    and install to the current python environment

    :param request: details of a request for the installation of the dragon package
    :returns: Integer return code, 0 for success, non-zero on failures"""
    if sys.platform == "darwin":
        logger.debug(f"Dragon not supported on platform: {sys.platform}")
        return 1

    asset_dir: t.Optional[pathlib.Path] = None

    try:
        asset_info = retrieve_asset_info(request)
        if asset_info is not None:
            asset_dir = retrieve_asset(request, asset_info)
            return install_package(request, asset_dir)

    except SmartSimCLIActionCancelled as ex:
        logger.warning(*ex.args)
    except Exception as ex:
        logger.error("Unable to install dragon runtime", exc_info=True)

    return 2


def display_post_install_logs() -> None:
    """Display post-installation instructions for the user"""

    examples = {
        "ofi-include": "/opt/cray/include",
        "ofi-build-lib": "/opt/cray/lib64",
        "ofi-runtime-lib": "/opt/cray/lib64",
    }

    config = ":".join(f"{k}={v}" for k, v in examples.items())
    example_msg1 = f"dragon-config -a \\"
    example_msg2 = f'    "{config}"'

    logger.info(
        "************************** Dragon Package Installed *****************************"
    )
    logger.info("To enable Dragon to use HSTA (default: TCP), configure the following:")

    for key in examples:
        logger.info(f"\t{key}")

    logger.info("Example:")
    logger.info(example_msg1)
    logger.info(example_msg2)
    logger.info(
        "*********************************************************************************"
    )


if __name__ == "__main__":
    # path for download and extraction of assets
    extraction_dir = CONFIG.core_path / ".dragon"
    dragon_repo = DEFAULT_DRAGON_REPO
    dragon_version = DEFAULT_DRAGON_VERSION

    request = DragonInstallRequest(
        extraction_dir,
        dragon_repo,
        dragon_version,
    )

    sys.exit(install_dragon(request))
