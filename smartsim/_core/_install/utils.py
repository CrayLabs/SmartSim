# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import pathlib
import shutil
import tarfile
import typing as t
import zipfile
from urllib.parse import urlparse
from urllib.request import urlretrieve

import git

from smartsim._core._install.platform import Architecture, OperatingSystem

_PathLike = t.Union[str, pathlib.Path]


class UnsupportedArchive(Exception):
    pass


class PathNotFound(Exception):
    pass


class PackageRetriever:
    """Helper class to handle retreiving git repos and archives

    :raises UnsupportedArchive: Thrown if the archive is not zip, git, tgz, or gz
    :raises PathNotFound: Thrown if the Path does not exist
    """

    @staticmethod
    def _from_local_archive(
        source: _PathLike,
        destination: pathlib.Path,
        **kwargs: t.Any,
    ) -> None:
        """Decompress a local archive

        :param source: Path to the archive on a local system
        :param destination: Where to unpack the archive
        """
        if tarfile.is_tarfile(source):
            tarfile.open(source).extractall(path=destination, **kwargs)
        if zipfile.is_zipfile(source):
            zipfile.ZipFile(source).extractall(path=destination, **kwargs)

    @staticmethod
    def _from_local_directory(
        source: _PathLike,
        destination: pathlib.Path,
        **kwargs: t.Any,
    ) -> None:
        """Copy the contents of a directory

        :param source: source directory
        :param destination: desitnation directory
        """
        shutil.copytree(source, destination, **kwargs)

    @classmethod
    def _from_http(
        cls,
        source: str,
        destination: pathlib.Path,
        **kwargs: t.Any,
    ) -> None:
        """Download and decompress a package

        :param source: URL to a particular package
        :param destination: Where to unpack the archive
        """
        local_file, _ = urlretrieve(source, **kwargs)
        cls._from_local_archive(local_file, destination)
        os.remove(local_file)

    @staticmethod
    def _from_git(
        source: str, destination: pathlib.Path, **clone_kwargs: t.Any
    ) -> None:
        """Clone a repository

        :param source: Path to the remote (URL or local) repository
        :param destination: where to clone the repository
        :param clone_kwargs: various options to send to the clone command
        """
        is_mac = OperatingSystem.autodetect() == OperatingSystem.DARWIN
        is_arm64 = Architecture.autodetect() == Architecture.ARM64
        if is_mac and is_arm64:
            config_options = ["--config core.autocrlf=false", "--config core.eol=lf"]
            allow_unsafe_options = True
        else:
            config_options = None
            allow_unsafe_options = False
        git.Repo.clone_from(
            source,
            destination,
            multi_options=config_options,
            allow_unsafe_options=allow_unsafe_options,
            **clone_kwargs
        )

    @classmethod
    def retrieve(
        cls, source: _PathLike, destination: pathlib.Path, **retrieve_kwargs: t.Any
    ) -> None:
        """Primary method for retrieval

        Automatically choose the correct method based on the extension and/or source
        of the archive. If downloaded, this will also decompress the archive and
        extract

        :param source: URL or path to find the package
        :param destination: where to place the package
        :raises UnsupportedArchive: Unknown archive type
        :raises PathNotFound: Path to archive does not exist
        """
        url_scheme = urlparse(str(source)).scheme
        if str(source).endswith(".git"):
            cls._from_git(str(source), destination, **retrieve_kwargs)
        elif url_scheme == "http":
            cls._from_http(str(source), destination, **retrieve_kwargs)
        elif url_scheme == "https":
            cls._from_http(str(source), destination, **retrieve_kwargs)
        else:  # This is probably a path
            source_path = pathlib.Path(source)
            if source_path.exists():
                if source_path.is_dir():
                    cls._from_local_directory(source, destination, **retrieve_kwargs)
                elif source_path.is_file() and source_path.suffix in (
                    ".gz",
                    ".zip",
                    ".tgz",
                ):
                    cls._from_local_archive(source, destination, **retrieve_kwargs)
                else:
                    raise UnsupportedArchive(
                        f"Source ({source}) is not a supported archive or directory "
                    )
            else:
                raise PathNotFound(f"Package path or file does not exist: {source}")
