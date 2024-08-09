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
from urllib.request import urlretrieve
from urllib.parse import urlparse
import zipfile

import git

from smartsim._core._install.platform import OperatingSystem, Architecture

_PathLike = t.Union[str, pathlib.Path]

class UnsupportedArchive(Exception):
    pass
class PathNotFound(Exception):
    pass

class PackageRetriever():
    @staticmethod
    def _from_local_archive(
        source: _PathLike,
        destination: pathlib.Path,
        **kwargs: t.Any,
    ) -> None:
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
        shutil.copytree(source, destination, **kwargs)

    @classmethod
    def _from_http(
        cls,
        source: _PathLike,
        destination: pathlib.Path,
        **kwargs: t.Any,
    ) -> None:
        local_file, _ = urlretrieve(source, **kwargs)
        cls._from_local_archive(local_file, destination)
        os.remove(local_file)

    @staticmethod
    def _from_git(source, destination, **clone_kwargs) -> None:
        is_mac = OperatingSystem.autodetect() == OperatingSystem.DARWIN
        is_arm64 = Architecture.autodetect() == Architecture.ARM64
        if is_mac and is_arm64:
            config_options = (
                "--config core.autocrlf=false",
                "--config core.eol=lf"
            )
        else:
            config_options = None
        git.Repo.clone_from(
            source, destination, multi_options=config_options, **clone_kwargs
        )

    @classmethod
    def retrieve(
        cls,
        source:_PathLike,
        destination: pathlib.Path,
        **retrieve_kwargs: t.Any
    ) -> None:
        url_scheme = urlparse(str(source)).scheme
        if str(source).endswith(".git"):
            return cls._from_git(source, destination, **retrieve_kwargs)
        elif url_scheme == "http":
            return cls._from_http(source, destination, **retrieve_kwargs)
        elif url_scheme == "https":
            return cls._from_http(source, destination, **retrieve_kwargs)
        else:  # This is probably a path
            source_path = pathlib.Path(source)
            if source_path.exists():
                if source_path.is_dir():
                    return cls._from_local_directory(source, destination, **retrieve_kwargs)
                elif source_path.is_file() and source_path.suffix in (".gz", ".zip", ".tgz"):
                    return cls._from_local_archive(source, destination, **retrieve_kwargs)
                else:
                    raise UnsupportedArchive(
                        f"Source ({source}) is not a supported archive or directory "
                    )
            else:
                raise PathNotFound(
                    f"Package path or file does not exist: {source}"
                )