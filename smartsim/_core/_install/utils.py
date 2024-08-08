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

import pathlib
import shutil
import tarfile
import typing as t
from urllib.request import urlretrieve
import urllib.parse as urlparse
import zipfile

import git

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
        **kwargs: t.Any
    ) -> None:
        if tarfile.is_tarfile(source):
            tarfile.open(source).extractall(path=destination, **kwargs)
        if zipfile.is_zipfile(source):
            zipfile.open(source).extractall(path=destination, **kwargs)

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
        **kwargs: t.Any
    ) -> None:
        urlretrieve(source, destination, **kwargs)
        cls.from_local_archive(destination, destination.parent)
        destination.unlink()

    @staticmethod
    def _from_git(source, destination, **clone_kwargs) -> None:
        git.Repo(source).clone_from(source, destination, **clone_kwargs)

    @classmethod
    def retrieve(
        cls,
        source:_PathLike,
        destination: pathlib.Path,
        retrieve_kwargs: t.Any
    ) -> None:
        url_scheme = urlparse(source).scheme
        if source.endswith(".git"):
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
                elif source_path.is_file():
                    return cls._from_local_archive(source, destination, **retrieve_kwargs)
                else:
                    raise UnsupportedArchive(
                        f"Source ({source}) is not a supported archive or directory "
                    )
            else:
                raise PathNotFound(
                    f"Package path or file does not exist: {source}"
                )