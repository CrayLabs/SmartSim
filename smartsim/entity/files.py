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
import typing as t
from os import path

from tabulate import tabulate


# TODO remove when Ensemble is addressed
class EntityFiles:
    """EntityFiles are the files a user wishes to have available to
    applications and nodes within SmartSim. Each entity has a method
    `entity.attach_generator_files()` that creates one of these
    objects such that at generation time, each file type will be
    present within the generated application or node directory.

    Tagged files are the configuration files for a application that
    can be searched through and edited by the ApplicationWriter.

    Copy files are files that a user wants to copy into the
    application or node directory without searching through and
    editing them for tags.

    Lastly, symlink can be used for big datasets or input
    files that a user just wants to have present in the directory
    without necessary having to copy the entire file.
    """

    def __init__(
        self,
        tagged: t.Optional[t.List[str]] = None,
        copy: t.Optional[t.List[str]] = None,
        symlink: t.Optional[t.List[str]] = None,
    ) -> None:
        """Initialize an EntityFiles instance

        :param tagged: tagged files for application configuration
        :param copy: files or directories to copy into application
                     or node directories
        :param symlink: files to symlink into application or node
                        directories
        """
        self.tagged = tagged or []
        self.copy = copy or []
        self.link = symlink or []
        self._check_files()

    def _check_files(self) -> None:
        """Ensure the files provided by the user are of the correct
           type and actually exist somewhere on the filesystem.

        :raises SSConfigError: If a user provides a directory within
                               the tagged files.
        """

        # type check all files provided by user
        self.tagged = self._type_check_files(self.tagged, "Tagged")
        self.copy = self._type_check_files(self.copy, "Copyable")
        self.link = self._type_check_files(self.link, "Symlink")

        for i, value in enumerate(self.copy):
            self.copy[i] = self._check_path(value)

        for i, value in enumerate(self.link):
            self.link[i] = self._check_path(value)

    @staticmethod
    def _type_check_files(
        file_list: t.Union[t.List[str], None], file_type: str
    ) -> t.List[str]:
        """Check the type of the files provided by the user.

        :param file_list: either tagged, copy, or symlink files
        :param file_type: name of the file type e.g. "tagged"
        :raises TypeError: if incorrect type is provided by user
        :return: file list provided
        """
        if file_list:
            if not isinstance(file_list, list):
                if isinstance(file_list, str):
                    file_list = [file_list]
                else:
                    raise TypeError(
                        f"{file_type} files given were not of type list or str"
                    )
            else:
                if not all(isinstance(f, str) for f in file_list):
                    raise TypeError(f"Not all {file_type} files were of type str")
        return file_list or []

    @staticmethod
    def _check_path(file_path: str) -> str:
        """Given a user provided path-like str, find the actual path to
           the directory or file and create a full path.

        :param file_path: path to a specific file or directory
        :raises FileNotFoundError: if file or directory does not exist
        :return: full path to file or directory
        """
        full_path = path.abspath(file_path)
        if path.isfile(full_path):
            return full_path
        if path.isdir(full_path):
            return full_path
        raise FileNotFoundError(f"File or Directory {file_path} not found")

    def __str__(self) -> str:
        """Return table summarizing attached files."""
        values = []

        if self.copy:
            values.append(["Copy", "\n".join(self.copy)])
        if self.link:
            values.append(["Symlink", "\n".join(self.link)])
        if self.tagged:
            values.append(["Configure", "\n".join(self.tagged)])

        if not values:
            return "No file attached to this entity."

        return tabulate(values, headers=["Strategy", "Files"], tablefmt="grid")
