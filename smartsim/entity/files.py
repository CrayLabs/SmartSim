# BSD 2-Clause License
#
# Copyright (c) 2021-2023, Hewlett Packard Enterprise
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
import typing as t

from os import path
from tabulate import tabulate


class EntityFiles:
    """EntityFiles are the files a user wishes to have available to
    models and nodes within SmartSim. Each entity has a method
    `entity.attach_generator_files()` that creates one of these
    objects such that at generation time, each file type will be
    present within the generated model or node directory.

    Tagged files are the configuration files for a model that
    can be searched through and edited by the ModelWriter.

    Copy files are files that a user wants to copy into the
    model or node directory without searching through and
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

        :param tagged: tagged files for model configuration
        :type tagged: list of str
        :param copy: files or directories to copy into model
                     or node directories
        :type copy: list of str
        :param symlink: files to symlink into model or node
                        directories
        :type symlink: list of str
        """
        self.tagged = tagged or []
        self.copy = copy or []
        self.link = symlink or []
        self.tagged_hierarchy = None
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

        self.tagged_hierarchy = TaggedFilesHierarchy.from_list_paths(
            self.tagged, dir_contents_to_base=True
        )

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
        :type file_list: list of str
        :param file_type: name of the file type e.g. "tagged"
        :type file_type: str
        :raises TypeError: if incorrect type is provided by user
        :return: file list provided
        :rtype: list of str
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
        :type file_path: str
        :raises FileNotFoundError: if file or directory does not exist
        :return: full path to file or directory
        :rtype: str
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


class TaggedFilesHierarchy:
    """The TaggedFilesHierarchy represents a directory
    containing potentially tagged files and subdirectories.

    TaggedFilesHierarchy.base is the directory path from
    the the root of the generated file structure

    TaggedFilesHierarchy.files is a collection of paths to
    files that need to be copied to directory that the
    TaggedFilesHierarchy represents

    TaggedFilesHierarchy.dirs is a collection of child
    TaggedFilesHierarchy, each representing a subdirectory
    that needs to generated

    By performing a depth first search over the entire
    hierarchy starting at the root directory structure, the
    tagged file directory structure can be replicated
    """

    def __init__(self, parent: t.Optional[t.Any] = None, subdir_name: str = "") -> None:
        """Initialize a TaggedFilesHierarchy

        :param parent: The parent hierarchy of the new hierarchy,
                       must be None if creating a root hierarchy,
                       must be provided if creating a subhierachy
        :type parent: TaggedFilesHierarchy | None, optional
        :param subdir_name: Name of subdirectory representd by the new hierarchy,
                            must be "" if creating a root hierarchy,
                            must be any valid dir name if subhierarchy,
                            invalid names are ".", ".." or contain path seperators
        :type subdir_name: str, optional
        :raises ValueError: if given a subdir_name without a parent,
                            if given a parent without a subdir_name,
                            or if the subdir_name is invalid
        """
        if parent is None and subdir_name:
            raise ValueError(
                "TaggedFilesHierarchies should not have a subdirectory name without a"
                + " parent"
            )
        if parent is not None and not subdir_name:
            raise ValueError(
                "Child TaggedFilesHierarchies must have a subdirectory name"
            )
        if subdir_name in {".", ".."} or path.sep in subdir_name:
            raise ValueError(
                "Child TaggedFilesHierarchies subdirectory names must not contain"
                + " path seperators or be reserved dirs '.' or '..'"
            )

        if parent:
            parent.dirs.add(self)

        self._base: str = path.join(parent.base, subdir_name) if parent else ""
        self.parent: t.Any = parent
        self.files: t.Set[str] = set()
        self.dirs: t.Set[TaggedFilesHierarchy] = set()

    @property
    def base(self) -> str:
        """Property to ensure that self.base is read-only"""
        return self._base

    @classmethod
    def from_list_paths(
        cls, path_list: t.List[str], dir_contents_to_base: bool = False
    ) -> t.Any:
        """Given a list of absolute paths to files and dirs, create and return
        a TaggedFilesHierarchy instance representing the file hierarchy of
        tagged files. All files in the path list will be placed in the base of
        the file hierarchy.

        :param path_list: list of absolute paths to tagged files or dirs
                          containing tagged files
        :type path_list: list[str]
        :param dir_contents_to_base: When a top level dir is encountered, if
                                     this value is truthy, files in the dir are
                                     put into the base hierarchy level.
                                     Otherwise, a new sub level is created for
                                     the dir
        :type dir_contents_to_base: bool
        :return: A built tagged file hierarchy for the given files
        :rtype: TaggedFilesHierarchy
        """
        tagged_file_hierarchy = cls()
        if dir_contents_to_base:
            new_paths = []
            for tagged_path in path_list:
                if os.path.isdir(tagged_path):
                    new_paths += [
                        os.path.join(tagged_path, file)
                        for file in os.listdir(tagged_path)
                    ]
                else:
                    new_paths.append(tagged_path)
            path_list = new_paths
        tagged_file_hierarchy._add_paths(path_list)
        return tagged_file_hierarchy

    def _add_file(self, file: str) -> None:
        """Add a file to the current level in the file hierarchy

        :param file: absoute path to a file to add to the hierarchy
        :type file: str
        """
        self.files.add(file)

    def _add_dir(self, dir_path: str) -> None:
        """Add a dir contianing tagged files by creating a new sub level in the
        tagged file hierarchy. All paths within the directroy are added to the
        the new level sub level tagged file hierarchy

        :param dir: absoute path to a dir to add to the hierarchy
        :type dir: str
        """
        tagged_file_hierarchy = TaggedFilesHierarchy(self, path.basename(dir_path))
        # pylint: disable-next=protected-access
        tagged_file_hierarchy._add_paths(
            [path.join(dir_path, file) for file in os.listdir(dir_path)]
        )

    def _add_paths(self, paths: t.List[str]) -> None:
        """Takes a list of paths and iterates over it, determining if each
        path is to a file or a dir and then appropriatly adding it to the
        TaggedFilesHierarchy.

        :param paths: list of paths to files or dirs to add to the hierarchy
        :type paths: list[str]
        :raises ValueError: if link to dir is found
        :raises FileNotFoundError: if path does not exist
        """
        for candidate in paths:
            candidate = os.path.abspath(candidate)
            if os.path.isdir(candidate):
                if os.path.islink(candidate):
                    raise ValueError(
                        "Tagged directories and thier subdirectories cannot be links"
                        + " to prevent circular directory structures"
                    )
                self._add_dir(candidate)
            elif os.path.isfile(candidate):
                self._add_file(candidate)
            else:
                raise FileNotFoundError(f"File or Directory {candidate} not found")
