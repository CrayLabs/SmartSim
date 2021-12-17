# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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
from os import path

from ..error import SSConfigError


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

    def __init__(self, tagged, copy, symlink):
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
        self.tagged = tagged
        self.copy = copy
        self.link = symlink
        self.tagged_hierarchy = None
        self._check_files()

    def _check_files(self):
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

        for i in range(len(self.copy)):
            self.copy[i] = self._check_path(self.copy[i])
        for i in range(len(self.link)):
            self.link[i] = self._check_path(self.link[i])

    def _type_check_files(self, file_list, file_type):
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
        return file_list

    @staticmethod
    def _check_path(file_path):
        """Given a user provided path-like str, find the actual path to
           the directory or file and create a full path.

        :param file_path: path to a specific file or directory
        :type file_path: str
        :raises SSConfigError: if file or directory does not exist
        :return: full path to file or directory
        :rtype: str
        """
        full_path = path.abspath(file_path)
        if path.isfile(full_path):
            return full_path
        if path.isdir(full_path):
            return full_path
        raise SSConfigError(f"File or Directory {file_path} not found")


class TaggedFilesHierarchy:
    """TaggedFilesHierarchy maintains a list of files and a list of child
    TaggedFilesHierarchies. The TaggedFilesHierarchy class is ment to be easily
    converted into a directory file structure each with each dir containing a
    copy of the specifed (possilby tagged) files specifed at each level of the
    hierarchy via a simply depth first search over the hierarchy.
    """

    def __init__(self, base=""):
        """Initialize a TaggedFilesHierarchy

        :param base: pathlike string specifing the generated directory
                     files are located
        :type base: str, optional
        """
        self.base = base
        self.files = set()
        self.dirs = set()

    @classmethod
    def from_list_paths(cls, path_list, dir_contents_to_base=False):
        """Given a list of absolute paths to files and dirs, create and return
        a TaggedFilesHierarchy instance to representing the file hierarchy

        :param path_list: list of absolute path like strings to tagged files
                          or dirs containing tagged files
        :type path_list: list of str
        :param dir_contents_to_base: When a top level dir is encountered, if
                                     this value is truthy, files in the dir are
                                     put into the base hierarchy level.
                                     Otherwise, a new sub level is created for
                                     the dir
        :type dir_contents_to_base: bool
        :return: A built tagged file hierarchy for the given files
        :rtype: TaggedFilesHierarchy
        """
        th = cls()
        if dir_contents_to_base:
            new_paths = []
            for p in path_list:
                if path.isdir(p):
                    new_paths += [path.join(p, f) for f in os.listdir(p)]
                else:
                    new_paths.append(p)
            path_list = new_paths
        th._add_paths(path_list)
        return th

    def _add_file(self, f):
        """Add a file to the current level in the file hierarchy

        :param f: absoute path to a file to add to the hierarchy
        :type f: str
        """
        self.files.add(f)

    def _add_dir(self, d):
        """Add a dir contianing tagged files by creating a new sub level in the
        tagged file hierarchy. All paths within the directroy are added to the
        the new level sub level tagged file hierarchy

        :param d: absoute path to a dir to add to the hierarchy
        :type d: str
        """
        th = TaggedFilesHierarchy(path.join(self.base, path.basename(d)))
        th._add_paths([path.join(d, f) for f in os.listdir(d)])
        self.dirs.add(th)

    def _add_paths(self, paths):
        """Adds files and dirs from the specified in a list of pah like strings
        to the tagged file hierarchy

        :param paths: list of pathlike strings to files to add to the hierarchy
        :type paths: list of str
        :raises SSConfigError: if link to dir is found
                               (prevent chance of circular hierarchy)
        """
        for p in paths:
            p = path.abspath(p)
            if path.isdir(p):
                if path.islink(p):
                    raise SSConfigError(
                        "Subdirectories of directories containing tagged files"
                        + " cannot be links"
                    )
                self._add_dir(p)
            elif path.isfile(p):
                self._add_file(p)
            else:
                raise SSConfigError(f"File or Directory {p} not found")
