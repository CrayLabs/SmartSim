# BSD 2-Clause License
#
# Copyright (c) 2021-2024 Hewlett Packard Enterprise
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

import argparse
import base64
import collections
import os
import pickle
import re
import shutil
from ast import literal_eval
from distutils import dir_util  # pylint: disable=deprecated-module


def move(parsed_args: argparse.Namespace) -> None:
    """Move a source file or directory to another location. If dest is an
    existing directory or a symlink to a directory, then the srouce will
    be moved inside that directory. The destination path in that directory
    must not already exist. If dest is an existing file, it will be overwritten.

    Sample usage:
        python _core/entrypoints/file_operations.py move
        /absolute/file/source/path /absolute/file/dest/path

        /absolute/file/source/path: File or directory to be moved
        /absolute/file/dest/path: Path to a file or directory location
    """
    shutil.move(parsed_args.source, parsed_args.dest)


def remove(parsed_args: argparse.Namespace) -> None:
    """Remove a file or directory.

    Sample usage:
        python _core/entrypoints/file_operations.py remove
        /absolute/file/path

        /absolute/file/path: Path to the file or directory to be deleted
    """
    if os.path.isdir(parsed_args.to_remove):
        os.rmdir(parsed_args.to_remove)
    else:
        os.remove(parsed_args.to_remove)


def copy(parsed_args: argparse.Namespace) -> None:
    """Copy the contents from the source file into the dest file.
    If source is a directory, copy the entire directory tree source to dest.

    Sample usage:
        python _core/entrypoints/file_operations.py copy
        /absolute/file/source/path /absolute/file/dest/path

        /absolute/file/source/path: Path to directory, or path to file to
            copy to a new location
        /absolute/file/dest/path: Path to destination directory or path to
            destination file
    """
    if os.path.isdir(parsed_args.source):
        dir_util.copy_tree(parsed_args.source, parsed_args.dest)
    else:
        shutil.copyfile(parsed_args.source, parsed_args.dest)


def symlink(parsed_args: argparse.Namespace) -> None:
    """
    Create a symbolic link pointing to the exisiting source file
    named link.

    Sample usage:
        python _core/entrypoints/file_operations.py symlink
        /absolute/file/source/path /absolute/file/dest/path

        /absolute/file/source/path: the exisiting source path
        /absolute/file/dest/path: target name where the symlink will be created.
    """
    os.symlink(parsed_args.source, parsed_args.dest)


def configure(parsed_args: argparse.Namespace) -> None:
    """Write a python script to set, search and replace the tagged parameters
    for the configure operation within tagged files attached to an entity.

    User-formatted files can be attached using the `configure` argument.
    These files will be modified during ``Application`` generation to replace
    tagged sections in the user-formatted files with values from the `params`
    initializer argument used during ``Application`` creation:

    Sample usage:
        python _core/entrypoints/file_operations.py configure
        /absolute/file/source/pat /absolute/file/dest/path tag_deliminator param_dict

        /absolute/file/source/path: The tagged files the search and replace operations
        to be performed upon
        /absolute/file/dest/path: Optional destination for configured files to be
        written to.
        tag_delimiter: tag for the configure operation to search for, defaults to
            semi-colon e.g. ";"
        param_dict: A dict of parameter names and values set for the file

    """
    tag_delimiter = ";"
    if parsed_args.tag_delimiter:
        tag_delimiter = parsed_args.tag_delimiter

    decoded_dict = base64.b64decode(literal_eval(parsed_args.param_dict))
    param_dict = pickle.loads(decoded_dict)

    if not param_dict:
        raise ValueError("param dictionary is empty")
    if not isinstance(param_dict, dict):
        raise TypeError("param dict is not a valid dictionary")

    def _get_prev_value(tagged_line: str) -> str:
        split_tag = tagged_line.split(tag_delimiter)
        return split_tag[1]

    def _is_ensemble_spec(tagged_line: str, application_params: dict[str, str]) -> bool:
        split_tag = tagged_line.split(tag_delimiter)
        prev_val = split_tag[1]
        if prev_val in application_params.keys():
            return True
        return False

    edited = []
    used_params = {}

    # Set the tag for the application writer to search for within
    # tagged files attached to an entity.
    regex = "".join(("(", tag_delimiter, ".+", tag_delimiter, ")"))

    # Set the lines to iterate over
    with open(parsed_args.source, "r+", encoding="utf-8") as file_stream:
        lines = file_stream.readlines()

    unused_tags = collections.defaultdict(list)

    # Replace the tagged parameters within the file attached to this
    # application. The tag defaults to ";"
    for i, line in enumerate(lines, 1):
        while search := re.search(regex, line):
            tagged_line = search.group(0)
            previous_value = _get_prev_value(tagged_line)
            if _is_ensemble_spec(tagged_line, param_dict):
                new_val = str(param_dict[previous_value])
                line = re.sub(regex, new_val, line, 1)
                used_params[previous_value] = new_val

            # if a tag_delimiter is found but is not in this application's
            # configurations put in placeholder value
            else:
                tag_delimiter_ = tagged_line.split(tag_delimiter)[1]
                unused_tags[tag_delimiter_].append(i)
                line = re.sub(regex, previous_value, line)
                break
        edited.append(line)

    lines = edited

    # write configured file to destination specified. Default is an overwrite
    if parsed_args.dest:
        file_stream = parsed_args.dest

    with open(parsed_args.source, "w+", encoding="utf-8") as file_stream:
        for line in lines:
            file_stream.write(line)


def get_parser() -> argparse.ArgumentParser:
    """Instantiate a parser to process command line arguments

    :returns: An argument parser ready to accept required command generator parameters
    """
    arg_parser = argparse.ArgumentParser(description="Command Generator")

    subparsers = arg_parser.add_subparsers(help="file_operations")

    # Subparser for move op
    move_parser = subparsers.add_parser("move")
    move_parser.set_defaults(func=move)
    move_parser.add_argument("source")
    move_parser.add_argument("dest")

    # Subparser for remove op
    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("to_remove", type=str)

    # Subparser for copy op
    copy_parser = subparsers.add_parser("copy")
    copy_parser.set_defaults(func=copy)
    copy_parser.add_argument("source", type=str)
    copy_parser.add_argument("dest", type=str)

    # Subparser for symlink op
    symlink_parser = subparsers.add_parser("symlink")
    symlink_parser.set_defaults(func=symlink)
    symlink_parser.add_argument("source", type=str)
    symlink_parser.add_argument("dest", type=str)

    # Subparser for configure op
    configure_parser = subparsers.add_parser("configure")
    configure_parser.set_defaults(func=configure)
    configure_parser.add_argument("source", type=str)
    configure_parser.add_argument("dest", type=str)
    configure_parser.add_argument("tag_delimiter", type=str)
    configure_parser.add_argument("param_dict", type=str)

    return arg_parser


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments

    :returns: the parsed command line arguments
    """
    parser = get_parser()
    parsed_args = parser.parse_args()
    parsed_args.func(parsed_args)
    return parsed_args


if __name__ == "__main__":
    """Run file operations move, remove, symlink, copy, and configure
    using command line arguments.
    """
    os.environ["PYTHONUNBUFFERED"] = "1"

    args = parse_arguments()
