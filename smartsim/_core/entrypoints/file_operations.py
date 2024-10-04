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

from __future__ import annotations

import argparse
import base64
import functools
import os
import pathlib
import pickle
import shutil
import typing as t
from typing import Callable

from ...log import get_logger

logger = get_logger(__name__)

"""Run file operations move, remove, symlink, copy, and configure
using command line arguments.
"""


def _abspath(input_path: str) -> pathlib.Path:
    """Helper function to check that paths are absolute"""
    path = pathlib.Path(input_path)
    if not path.is_absolute():
        raise ValueError(f"Path `{path}` must be absolute.")
    return path


def _make_substitution(
    tag_name: str, replacement: str | int | float, tag_delimiter: str
) -> Callable[[str], str]:
    """Helper function to replace tags"""
    return lambda s: s.replace(
        f"{tag_delimiter}{tag_name}{tag_delimiter}", str(replacement)
    )


def _prepare_param_dict(param_dict: str) -> dict[str, t.Any]:
    """Decode and deserialize a base64-encoded parameter dictionary.

    This function takes a base64-encoded string representation of a dictionary,
    decodes it, and then deserializes it using pickle. It performs validation
    to ensure the resulting object is a non-empty dictionary.
    """
    decoded_dict = base64.b64decode(param_dict)
    deserialized_dict = pickle.loads(decoded_dict)
    if not isinstance(deserialized_dict, dict):
        raise TypeError("param dict is not a valid dictionary")
    if not deserialized_dict:
        raise ValueError("param dictionary is empty")
    return deserialized_dict


def _replace_tags_in(
    item: str,
    substitutions: t.Sequence[Callable[[str], str]],
) -> str:
    """Helper function to derive the lines in which to make the substitutions"""
    return functools.reduce(lambda a, fn: fn(a), substitutions, item)


def _process_file(
    substitutions: t.Sequence[Callable[[str], str]],
    source: pathlib.Path,
    destination: pathlib.Path,
) -> None:
    """
    Process a source file by replacing tags with specified substitutions and
    write the result to a destination file.
    """
    # Set the lines to iterate over
    with open(source, "r+", encoding="utf-8") as file_stream:
        lines = [_replace_tags_in(line, substitutions) for line in file_stream]
    # write configured file to destination specified
    with open(destination, "w+", encoding="utf-8") as file_stream:
        file_stream.writelines(lines)


def move(parsed_args: argparse.Namespace) -> None:
    """Move a source file or directory to another location. If dest is an
    existing directory or a symlink to a directory, then the srouce will
    be moved inside that directory. The destination path in that directory
    must not already exist. If dest is an existing file, it will be overwritten.

    Sample usage:
    .. highlight:: bash
    .. code-block:: bash
            python -m smartsim._core.entrypoints.file_operations \
                move /absolute/file/source/path /absolute/file/dest/path

    /absolute/file/source/path: File or directory to be moved
    /absolute/file/dest/path: Path to a file or directory location
    """
    shutil.move(parsed_args.source, parsed_args.dest)


def remove(parsed_args: argparse.Namespace) -> None:
    """Remove a file or directory.

    Sample usage:
    .. highlight:: bash
    .. code-block:: bash
            python -m smartsim._core.entrypoints.file_operations \
                remove /absolute/file/path

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
    .. highlight:: bash
    .. code-block:: bash
            python -m smartsim._core.entrypoints.file_operations copy \
                /absolute/file/source/path /absolute/file/dest/path \
                --dirs_exist_ok

    /absolute/file/source/path: Path to directory, or path to file to
        copy to a new location
    /absolute/file/dest/path: Path to destination directory or path to
        destination file
    --dirs_exist_ok: if the flag is included, the copying operation will 
        continue if the destination directory and files alrady exist, 
        and will be overwritten by corresponding files. If the flag is 
        not includedm and the destination file already exists, a
        FileExistsError will be raised
    """
    if os.path.isdir(parsed_args.source):
        shutil.copytree(
            parsed_args.source,
            parsed_args.dest,
            dirs_exist_ok=parsed_args.dirs_exist_ok,
        )
    else:
        shutil.copy(parsed_args.source, parsed_args.dest)


def symlink(parsed_args: argparse.Namespace) -> None:
    """
    Create a symbolic link pointing to the exisiting source file
    named link.

    Sample usage:
    .. highlight:: bash
    .. code-block:: bash
            python -m smartsim._core.entrypoints.file_operations \
                symlink /absolute/file/source/path /absolute/file/dest/path

    /absolute/file/source/path: the exisiting source path
    /absolute/file/dest/path: target name where the symlink will be created.
    """
    os.symlink(parsed_args.source, parsed_args.dest)


def configure(parsed_args: argparse.Namespace) -> None:
    """Set, search and replace the tagged parameters for the
    configure_file operation within tagged files attached to an entity.

    User-formatted files can be attached using the `configure_file` argument.
    These files will be modified during ``Application`` generation to replace
    tagged sections in the user-formatted files with values from the `params`
    initializer argument used during ``Application`` creation:

    Sample usage:
    .. highlight:: bash
    .. code-block:: bash
            python -m smartsim._core.entrypoints.file_operations \
                configure_file /absolute/file/source/path /absolute/file/dest/path \
                tag_deliminator param_dict

    /absolute/file/source/path: The tagged files the search and replace operations
    to be performed upon
    /absolute/file/dest/path: The destination for configured files to be
    written to.
    tag_delimiter: tag for the configure_file operation to search for, defaults to
        semi-colon e.g. ";"
    param_dict: A dict of parameter names and values set for the file

    """
    tag_delimiter = parsed_args.tag_delimiter
    param_dict = _prepare_param_dict(parsed_args.param_dict)

    substitutions = tuple(
        _make_substitution(k, v, tag_delimiter) for k, v in param_dict.items()
    )
    if parsed_args.source.is_dir():
        for dirpath, _, filenames in os.walk(parsed_args.source):
            new_dir_dest = dirpath.replace(
                str(parsed_args.source), str(parsed_args.dest), 1
            )
            os.makedirs(new_dir_dest, exist_ok=True)
            for file_name in filenames:
                src_file = os.path.join(dirpath, file_name)
                dst_file = os.path.join(new_dir_dest, file_name)
                print(type(substitutions))
                _process_file(substitutions, src_file, dst_file)
    else:
        dst_file = parsed_args.dest / os.path.basename(parsed_args.source)
        _process_file(substitutions, parsed_args.source, dst_file)


def get_parser() -> argparse.ArgumentParser:
    """Instantiate a parser to process command line arguments

    :returns: An argument parser ready to accept required command generator parameters
    """
    arg_parser = argparse.ArgumentParser(description="Command Generator")

    subparsers = arg_parser.add_subparsers(help="file_operations")

    # Subparser for move op
    move_parser = subparsers.add_parser("move")
    move_parser.set_defaults(func=move)
    move_parser.add_argument("source", type=_abspath)
    move_parser.add_argument("dest", type=_abspath)

    # Subparser for remove op
    remove_parser = subparsers.add_parser("remove")
    remove_parser.set_defaults(func=remove)
    remove_parser.add_argument("to_remove", type=_abspath)

    # Subparser for copy op
    copy_parser = subparsers.add_parser("copy")
    copy_parser.set_defaults(func=copy)
    copy_parser.add_argument("source", type=_abspath)
    copy_parser.add_argument("dest", type=_abspath)
    copy_parser.add_argument("--dirs_exist_ok", action="store_true")

    # Subparser for symlink op
    symlink_parser = subparsers.add_parser("symlink")
    symlink_parser.set_defaults(func=symlink)
    symlink_parser.add_argument("source", type=_abspath)
    symlink_parser.add_argument("dest", type=_abspath)

    # Subparser for configure op
    configure_parser = subparsers.add_parser("configure")
    configure_parser.set_defaults(func=configure)
    configure_parser.add_argument("source", type=_abspath)
    configure_parser.add_argument("dest", type=_abspath)
    configure_parser.add_argument("tag_delimiter", type=str, default=";")
    configure_parser.add_argument("param_dict", type=str)

    return arg_parser


def parse_arguments() -> argparse.Namespace:
    """Parse the command line arguments

    :returns: the parsed command line arguments
    """
    parser = get_parser()
    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    args = parse_arguments()
    args.func(args)
