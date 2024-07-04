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
import os
import typing as t
import shutil
from distutils import dir_util
import re
import collections


def get_dst_path(application_path, input_file):
    return os.path.join(application_path, os.path.basename(input_file))


def _check_path(file_path: str) -> str:
    """Given a user provided path-like str, find the actual path to
        the directory or file and create a full path.

    :param file_path: path to a specific file or directory
    :raises FileNotFoundError: if file or directory does not exist
    :return: full path to file or directory
    """
    full_path = os.path.abspath(file_path)
    if os.path.isfile(full_path):
        return full_path
    if os.path.isdir(full_path):
        return full_path
    raise FileNotFoundError(f"File or Directory {file_path} not found")



def move(source: str, dest: str):
    """Move a file

    :param to_move: Path to a source file to be copied
    :param dst_path: Path to a file to copy the contents from the source file into
    :return: string of python code to perform operation that can be executed
    """

    _check_path(source)
  
    shutil.move(source, dest)


def remove(to_delete: str):
    """Write a python script that removes a file when executed.

    :param to_delete: Path to the file to be deleted
    :return: string of python code to perform operation that can be executed
    """
    _check_path(to_delete)
    
    os.remove(to_delete)


# TODO: remove the entity path things and keep it as a helper funtion that 
# gives the user to add the application, and then it would take the applciation path and do the copy 


def copy(args):
    # args.source_path: str, dest_path: str):
    """
    Write a python script to copy the entity files and directories attached
    to this entity into an entity directory

    :param source_path: Path to directory, or path to file to copy into an entity directory
    :param dest_path: Path to destination directory or path to destination file to copy 
    :return: string of python code to perform operation that can be executed
    """

    print("MADE IT INTO THE COPY FUNCTION")
    _check_path(args.source_path)
    _check_path(args.dest_path)

    if os.path.isdir(args.source_path):
        dir_util.copy_tree(args.source_path, args.dest_path)
    else:
        shutil.copyfile(args.source_path, args.dest_path)



def symlink(source: str, link: str):
    """
    Create a symbolic link pointing to the exisiting source file
    named link

    :param src_file: the exisiting source path
    :param dst_file: target name where the symlink will be created. 
    """

    _check_path(source)

    os.symlink(source, link)



def configure(
    to_configure: str, dest_path: t.Optional[str], param_dict: t.Dict[str, str], tag_delimiter: str = ';'
):
    """Write a python script to set, search and replace the tagged parameters for the configure operation
    within tagged files attached to an entity.

    User-formatted files can be attached using the `to_configure` argument. These files will be modified
during ``Model`` generation to replace tagged sections in the user-formatted files with
values from the `params` initializer argument used during ``Model`` creation:

    :param to_configure: The tagged files the search and replace operations to be performed upon
    :param dest: Optional destination for configured files to be written to
    :param param_dict: A dict of parameter names and values set for the file
    :tag_delimiter: tag for the configure operation to search for, defaults to semi-colon e.g. ";"
    :return: string of python code to perform operation that can be executed
    """

    _check_path(to_configure)
    if dest_path:
        _check_path(dest_path)

    def _get_prev_value(tagged_line: str) -> str:
        split_tag = tagged_line.split(tag_delimiter)
        return split_tag[1]

    def _is_ensemble_spec(
        tagged_line: str, model_params: dict
    ) -> bool:
        split_tag = tagged_line.split(tag_delimiter)
        prev_val = split_tag[1]
        if prev_val in model_params.keys():
            return True
        return False

    edited = []
    used_params = {}
    #param_dict = {param_dict}

   # tag_delimiter = tag_delimiter

    # Set the tag for the modelwriter to search for within
    # tagged files attached to an entity.
    regex = "".join(("(", tag_delimiter, ".+", tag_delimiter, ")"))

    # Set the lines to iterate over
    with open(to_configure,'r+', encoding='utf-8') as file_stream:
        lines = file_stream.readlines()

    unused_tags = collections.defaultdict(list)

    # Replace the tagged parameters within the file attached to this
    # model. The tag defaults to ";"
    for i, line in enumerate(lines, 1):
        while search := re.search(regex, line):
            tagged_line = search.group(0)
            previous_value = _get_prev_value(tagged_line)
            if _is_ensemble_spec(tagged_line, param_dict):
                new_val = str(param_dict[previous_value])
                line = re.sub(regex, new_val, line, 1)
                used_params[previous_value] = new_val

            # if a tag_delimiter is found but is not in this model's configurations
            # put in placeholder value
            else:
                tag_delimiter_ = tagged_line.split(tag_delimiter)[1]
                unused_tags[tag_delimiter_].append(i)
                line = re.sub(regex, previous_value, line)
                break
        edited.append(line)

    lines = edited

    # write configured file to destination specified. Default is an overwrite
    if dest_path:
        file_stream = dest_path

    with open(to_configure, "w+", encoding="utf-8") as file_stream:
        for line in lines:
            file_stream.write(line)


# python _core/entrypoints/file_operations.py remove /absolute/file/path   # must be absolute path
# python _core/entrypoints/file_operations.py move /absolute/file/src/path /absolute/file/dest/path 
# python _core/entrypoints/file_operations.py symlink /absolute/file/src/path /absolute/file/dest/path
# python _core/entrypoints/file_operations.py copy /absolute/file/src/path /absolute/file/dest/path
# python _core/entrypoints/file_operations.py configure /absolte/file/src/path /absolte/file/dest/path tagged_deliminator params  # base64 encoded dictionary for params


import json
def get_parser() -> argparse.ArgumentParser:
    """Instantiate a parser to process command line arguments

    :returns: An argument parser ready to accept required command generator parameters
    """
    arg_parser = argparse.ArgumentParser(description="Command Generator")
    
    subparsers = arg_parser.add_subparsers(help='file_operations')
    
    # subparser for move op
    move_parser = subparsers.add_parser("move")
    move_parser.set_defaults(func=move)
    move_parser.add_argument("src_path")
    move_parser.add_argument("dest_path")

  # subparser for remove op
    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument(
        "to_remove",
        type = str)

    # subparser for copy op
    copy_parser = subparsers.add_parser("copy")
    copy_parser.set_defaults(func=copy)
    copy_parser.add_argument(
        "source_path",
        type = str)
    copy_parser.add_argument(
        "dest_path",
        type = str)
    
    # subparser for symlink op
    symlink_parser = subparsers.add_parser("symlink")
    symlink_parser.set_defaults(func=symlink)
    symlink_parser.add_argument(
        "source_path",
        type = str)
    symlink_parser.add_argument(
        "dest_path",
        type = str)

    # subparser for configure op
    configure_parser = subparsers.add_parser("configure")
    configure_parser.set_defaults(func=configure)
    configure_parser.add_argument(
        "source_path",
        type = str)
    configure_parser.add_argument(
        "dest_path",
        type = str)
    configure_parser.add_argument(
        "tag_delimiter",
        type = str)
    configure_parser.add_argument( 
        "param_dict",
        type = str)

    return arg_parser


def parse_arguments() -> str: # -> TelemetryMonitorArgs:
    # """Parse the command line arguments and return an instance
    # of TelemetryMonitorArgs populated with the CLI inputs

    # :returns: `TelemetryMonitorArgs` instance populated with command line arguments
    #"""
    parser = get_parser()
    parsed_args = parser.parse_args()

    parsed_args.func(parsed_args)
    #print(parsed_args)
    # return TelemetryMonitorArgs(
    #     parsed_args.exp_dir,
    #     parsed_args.frequency,
    #     parsed_args.cooldown,
    #     parsed_args.loglevel,
    # )
    #parsed_args.func(parsed_args)
    return parsed_args




if __name__ == "__main__":
    """Prepare the telemetry monitor process using command line arguments.

    Sample usage:
    python -m smartsim._core.entrypoints.telemetrymonitor -exp_dir <exp_dir>
          -frequency 30 -cooldown 90 -loglevel INFO
    The experiment id is generated during experiment startup
    and can be found in the manifest.json in <exp_dir>/.smartsim/telemetry
    """
    os.environ["PYTHONUNBUFFERED"] = "1"

    args = parse_arguments() #JPNOTE get from here  - pos args, first one in, the rest will get fed into the rest of the functions
    #sys args? - some number of strings that come after
    #configure_logger(logger, args.log_level, args.exp_dir)

    print(args)

    print("IN MAIN HERE")
    import json



   # if arg.copy
   # my_dictionary = json.loads(args.my_dict)

      # args = parser.parse_args()
    

    # def command1(args):
    #     print("command1: %s" % args.name)

    # def command2(args):
    #     print("comamnd2: %s" % args.frequency)

    # if __name__ == '__main__':
    #     main()

    #ns = parser.parse_args(args)

    # if args[1] == 'remove':

    #     to_remove = args[2]
    #     remove_op(to_remove)

    # if args[1] == 'move':
    #     to_move = args [2]
    #     entity_path = args[3]
    #     move_op(to_move, entity_path)

    # if args[1] == 'symlink':

    #     symlink_op()
    
    # if args[1] == 'copy':
    #     copy_op()

    # if args[1] == 'configure':
    #     configure_op()


 
    





   # sys.exit(1) # do I need? 
