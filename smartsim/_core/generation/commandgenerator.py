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


def move_op(to_move: str, dest_file: str):
    """Write a python script to moves a file

    :param to_move: Path to a source file to be copied
    :param dest_file: Path to a file to copy the contents from the source file into
    :return: string of python code to perform operation that can be executed
    """
    return f"import shutil;shutil.move('{to_move}','{dest_file}')"


def delete_op(to_delete: str):
    """Write a python script that deletes a file when executed.

    :param to_delete: Path to the file to be deleted
    :return: string of python code to perform operation that can be executed
    """

    if path.isfile(to_delete):
        return rf"""
import os
os.remove('{to_delete}')
    """
    else:
        raise FileNotFoundError(f"{to_delete} is not a valid path")


def copy_op(to_copy: str, entity_path: str):
    """
    Write a python script to copy the entity files and directories attached
    to this entity into an entity directory

    :param to_copy: Path to directory, or path to file to copy into an entity directory
    :param entity_path: Path to a directory of an entity
    :return: string of python code to perform operation that can be executed
    """

    if not path.exists(entity_path):
        raise FileNotFoundError(f"{entity_path} is not a valid path")
    elif not path.exists(to_copy):
        raise FileNotFoundError(f"{to_copy} is not a valid path")

    else:

        return rf"""import shutil
from distutils import dir_util
from os import path
dst_path = path.join('{entity_path}', path.basename('{to_copy}'))


if path.isdir('{to_copy}'):
  dir_util.copy_tree('{to_copy}', '{entity_path}')
else:
  shutil.copyfile('{to_copy}', dst_path)
"""


def symlink_op(to_link: str, entity_path: str):
    """
    Write a python script to make the to_link path a symlink pointing to the entity path.

    :param to_link: path to link
    :param entity_path:
    :return: string of python code to perform operation that can be executed
    """
    return rf"""import os
from os import path
dest_path = path.join('{entity_path}', path.basename('{to_link}'))

os.symlink('{to_link}', dest_path)
  """


def configure_op(
    to_configure: str, params: t.Dict[str, str], tag_delimiter: t.Optional[str] = None
):
    """Write a python script to set, search and replace the tagged parameters for the configure operation
    within tagged files attached to an entity.

    :param to configure:
    :param params: A dict of parameter names and values set for the file
    :tag_delimiter: tag for the configure operation to search for, defaults to semi-colon e.g. ";"
    :return: string of python code to perform operation that can be executed
    """

    return rf"""import re
import collections

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
used_params = {{}}
params = {params}

tag_delimiter = '{tag_delimiter}'

if tag_delimiter == 'False':
  tag_delimiter = ';'

regex = "".join(("(", tag_delimiter, ".+", tag_delimiter, ")"))

# Set the lines to iterate over
with open('{to_configure}','r+', encoding='utf-8') as file_stream:
  lines = file_stream.readlines()

unused_tags = collections.defaultdict(list)

# read lines in file
for i, line in enumerate(lines, 1):
    while search := re.search(regex, line):
        tagged_line = search.group(0)
        previous_value = _get_prev_value(tagged_line)
        if _is_ensemble_spec(tagged_line, params):
            new_val = str(params[previous_value])
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

with open('{to_configure}', "w+", encoding="utf-8") as file_stream:
    for line in lines:
        file_stream.write(line)
  """
