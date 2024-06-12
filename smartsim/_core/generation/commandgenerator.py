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

# TODO: rename and make all the descriptions accurate and good
# including the params and stuff

def delete_op(to_delete: str):
  """Produce string of a script that deletes a file when executed.

  :param to_delete: path to the file to be deleted
  :return: string to be shipped as a command that deletes the given file
    """
  return f"import os;os.remove('{to_delete}')"


def copy_op(source_file: str, dest_file: str):
   """return command that can be popened"""
   return f"import shutil;shutil.copyfile('{source_file}', '{dest_file}')"


def symlink_op(dest_file: str, source_file: str):
  """
  Make the source path a symlink pointing to the destination path.

  :param source_file:
  :param dest_file:
  :return:
  """
  return f"import os;os.symlink(str('{source_file}'), str('{dest_file}'))"

def move_op(source_file, dest_file):
    """Execute a script that moves a file"""
    return f"import shutil;shutil.move('{source_file}','{dest_file}')"


def configure_op(file_path, param_dict):
  """configure file with params
  TODO: clean this up"""

  return fr"""
import re
import collections

edited = []
used_params = {{}}
files_to_tags = {{}}
params = {param_dict}
regex = "(;.+;)"
tag = ";"

def _get_prev_value(tagged_line: str) -> str:
    split_tag = tagged_line.split(tag)
    return split_tag[1]

def _is_ensemble_spec(
    tagged_line: str, model_params: dict
  ) -> bool:
      split_tag = tagged_line.split(tag)
      prev_val = split_tag[1]
      if prev_val in model_params.keys():
          return True
      return False

with open('{file_path}','r+', encoding='utf-8') as file_stream:
  lines = file_stream.readlines()

unused_tags = collections.defaultdict(list)

for i, line in enumerate(lines, 1):
    while search := re.search(regex, line):
        tagged_line = search.group(0)
        previous_value = _get_prev_value(tagged_line)
        if _is_ensemble_spec(tagged_line, params):
            new_val = str(params[previous_value])
            line = re.sub(regex, new_val, line, 1)
            used_params[previous_value] = new_val

        # if a tag is found but is not in this model's configurations
        # put in placeholder value
        else:
            tag = tagged_line.split(tag)[1]
            unused_tags[tag].append(i)
            line = re.sub(regex, previous_value, line)
            break
    edited.append(line)

lines = edited

with open('{file_path}', "w+", encoding="utf-8") as file_stream:
    for line in lines:
        file_stream.write(line)
"""
