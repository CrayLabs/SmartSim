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

import re
import typing as t

from smartsim.error.errors import SmartSimError

from ...error import ParameterWriterError
from ...log import get_logger

logger = get_logger(__name__)


class ModelWriter:
    def __init__(self) -> None:
        self.tag = ";"
        self.regex = "(;[^;]+;)"
        self.lines: t.List[str] = []

    def set_tag(self, tag: str, regex: t.Optional[str] = None) -> None:
        """Set the tag for the modelwriter to search for within
           tagged files attached to an entity.

        :param tag: tag for the modelwriter to search for,
                    defaults to semi-colon e.g. ";"
        :type tag: str
        :param regex: full regex for the modelwriter to search for,
                     defaults to "(;.+;)"
        :type regex: str, optional
        """
        if regex:
            self.regex = regex
        else:
            self.tag = tag
            self.regex = "".join(("(", tag, ".+", tag, ")"))

    def configure_tagged_model_files(
        self,
        tagged_files: t.List[str],
        params: t.Dict[str, str],
        make_missing_tags_fatal: bool = False,
    ) -> None:
        """Read, write and configure tagged files attached to a Model
           instance.

        :param tagged_files: list of paths to tagged files
        :type model: list[str]
        :param params: model parameters
        :type params: dict[str, str]
        :param make_missing_tags_fatal: blow up if a tag is missing
        :type make_missing_tags_fatal: bool
        """
        for tagged_file in tagged_files:
            self._set_lines(tagged_file)
            self._replace_tags(params, make_missing_tags_fatal)
            self._write_changes(tagged_file)

    def _set_lines(self, file_path: str) -> None:
        """Set the lines for the modelwrtter to iterate over

        :param file_path: path to the newly created and tagged file
        :type file_path: str
        :raises ParameterWriterError: if the newly created file cannot be read
        """
        try:
            with open(file_path, "r+", encoding="utf-8") as file_stream:
                self.lines = file_stream.readlines()
        except (IOError, OSError) as e:
            raise ParameterWriterError(file_path) from e

    def _write_changes(self, file_path: str) -> None:
        """Write the ensemble-specific changes

        :raises ParameterWriterError: if the newly created file cannot be read
        """
        try:
            with open(file_path, "w+", encoding="utf-8") as file_stream:
                for line in self.lines:
                    file_stream.write(line)
        except (IOError, OSError) as e:
            raise ParameterWriterError(file_path, read=False) from e

    def _replace_tags(self, params: t.Dict[str, str], make_fatal: bool = False) -> None:
        """Replace the tagged within the tagged file attached to this
           model. The tag defaults to ";"

        :param model: The model instance
        :type model: Model
        :param make_fatal: (Optional) Set to True to force a fatal error
            if a tag is not matched
        :type make_fatal: bool
        """
        edited = []
        unused_tags: t.Dict[str, t.List[int]] = {}
        for i, line in enumerate(self.lines):
            search = re.search(self.regex, line)
            if search:
                while search:
                    tagged_line = search.group(0)
                    previous_value = self._get_prev_value(tagged_line)
                    if self._is_ensemble_spec(tagged_line, params):
                        new_val = str(params[previous_value])
                        new_line = re.sub(self.regex, new_val, line, 1)
                        search = re.search(self.regex, new_line)
                        if not search:
                            edited.append(new_line)
                        else:
                            line = new_line

                    # if a tag is found but is not in this model's configurations
                    # put in placeholder value
                    else:
                        tag = tagged_line.split(self.tag)[1]
                        if tag not in unused_tags:
                            unused_tags[tag] = []
                        unused_tags[tag].append(i + 1)
                        edited.append(re.sub(self.regex, previous_value, line))
                        search = None  # Move on to the next tag
            else:
                edited.append(line)
        for tag, value in unused_tags.items():
            missing_tag_message = (
                f"Unused tag {tag} on line(s): {str(value)}"
            )
            if make_fatal:
                raise SmartSimError(missing_tag_message)
            logger.warning(missing_tag_message)
        self.lines = edited

    def _is_ensemble_spec(
        self, tagged_line: str, model_params: t.Dict[str, str]
    ) -> bool:
        split_tag = tagged_line.split(self.tag)
        prev_val = split_tag[1]
        if prev_val in model_params.keys():
            return True
        return False

    def _get_prev_value(self, tagged_line: str) -> str:
        split_tag = tagged_line.split(self.tag)
        return split_tag[1]
