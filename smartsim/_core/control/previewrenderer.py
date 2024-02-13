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

import pathlib
import typing as t
from enum import Enum

import jinja2
from jinja2 import pass_eval_context
import jinja2.utils as u
from ..._core.config import CONFIG
from ..._core.control import Manifest
from ...error.errors import PreviewFormatError
from ...log import get_logger

logger = get_logger(__name__)

if t.TYPE_CHECKING:
    from smartsim import Experiment


class Format(str, Enum):
    PLAINTEXT = "plain_text"


class Verbosity(str, Enum):
    INFO = "info"
    DEBUG = "debug"
    DEVELOPER = "developer"


def render(
    exp: "Experiment",
    manifest: t.Optional[Manifest] = None,
    verbosity_level: Verbosity = Verbosity.INFO,
    output_format: Format = Format.PLAINTEXT,
) -> str:
    """
    Render the template from the supplied entities.
    :param experiment: the experiment to be previewed.
    :type experiment: Experiment
    :param manifest: the manifest to be previewed.
    :type manifest: Manifest
    :param verbosity_level: the verbosity level
    :type verbosity_level: Verbosity
    :param output_format: the output format.
    :type output_format: _OutputFormatString
    """

    verbosity_level = _check_verbosity_level(verbosity_level)

    loader = jinja2.PackageLoader("templates")
    env = jinja2.Environment(loader=loader, autoescape=True)

    env.filters["as_toggle"] = as_toggle

    version = f"_{output_format}"
    tpl_path = f"preview/base{version}.template"

    _check_output_format(output_format)

    tpl = env.get_template(tpl_path)

    rendered_preview = tpl.render(
        exp_entity=exp,
        manifest=manifest,
        config=CONFIG,
        verbosity_level=verbosity_level,
    )
    return rendered_preview


@pass_eval_context
def as_toggle(_eval_ctx: u.F, value: bool) -> str:
    return "On" if value else "Off"


def preview_to_file(content: str, filename: str) -> None:
    """
    Output preview to a file if an output filename
    is specified.
    """
    filename = find_available_filename(filename)

    with open(filename, "w", encoding="utf-8") as prev_file:
        prev_file.write(content)


def find_available_filename(filename: str) -> str:
    """Iterate through potentially unique names until one is found that does
    not already exist. Return an unused name variation"""

    path = pathlib.Path(filename)
    candidate_path = pathlib.Path(filename)
    index = 1

    while candidate_path.exists():
        candidate_path = path.with_stem(f"{path.stem}_{index}")
        index += 1
    return str(candidate_path)


def _check_output_format(output_format: Format) -> None:
    """
    Check that the output format given is valid.
    """
    if not output_format == Format.PLAINTEXT:
        raise PreviewFormatError(
            f"The only valid output format currently available is {Format.PLAINTEXT}"
        )


def _check_verbosity_level(
    verbosity_level: Verbosity,
) -> Verbosity:
    """
    Check that the given verbosity level is valid.
    """
    if not isinstance(verbosity_level, Verbosity):
        logger.warning(
            f"'{verbosity_level}' is an unsupported verbosity level.\
 Setting verbosity to: {Verbosity.INFO}"
        )
        return Verbosity.INFO
    return verbosity_level
