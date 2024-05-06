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
import jinja2.utils as u
from jinja2 import pass_eval_context

from ..._core.config import CONFIG
from ..._core.control import Manifest
from ...error.errors import PreviewFormatError
from ...log import get_logger
from .job import Job

logger = get_logger(__name__)

if t.TYPE_CHECKING:
    from smartsim import Experiment


class Format(str, Enum):
    PLAINTEXT = "plain_text"


class Verbosity(str, Enum):
    INFO = "info"
    DEBUG = "debug"
    DEVELOPER = "developer"


@pass_eval_context
def as_toggle(_eval_ctx: u.F, value: bool) -> str:
    """Return "On" if value returns True,
    and "Off" is value returns False.
    """
    return "On" if value else "Off"


@pass_eval_context
def get_ifname(_eval_ctx: u.F, value: t.List[str]) -> str:
    """Extract Network Interface from orchestrator run settings."""
    if value:
        for val in value:
            if "ifname=" in val:
                output = val.split("=")[-1]
                return output
    return ""


@pass_eval_context
def get_dbtype(_eval_ctx: u.F, value: str) -> str:
    """Extract data base type."""
    if value:
        if "-cli" in value:
            db_type, _ = value.split("/")[-1].split("-", 1)
            return db_type
    return ""


@pass_eval_context
def is_list(_eval_ctx: u.F, value: str) -> bool:
    """Return True if item is of type list, and False
    otherwise, to determine how Jinja template should
    render an item.
    """
    return isinstance(value, list)


def render_to_file(content: str, filename: str) -> None:
    """Output preview to a file if an output filename
    is specified.

    :param content: The rendered preview.
    :param filename: The name of the file to write the preview to.
    """
    filename = find_available_filename(filename)

    with open(filename, "w", encoding="utf-8") as prev_file:
        prev_file.write(content)


def render(
    exp: "Experiment",
    manifest: t.Optional[Manifest] = None,
    verbosity_level: Verbosity = Verbosity.INFO,
    output_format: Format = Format.PLAINTEXT,
    output_filename: t.Optional[str] = None,
    active_dbjobs: t.Optional[t.Dict[str, Job]] = None,
) -> str:
    """
    Render the template from the supplied entities.
    :param experiment: the experiment to be previewed.
    :param manifest: the manifest to be previewed.
    :param verbosity_level: the verbosity level
    :param output_format: the output format.
    """

    verbosity_level = Verbosity(verbosity_level)

    _check_output_format(output_format)

    loader = jinja2.PackageLoader(
        "smartsim.templates.templates.preview", output_format.value
    )
    env = jinja2.Environment(loader=loader, autoescape=True)

    env.filters["as_toggle"] = as_toggle
    env.filters["get_ifname"] = get_ifname
    env.filters["get_dbtype"] = get_dbtype
    env.filters["is_list"] = is_list
    env.globals["Verbosity"] = Verbosity

    tpl_path = "base.template"

    tpl = env.get_template(tpl_path)

    if verbosity_level == Verbosity.INFO:
        logger.warning(
            "Only showing user set parameters. Some internal entity "
            "fields are truncated. To view truncated fields: use verbosity_level "
            "'developer' or 'debug.'"
        )

    rendered_preview = tpl.render(
        exp_entity=exp,
        active_dbjobs=active_dbjobs,
        manifest=manifest,
        config=CONFIG,
        verbosity_level=verbosity_level,
    )

    if output_filename:
        render_to_file(
            rendered_preview,
            output_filename,
        )
    else:
        logger.info(rendered_preview)
    return rendered_preview


def find_available_filename(filename: str) -> str:
    """Iterate through potentially unique names until one is found that does
    not already exist. Return an unused name variation

    :param filename: The name of the file to write the preview to.
    """

    path = pathlib.Path(filename)
    candidate_path = pathlib.Path(filename)
    index = 1

    while candidate_path.exists():
        candidate_path = path.with_name(f"{path.stem}_{index:02}.txt")
        index += 1
    return str(candidate_path)


def _check_output_format(output_format: Format) -> None:
    """
    Check that a valid file output format is given.
    """
    if not output_format == Format.PLAINTEXT:
        raise PreviewFormatError(f"The only valid output format currently available \
is {Format.PLAINTEXT.value}")
