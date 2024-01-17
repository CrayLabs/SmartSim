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
import typing as t
import jinja2
import logging
from ...log import log_to_file_preview
from pathlib import Path

from smartsim._core.control.manifest import LaunchedManifest as _Manifest

TStepLaunchMetaData = t.Tuple[
    t.Optional[str], t.Optional[str], t.Optional[bool], str, str, Path
]


_OutputFormatString = t.Optional[t.Literal["html"]]
_VerbosityLevelString = t.Literal["info", "debug", "developer"] 

def render(
    experiment,
    manifest,
    verbosity_level: _VerbosityLevelString = "info",
    output_format: _OutputFormatString = None,
    output_filename: t.Optional[str] = None,
) -> str:
    """
    Render the template from the supplied entity
    """

    _check_verbosity_level(verbosity_level)

    loader = jinja2.PackageLoader("templates")
    env = jinja2.Environment(loader=loader, autoescape=True)
    if output_format:
        _check_output_format(output_format)
        output_filename = _check_output_filename(output_filename)

        tpl = env.get_template(f"base_{output_format}.template")
        rendered_preview = tpl.render(
            exp_entity=experiment, manifest=manifest, verbosity_level=verbosity_level
        )
        preview_to_file(rendered_preview, output_filename)
    else:
        if output_filename:
            raise ValueError(
                "Output filename is only a valid parameter when an output \
format is specified"
            )
        tpl = env.get_template("base.template")
        rendered_preview = tpl.render(
            exp_entity=experiment, manifest=manifest, verbosity_level=verbosity_level
        )

    #_Manifest()

    #print(LaunchedManifest[TStepLaunchMetaData].databases)

    logging.info(rendered_preview)
    return rendered_preview




#          for shard in dbnode.get_launched_shard_info()
#          **shard.to_dict(),
#      for dbnode, (step_id, task_id, managed, out_file, err_file) in nodes
#  for shard in dbnode.get_launched_shard_info()

#         "orchestrator": [
#             _dictify_db(
#                 db, nodes_info, telemetry_data_root / "database"
#             )
#             for db, nodes_info in manifest.databases
#         ],

#         def _dictify_db(
#     db: Orchestrator,
#     nodes: t.Sequence[t.Tuple[DBNode, TStepLaunchMetaData]],
#     telemetry_data_path: Path,
# ) -> t.Dict[str, t.Any]:
#     db_path = _utils.get_db_path()
#     if db_path:
#         db_type, _ = db_path.name.split("-", 1)
#     else:
#         db_type = "Unknown"
#     return {
#         "name": db.name,
#         "type": db_type,
#         "interface": db._interfaces,
#         "shards": [
#             {
#                 **shard.to_dict(),
#                 "conf_file": shard.cluster_conf_file,
#                 "out_file": out_file,
#                 "err_file": err_file,
#                 "telemetry_metadata": {
#                     "status_dir": str(
#                         telemetry_data_path / f"{db.name}/{dbnode.name}"
#                     ),
#                     "step_id": step_id,
#                     "task_id": task_id,
#                     "managed": managed,
#                 },
#             }
#             for dbnode, (step_id, task_id, managed, out_file, err_file) in nodes
#             for shard in dbnode.get_launched_shard_info()
#         ],
#     }






def preview_to_file(content: str, file_name: str) -> None:
    logger = logging.getLogger("preview-logger")
    log_to_file_preview(filename=file_name, logger=logger)
    logger.info(content)


def _check_output_format(output_format: str) -> None:
    if not output_format.startswith("html"):
        raise ValueError("The only valid currently available is html")


def _check_output_filename(output_filename: t.Optional[str]) -> str:
    if not output_filename:
        raise ValueError("An output filename is required when an output format is set.")

    return output_filename

def _check_verbosity_level(
    verbosity_level: _VerbosityLevelString
) -> None:
    """
    Check verbosity_level
    """
    if verbosity_level == "debug":
        raise NotImplementedError
    if verbosity_level == "developer":
        raise NotImplementedError
    verbosity_level = t.cast(_VerbosityLevelString, verbosity_level)
    if (
        not verbosity_level.startswith("info")
        and not verbosity_level.startswith("debug")
        and not verbosity_level.startswith("developer")
    ):
        raise ValueError("The only valid verbosity level currently available is info")
