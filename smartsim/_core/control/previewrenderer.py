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
from ...log import get_logger
from ..._core.config import CONFIG
from ..._core.control import Manifest

logger = get_logger(__name__)

if t.TYPE_CHECKING:
    from smartsim import Experiment

_OutputFormatString = t.Optional[t.Literal["html"]]
_VerbosityLevelString = t.Literal["info", "debug", "developer"]


def render(
    exp: "Experiment",
    manifest: t.Optional[Manifest] = None,
    verbosity_level: _VerbosityLevelString = "info",
    output_format: _OutputFormatString = None,
) -> str:
    """
    Render the template from the supplied entities.
    :param experiment: the experiment to be previewed.
    :type experiment: Experiment
    :param manifest: the manifest to be previewed.
    :type manifest: Manifest
    :param verbosity_level: the verbosity level
    :type verbosity_level: _VerbosityLevelString
    :param output_format: the output destination.
    If no output format is set, the preview will be output to stdout
    :type output_format: _OutputFormatString
    """

    verbosity_level = _check_verbosity_level(verbosity_level)

    loader = jinja2.PackageLoader("templates")
    env = jinja2.Environment(loader=loader, autoescape=True)

    version = f"_{output_format}" if output_format else ""
    tpl_path = f"preview/base{version}.template"

    if output_format:
        _check_output_format(output_format)

    tpl = env.get_template(tpl_path)

    rendered_preview = tpl.render(
        exp_entity=exp,
        manifest=manifest,
        config=CONFIG,
        verbosity_level=verbosity_level,
    )

    return rendered_preview


def preview_to_file(content: str, filename: str) -> None:
    """
    Output preview to a file if output format and filename
    are specified.
    """

    with open(filename, "w", encoding="utf-8") as prev_file:
        prev_file.write(content)


def _check_output_format(output_format: str) -> None:
    """
    Check that the output format given is valid.
    """
    if not output_format == "html":
        raise ValueError("The only valid currently available is html")


def _check_verbosity_level(verbosity_level: _VerbosityLevelString) -> str:
    """
    Check that the given verbosity level is valid.
    """
    verbosity_level = t.cast(_VerbosityLevelString, verbosity_level)

    if verbosity_level not in ["info", "debug", "developer"]:
        raise ValueError("The only valid verbosity level currently available is info")

    if verbosity_level in ("debug", "developer"):
        logger.warning(
            f"'{verbosity_level}' is an unsupported verbosity level requested.\
Setting verbosity to: info"
        )
        return "info"
    return "info"



#     def model_properties(self, *args) -> str:
#         """get the model properties"""
#         # just get what you can from the model here

#         # for all models
#         print("\n\n--------------\n")
#         models = self.models

#         #  "model": [
#         #     _dictify_model(
#         #         model,
#         #         *telemetry_metadata,
#         #         telemetry_data_root / "model",
#         #     )
#         #     for model, telemetry_metadata in manifest.models
#         for model in models:
#             print(f"name: {model.name}")
#             print(f"path: {model.path}")
#             # print(f"run_settings: {_dictify_run_settings(model.run_settings)}")
#             print(f"exe: {model.run_settings.exe}")
#             print(f"run_command: {model.run_settings.exe_args}")
#             print(f"run_args: {model.run_settings.run_args}")
#             # print(f"batch_settings: {_dictify_batch_settings(model.batch_settings)}")
#             if model.batch_settings:
#                 print(f"batch_command: {model.batch_settings.batch_cmd}")
#                 print(f"batch_args: {model.batch_settings.batch_args}")
#             print(f"params: {model.params}")
#             if model.files:
#                 print(f"Symlink: {model.files.link}")
#                 print(f"Configure: {model.files.tagged}")
#                 print(f"Copy: {model.files.copy}")

#             # COLOCATED STUFF
#             colo_settings = (model.run_settings.colocated_db_settings or {}).copy()
#             if colo_settings:
#                 # "colocated_db": {}
#                 print(f"settings: {model.colo_settings}")

#                 # how to get db scripts
#                 db_scripts = t.cast(
#                     "t.List[DBScript]", colo_settings.pop("db_scripts", [])
#                 )

#                 for script in db_scripts:
#                     # "scripts":
#                     print("scripts")
#                     print(f"script name: {script.name}")
#                     print(f"backend: TORCH")  # why is this hardcoded as
#                     print(f"device: {script.device}")

#                 # how to get db_models
#                 db_models = t.cast(
#                     "t.List[DBModel]", colo_settings.pop("db_models", [])
#                 )
#                 for model in db_models:
#                     print("models")
#                     print(f"{model.name}")
#                     print(f"backend: {model.backend}")
#                     print(f"device: {model.device}")

#     # looks at the step ---- so will need the step creation data for out and err files
#     # out_file, err_file = step.get_output_files()
#     # def get_step_file(
#     #     self, ending: str = ".sh", script_name: t.Optional[str] = None
#     # ) -> str:
#     #     """Get the name for a file/script created by the step class

#     #     Used for Batch scripts, mpmd scripts, etc.
#     #     """
#     #     if script_name:
#     #         script_name = script_name if "." in script_name else script_name + ending
#     #         return osp.join(self.cwd, script_name)
#     #     return osp.join(self.cwd, self.entity_name + ending)
#     #   def get_output_files(self) -> t.Tuple[str, str]:
#     #     """Return two paths to error and output files based on cwd"""
#     #     output = self.get_step_file(ending=".out")
#     #     error = self.get_step_file(ending=".err")
#     #     return output, error

#     # dictify_model:
#     #  colo_settings = (model.run_settings.colocated_db_settings or {}).copy()
#     #     db_scripts = t.cast("t.List[DBScript]", colo_settings.pop("db_scripts", []))
#     #     db_models = t.cast("t.List[DBModel]", colo_settings.pop("db_models", []))
#     #     return {

#     #          "batch_settings": _dictify_batch_settings(model.batch_settings)
#     #         if model.batch_settings
#     #         else {},
#     #         "params": model.params,
#     #         "files": {
#     #             "Symlink": model.files.link,
#     #             "Configure": model.files.tagged,
#     #             "Copy": model.files.copy,
#     #         }
#     #         if model.files
#     #         else {
#     #             "Symlink": [],
#     #             "Configure": [],
#     #             "Copy": [],
#     #         },
#     #         "colocated_db": {
#     #             "settings": colo_settings,
#     #             "scripts": [
#     #                 {
#     #                     script.name: {
#     #                         "backend": "TORCH",

#     #                         "device": script.device,
#     #                     }
#     #                 }
#     #                 for script in db_scripts
#     #             ],
#     #             "models": [
#     #                 {
#     #                     model.name: {
#     #                         "backend": model.backend,
#     #                         "device": model.device,
#     #                     }
#     #                 }
#     #                 for model in db_models
#     #             ],
#     #         }
#     #         if colo_settings
#     #         else {},
#     #         "telemetry_metadata": {
#     #             "status_dir": str(telemetry_data_path / model.name),
#     #             "step_id": step_id,
#     #             "task_id": task_id,
#     #             "managed": managed,
#     #         },
#     #         "out_file": out_file,
#     #         "err_file": err_file,
#     #     }

#     #         telemetry_metadata :

#     #         telemetry_data_root :

#     def render(self, *args) -> str:
#         """
#         Render the template from the supplied entity
#         """
#         loader = jinja2.PackageLoader("templates")
#         env = jinja2.Environment(loader=loader, autoescape=True)
#         tpl = env.get_template("master.pytpl")
#         # print("THIS SHOULD BE THE MODEL", self.model)
#         template = tpl.render(
#             exp_entity=self.exp_entity,
#             model=self.models[0],
#             books=["one", "two", "three"],
#         )


#         return template
