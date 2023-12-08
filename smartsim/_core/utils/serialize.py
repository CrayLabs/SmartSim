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

from __future__ import annotations

import json
import time
import typing as t
from pathlib import Path

import smartsim._core._cli.utils as _utils
import smartsim.log
from smartsim._core.config import CONFIG

if t.TYPE_CHECKING:
    from smartsim import Experiment
    from smartsim._core.control.manifest import LaunchedManifest as _Manifest
    from smartsim.database.orchestrator import Orchestrator
    from smartsim.entity import DBNode, Ensemble, Model
    from smartsim.entity.dbobject import DBModel, DBScript
    from smartsim.settings.base import BatchSettings, RunSettings


TStepLaunchMetaData = t.Tuple[
    t.Optional[str], t.Optional[str], t.Optional[bool], str, str, Path
]
TELMON_SUBDIR: t.Final[str] = ".smartsim/telemetry"
MANIFEST_FILENAME: t.Final[str] = "manifest.json"

_LOGGER = smartsim.log.get_logger(__name__)


def save_launch_manifest(manifest: _Manifest[TStepLaunchMetaData]) -> None:
    if not CONFIG.telemetry_enabled:
        return

    manifest.metadata.run_telemetry_subdirectory.mkdir(parents=True, exist_ok=True)

    new_run = {
        "run_id": manifest.metadata.run_id,
        "timestamp": int(time.time_ns()),
        "model": [
            _dictify_model(model, *telemetry_metadata)
            for model, telemetry_metadata in manifest.models
        ],
        "orchestrator": [
            _dictify_db(db, nodes_info) for db, nodes_info in manifest.databases
        ],
        "ensemble": [
            _dictify_ensemble(ens, member_info)
            for ens, member_info in manifest.ensembles
        ],
    }
    try:
        with open(manifest.metadata.manifest_file_path, "r", encoding="utf-8") as file:
            manifest_dict = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        manifest_dict = {
            "schema info": {
                "schema_name": "entity manifest",
                "version": "0.0.2",
            },
            "experiment": {
                "name": manifest.metadata.exp_name,
                "path": manifest.metadata.exp_path,
                "launcher": manifest.metadata.launcher_name,
                "out_file": str(Path(manifest.metadata.exp_path) / "smartsim.out"),
                "err_file": str(Path(manifest.metadata.exp_path) / "smartsim.err"),
            },
            "runs": [new_run],
        }
    else:
        manifest_dict["runs"].append(new_run)
    finally:
        with open(manifest.metadata.manifest_file_path, "w", encoding="utf-8") as file:
            json.dump(manifest_dict, file, indent=2)


def _dictify_model(
    model: Model,
    step_id: t.Optional[str],
    task_id: t.Optional[str],
    managed: t.Optional[bool],
    out_file: str,
    err_file: str,
    telemetry_data_path: Path,
) -> t.Dict[str, t.Any]:
    colo_settings = (model.run_settings.colocated_db_settings or {}).copy()
    db_scripts = t.cast("t.List[DBScript]", colo_settings.pop("db_scripts", []))
    db_models = t.cast("t.List[DBModel]", colo_settings.pop("db_models", []))
    return {
        "name": model.name,
        "path": model.path,
        "exe_args": model.run_settings.exe_args,
        "run_settings": _dictify_run_settings(model.run_settings),
        "batch_settings": (
            _dictify_batch_settings(model.batch_settings)
            if model.batch_settings
            else {}
        ),
        "params": model.params,
        "files": (
            {
                "Symlink": model.files.link,
                "Configure": model.files.tagged,
                "Copy": model.files.copy,
            }
            if model.files
            else {
                "Symlink": [],
                "Configure": [],
                "Copy": [],
            }
        ),
        "colocated_db": (
            {
                "settings": colo_settings,
                "scripts": [
                    {
                        script.name: {
                            "backend": "TORCH",
                            "device": script.device,
                        }
                    }
                    for script in db_scripts
                ],
                "models": [
                    {
                        model.name: {
                            "backend": model.backend,
                            "device": model.device,
                        }
                    }
                    for model in db_models
                ],
            }
            if colo_settings
            else {}
        ),
        "telemetry_metadata": {
            "status_dir": str(telemetry_data_path),
            "step_id": step_id,
            "task_id": task_id,
            "managed": managed,
        },
        "out_file": out_file,
        "err_file": err_file,
    }


def _dictify_ensemble(
    ens: Ensemble,
    members: t.Sequence[t.Tuple[Model, TStepLaunchMetaData]],
) -> t.Dict[str, t.Any]:
    return {
        "name": ens.name,
        "params": ens.params,
        "batch_settings": (
            _dictify_batch_settings(ens.batch_settings)
            # FIXME: Typehint here is wrong, ``ens.batch_settings`` can
            # also be an empty dict for no discernible reason...
            if ens.batch_settings
            else {}
        ),
        "models": [
            _dictify_model(model, *launching_metadata)
            for model, launching_metadata in members
        ],
    }


def _dictify_run_settings(run_settings: RunSettings) -> t.Dict[str, t.Any]:
    # TODO: remove this downcast
    if hasattr(run_settings, "mpmd") and run_settings.mpmd:
        _LOGGER.warning(
            "SmartSim currently cannot properly serialize all information in "
            "MPMD run settings"
        )
    return {
        "exe": run_settings.exe,
        # TODO: We should try to move this back
        # "exe_args": run_settings.exe_args,
        "run_command": run_settings.run_command,
        "run_args": run_settings.run_args,
        # TODO: We currently do not have a way to represent MPMD commands!
        #       Maybe add a ``"mpmd"`` key here that is a
        #       ``list[TDictifiedRunSettings]``?
    }


def _dictify_batch_settings(batch_settings: BatchSettings) -> t.Dict[str, t.Any]:
    return {
        "batch_command": batch_settings.batch_cmd,
        "batch_args": batch_settings.batch_args,
    }


def _dictify_db(
    db: Orchestrator,
    nodes: t.Sequence[t.Tuple[DBNode, TStepLaunchMetaData]],
) -> t.Dict[str, t.Any]:
    db_path = _utils.get_db_path()
    if db_path:
        db_type, _ = db_path.name.split("-", 1)
    else:
        db_type = "Unknown"
    return {
        "name": db.name,
        "type": db_type,
        "interface": db._interfaces,  # pylint: disable=protected-access
        "shards": [
            {
                **shard.to_dict(),
                "conf_file": shard.cluster_conf_file,
                "out_file": out_file,
                "err_file": err_file,
                "telemetry_metadata": {
                    "status_dir": str(status_dir),
                    "step_id": step_id,
                    "task_id": task_id,
                    "managed": managed,
                },
            }
            for dbnode, (
                step_id,
                task_id,
                managed,
                out_file,
                err_file,
                status_dir,
            ) in nodes
            for shard in dbnode.get_launched_shard_info()
        ],
    }
