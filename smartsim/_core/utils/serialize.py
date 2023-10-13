from __future__ import annotations

import json
import time
import typing as t
import uuid
from pathlib import Path

import smartsim._core._cli.utils as _utils

if t.TYPE_CHECKING:
    from smartsim import Experiment
    from smartsim._core.control.manifest import LaunchedManifest as _Manifest
    from smartsim.database.orchestrator import Orchestrator
    from smartsim.entity import DBNode, Ensemble, Model
    from smartsim.entity.dbobject import DBModel, DBScript
    from smartsim.settings.base import BatchSettings, RunSettings


# Many of the required fields for serialization require attrs that protected.
# I do not want to accidentally expose something to users that should not see
# for a prototype feature. This suppression should probably be removed before
# this is officially released!
#
# pylint: disable=protected-access


TStepLaunchMetaData = t.Tuple[
    t.Optional[str], t.Optional[str], t.Optional[bool], str, str
]


def save_launch_manifest(manifest: _Manifest[TStepLaunchMetaData]) -> None:
    manifest_dir = Path(manifest.metadata.exp_path) / ".smartsim/manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "manifest.json"

    run_id = str(uuid.uuid4())
    telemetry_data_root = manifest_dir / f"{manifest.metadata.exp_name}/{run_id}"

    new_run = {
        "run_id": run_id,
        "timestamp": int(time.time_ns()),
        "model": [
            _dictify_model(
                model,
                *telemetry_metadata,
                telemetry_data_root / "model",
            )
            for model, telemetry_metadata in manifest.models
        ],
        "orchestrator": [
            _dictify_db(
                db, nodes_info, telemetry_data_root / "database"
            )
            for db, nodes_info in manifest.databases
        ],
        "ensemble": [
            _dictify_ensemble(
                ens, member_info, telemetry_data_root / "ensemble"
            )
            for ens, member_info in manifest.ensembles
        ],
    }
    try:
        with open(manifest_file, "r", encoding="utf-8") as file:
            manifest_dict = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        manifest_dict = {
            "experiment": {
                "name": manifest.metadata.exp_name,
                "path": manifest.metadata.exp_path,
                "launcher": manifest.metadata.launcher_name,
            },
            "runs": [new_run],
        }
    else:
        manifest_dict["runs"].append(new_run)
    finally:
        with open(manifest_file, "w", encoding="utf-8") as file:
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
        "run_settings": _dictify_run_settings(model.run_settings),
        "batch_settings": _dictify_batch_settings(model.batch_settings)
        if model.batch_settings
        else None,
        "params": model.params,
        "files": {
            "Symlink": model.files.link,
            "Configure": model.files.tagged,
            "Copy": model.files.copy,
        }
        if model.files
        else {
            "Symlink": [],
            "Configure": [],
            "Copy": [],
        },
        "colocated_db": {
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
        else None,
        "telemetry_metadata": {
            "status_dir": str(telemetry_data_path / model.name),
            "step_id": step_id,
            "task_id": task_id,
            "managed": managed,
        },
        "out_file": out_file,
        "error_file": err_file,
    }


def _dictify_ensemble(
    ens: Ensemble,
    members: t.Sequence[t.Tuple[Model, TStepLaunchMetaData]],
    telemetry_data_path: Path,
) -> t.Dict[str, t.Any]:
    return {
        "name": ens.name,
        "params": ens.params,
        "batch_settings": _dictify_batch_settings(ens.batch_settings)
        # FIXME: Typehint here is wrong, ``ens.batch_settings`` can
        # also be an empty dict for no discernible reason...
        if ens.batch_settings else None,
        "models": [
            _dictify_model(
                model, *launching_metadata, telemetry_data_path / ens.name
            )
            for model, launching_metadata in members
        ],
    }


def _dictify_run_settings(run_settings: RunSettings) -> t.Dict[str, t.Any]:
    return {
        "exe": run_settings.exe,
        "exe_args": run_settings.exe_args,
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
    telemetry_data_path: Path,
) -> t.Dict[str, t.Any]:
    db_path = _utils.get_db_path()
    if db_path:
        db_type, _ = db_path.name.split("-", 1)
    else:
        db_type = "Unknown"
    return {
        "name": db.name,
        "type": db_type,
        "interface": db._interfaces,
        "shards": [
            {
                **shard.to_dict(),
                "conf_file": shard.cluster_conf_file,
                "out_file": out_file,
                "err_file": err_file,
                "telemetry_metadata": {
                    "status_dir": str(
                        telemetry_data_path / f"{db.name}/{dbnode.name}"
                    ),
                    "step_id": step_id,
                    "task_id": task_id,
                    "managed": managed,
                },
            }
            for dbnode, (step_id, task_id, managed, out_file, err_file) in nodes
            for shard in dbnode.get_launched_shard_info()
        ],
    }
