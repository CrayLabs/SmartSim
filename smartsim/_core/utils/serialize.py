from __future__ import annotations

import json
import time
import typing as t
from pathlib import Path

import smartsim._core._cli.utils as _utils

if t.TYPE_CHECKING:
    from smartsim import Experiment
    from smartsim._core.control.manifest import LaunchedManifest as _Manifest
    from smartsim.database.orchestrator import Orchestrator
    from smartsim.entity import DBNode, Ensemble, Model
    from smartsim.settings.base import BatchSettings, RunSettings

_TStepLaunchMetaData = t.Tuple[
    t.Optional[str], t.Optional[str], t.Optional[bool], str, str
]


def save_launch_manifest(
    experiment: Experiment, manifest: _Manifest[_TStepLaunchMetaData]
) -> None:
    manifest_dir = Path(experiment.exp_path) / ".smartsim/manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / "manifest.json"

    # FIXME: Did we decide on if this should be uuid/literal count of the
    #        runs/time since epoch/etc.? Whatever it is, it probably should not
    #        be secs since epoch as this could lead to conflicts if many
    #        non-blocking runs are made in quick succession.
    run_id = int(time.time())

    new_run = {
        "run_id": run_id,
        "model": [
            _dictify_model(
                model,
                *telemetry_metadata,
                manifest_dir / f"{experiment.name}/{run_id}/model",
            )
            for model, telemetry_metadata in manifest.models
        ],
        "orchestrator": [
            _dictify_db(
                db, nodes_info, manifest_dir / f"{experiment.name}/{run_id}/database"
            )
            for db, nodes_info in manifest.database
        ],
        "ensemble": [
            _dictify_ensemble(
                ens, member_info, manifest_dir / f"{experiment.name}/{run_id}/ensemble"
            )
            for ens, member_info in manifest.ensembles
        ],
    }
    try:
        with open(manifest_file, "r") as fd:
            manifest_dict = json.load(fd)
    except (FileNotFoundError, json.JSONDecodeError):
        manifest_dict = {
            "experiment": _dictify_experiment(experiment),
            "runs": [new_run],
        }
    else:
        manifest_dict["runs"].append(new_run)
    finally:
        with open(manifest_file, "w") as fd:
            json.dump(manifest_dict, fd, indent=2)


def _dictify_experiment(exp: Experiment) -> t.Dict[str, str]:
    return {
        "name": exp.name,
        "path": exp.exp_path,
        "launcher": exp._launcher,
    }


def _dictify_model(
    model: Model,
    step_id: t.Optional[str],
    task_id: t.Optional[str],
    managed: t.Optional[bool],
    out_file: str,
    err_file: str,
    telemetry_data_path: Path,
) -> t.Dict[str, t.Any]:
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
            "settings": model.run_settings.colocated_db_settings,
            "scripts": [
                {
                    script.name: {
                        "backend": "TORCH",
                        "device": script.device,
                    }
                }
                for script in model._db_scripts
            ],
            "models": [
                {
                    model.name: {
                        "backend": model.backend,
                        "device": model.device,
                    }
                }
                for model in model._db_models
            ],
        }
        if model.run_settings.colocated_db_settings
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
    members: t.List[t.Tuple[Model, _TStepLaunchMetaData]],
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
                model, *launching_metadata, telemetry_data_path / f"ensemble/{ens.name}"
            )
            for model, launching_metadata in members
        ],
    }


def _dictify_run_settings(run_settings: RunSettings) -> t.Dict[str, t.Any]:
    return {
        "exe": run_settings.exe[0],
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
    nodes: t.List[t.Tuple[DBNode, _TStepLaunchMetaData]],
    telemetry_data_path: Path,
) -> t.Dict[str, t.Any]:
    db_path = _utils.get_db_path()
    if db_path:
        db_type, _ = db_path.name.split("-", 1)
    else:
        db_type = "Unkown"
    return {
        "name": db.name,
        "type": db_type,
        "interface": db._interfaces,
        "shards": [
            {
                # FIXME: very sloppy, make this comprehension a fn
                **shard.to_dict(),
                "conf_file": shard.cluster_conf_file,
                "out_file": out_file,
                "err_file": err_file,
                "telemetry_metadata": {
                    "status_dir": str(
                        telemetry_data_path / f"database/{db.name}/{dbnode.name}"
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
