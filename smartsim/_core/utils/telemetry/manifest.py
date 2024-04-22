# BSD 2-Clause License
#
# Copyright (c) 2021-2024 Hewlett Packard Enterprise
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
import json
import logging
import pathlib
import time
import typing as t
from dataclasses import dataclass, field

from smartsim._core.control.job import JobEntity

logger = logging.getLogger("TelemetryMonitor")


@dataclass
class Run:
    """
    A Run contains the collection of entities created when a `SmartSim`
    driver script executes `Experiment.start`"""

    timestamp: int
    """the timestamp at the time the `Experiment.start` is called"""
    models: t.List[JobEntity]
    """models started in this run"""
    orchestrators: t.List[JobEntity]
    """orchestrators started in this run"""
    ensembles: t.List[JobEntity]
    """ensembles started in this run"""

    def flatten(
        self, filter_fn: t.Optional[t.Callable[[JobEntity], bool]] = None
    ) -> t.Sequence[JobEntity]:
        """Flatten all `JobEntity`'s in the `Run` into a 1-dimensional list

        :param filter_fn: optional boolean filter that returns
        True for entities to include in the result
        """
        entities = self.models + self.orchestrators + self.ensembles
        if filter_fn:
            entities = [entity for entity in entities if filter_fn(entity)]
        return entities

    @staticmethod
    def load_entity(
        entity_type: str,
        entity_dict: t.Dict[str, t.Any],
        exp_dir: pathlib.Path,
    ) -> t.List[JobEntity]:
        """Map entity data persisted in a manifest file to an object

        :param entity_type: type of the associated `SmartSimEntity`
        :param entity_dict: raw dictionary deserialized from manifest JSON
        :param exp_dir: root path to experiment outputs
        :return: list of loaded `JobEntity` instances
        """
        entities = []

        # an entity w/parent keys must create entities for the items that it
        # comprises. traverse the children and create each entity
        parent_keys = {"shards", "models"}
        parent_keys = parent_keys.intersection(entity_dict.keys())
        if parent_keys:
            container = "shards" if "shards" in parent_keys else "models"
            child_type = "orchestrator" if container == "shards" else "model"
            for child_entity in entity_dict[container]:
                entity = JobEntity.from_manifest(child_type, child_entity, str(exp_dir))
                entities.append(entity)

            return entities

        # not a parent type, just create the entity w/the entity_type passed in
        entity = JobEntity.from_manifest(entity_type, entity_dict, str(exp_dir))
        entities.append(entity)
        return entities

    @staticmethod
    def load_entities(
        entity_type: str,
        run: t.Dict[str, t.Any],
        exp_dir: pathlib.Path,
    ) -> t.Dict[str, t.List[JobEntity]]:
        """Map a collection of entity data persisted in a manifest file to an object

        :param entity_type: type of the associated `SmartSimEntity`
        :param run: raw dictionary containing `Run` data deserialized from JSON
        :param exp_dir: root path to experiment outputs
        :return: list of loaded `JobEntity` instances
        """
        persisted: t.Dict[str, t.List[JobEntity]] = {
            "model": [],
            "orchestrator": [],
        }
        for item in run[entity_type]:
            entities = Run.load_entity(entity_type, item, exp_dir)
            for new_entity in entities:
                persisted[new_entity.type].append(new_entity)

        return persisted

    @staticmethod
    def load_run(raw_run: t.Dict[str, t.Any], exp_dir: pathlib.Path) -> "Run":
        """Map run data persisted in a manifest file to an object

        :param runs: raw dictionary containing `Run` data deserialized from JSON
        :param exp_dir: root path to experiment outputs
        :return: populated `Run` instance
        """

        # create an output mapping to hold the deserialized entities
        run_entities: t.Dict[str, t.List[JobEntity]] = {
            "model": [],
            "orchestrator": [],
            "ensemble": [],
        }

        # use the output mapping keys to load all the target
        # entities from the deserialized JSON
        for entity_type in run_entities:
            _entities = Run.load_entities(entity_type, raw_run, exp_dir)

            # load_entities may return a mapping containing types different from
            # entity_type IF it was a parent entity. Iterate through the keys in
            # the output dictionary and put them in the right place
            for entity_type, new_entities in _entities.items():
                if not new_entities:
                    continue
                run_entities[entity_type].extend(new_entities)

        loaded_run = Run(
            raw_run["timestamp"],
            run_entities["model"],
            run_entities["orchestrator"],
            run_entities["ensemble"],
        )
        return loaded_run


@dataclass
class RuntimeManifest:
    """The runtime manifest holds information about the entities created
    at runtime during a SmartSim Experiment. The runtime manifest differs
    from a standard manifest - it may contain multiple experiment
    executions in a `runs` collection and holds information that is unknown
    at design-time, such as IP addresses of host machines.
    """

    name: str
    """The name of the `Experiment` associated to the `RuntimeManifest`"""
    path: pathlib.Path
    """The path to the `Experiment` working directory"""
    launcher: str
    """The launcher type used by the `Experiment`"""
    runs: t.List[Run] = field(default_factory=list)
    """A `List` of 0 to many `Run` instances"""

    @staticmethod
    def load_manifest(file_path: str) -> t.Optional["RuntimeManifest"]:
        """Load a persisted manifest and return the content

        :param file_path: path to the manifest file to load
        :return: deserialized `RuntimeManifest` if the manifest file is found,
        otherwise None
        """
        manifest_dict: t.Optional[t.Dict[str, t.Any]] = None
        try_count, max_attempts = 1, 5

        # allow multiple read attempts in case the manifest is being
        # written at the time load_manifest is called
        while manifest_dict is None and try_count <= max_attempts:
            source = pathlib.Path(file_path)
            source = source.resolve()
            time.sleep(0.01)  # a tiny sleep avoids reading partially written json

            try:
                if text := source.read_text(encoding="utf-8").strip():
                    manifest_dict = json.loads(text)
            except json.JSONDecodeError as ex:
                print(f"Error loading manifest: {ex}")
                # hack/fix: handle issues reading file before it is fully written
                time.sleep(0.1 * try_count)
            finally:
                try_count += 1

        if not manifest_dict:
            return None

        # if we don't have an experiment, the manifest is malformed
        exp = manifest_dict.get("experiment", None)
        if not exp:
            raise ValueError("Manifest missing required experiment")

        # if we don't have runs, the manifest is malformed
        runs = manifest_dict.get("runs", None)
        if runs is None:
            raise ValueError("Manifest missing required runs")

        exp_dir = pathlib.Path(exp["path"])
        runs = [Run.load_run(raw_run, exp_dir) for raw_run in runs]

        manifest = RuntimeManifest(
            name=exp["name"],
            path=exp_dir,
            launcher=exp["launcher"],
            runs=runs,
        )
        return manifest
