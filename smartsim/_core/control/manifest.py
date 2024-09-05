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

import itertools
import pathlib
import typing as t
from dataclasses import dataclass, field

from smartsim.entity._mock import Mock

from ...database import FeatureStore
from ...entity import Application, Ensemble, FSNode, SmartSimEntity
from ...error import SmartSimError
from ..config import CONFIG
from ..utils import helpers as _helpers
from ..utils import serialize as _serialize

_T = t.TypeVar("_T")
_U = t.TypeVar("_U")
_AtomicLaunchableT = t.TypeVar("_AtomicLaunchableT", Application, FSNode)

if t.TYPE_CHECKING:
    import os

class Manifest:
    """This class is used to keep track of all deployables generated by an
    experiment.  Different types of deployables (i.e. different
    `SmartSimEntity`-derived objects) can
    be accessed by using the corresponding accessor.

    Instances of ``Application``, ``Ensemble`` and ``FeatureStore``
    can all be passed as arguments
    """

    def __init__(
        self, *args: t.Union[SmartSimEntity]
    ) -> None:
        self._deployables = list(args)
        self._check_types(self._deployables)
        self._check_names(self._deployables)

    @property
    def fss(self) -> t.List[FeatureStore]:
        """Return a list of FeatureStore instances in Manifest

        :raises SmartSimError: if user added to feature stores to manifest
        :return: List of feature store instances
        """
        fss = [item for item in self._deployables if isinstance(item, FeatureStore)]
        return fss

    @property
    def applications(self) -> t.List[Application]:
        """Return Application instances in Manifest

        :return: application instances
        """
        _applications: t.List[Application] = [
            item for item in self._deployables if isinstance(item, Application)
        ]
        return _applications

    @property
    def ensembles(self) -> t.List[Ensemble]:
        """Return Ensemble instances in Manifest

        :return: list of ensembles
        """
        return [e for e in self._deployables if isinstance(e, Ensemble)]

    @property
    def has_deployable(self) -> bool:
        """
        Return True if the manifest contains entities that
        must be physically deployed
        """
        return bool(self._deployables)

    @staticmethod
    def _check_names(deployables: t.List[t.Any]) -> None:
        used = []
        for deployable in deployables:
            name = getattr(deployable, "name", None)
            if not name:
                raise AttributeError("Entity has no name. Please set name attribute.")
            if name in used:
                raise SmartSimError("User provided two entities with the same name")
            used.append(name)

    @staticmethod
    def _check_types(deployables: t.List[t.Any]) -> None:
        for deployable in deployables:
            if not isinstance(deployable, (SmartSimEntity)):
                raise TypeError(
                    f"Entity has type {type(deployable)}, not "
                    + "SmartSimEntity"
                )

    def __str__(self) -> str:
        output = ""
        e_header = "=== Ensembles ===\n"
        a_header = "=== Applications ===\n"
        fs_header = "=== Feature Stores ===\n"
        if self.ensembles:
            output += e_header

            all_ensembles = self.ensembles
            for ensemble in all_ensembles:
                output += f"{ensemble.name}\n"
                output += f"Members: {len(ensemble)}\n"
                output += f"Batch Launch: {ensemble.batch}\n"
                if ensemble.batch:
                    output += f"{str(ensemble.batch_settings)}\n"
            output += "\n"

        if self.applications:
            output += a_header
            for application in self.applications:
                output += f"{application.name}\n"
                if application.batch_settings:
                    output += f"{application.batch_settings}\n"
                output += f"{application.run_settings}\n"
                if application.params:
                    output += f"Parameters: \n{_helpers.fmt_dict(application.params)}\n"
            output += "\n"

        for afs in self.fss:
            output += fs_header
            output += f"Shards: {afs.num_shards}\n"
            output += f"Port: {str(afs.ports[0])}\n"
            output += f"Network: {afs._interfaces}\n"
            output += f"Batch Launch: {afs.batch}\n"
            if afs.batch:
                output += f"{str(afs.batch_settings)}\n"

        output += "\n"
        return output

    @property
    def has_fs_objects(self) -> bool:
        """Check if any entity has FSObjects to set"""
        ents: t.Iterable[t.Union[Application, Ensemble]] = itertools.chain(
            self.applications,
            self.ensembles,
            (member for ens in self.ensembles for member in ens.entities),
        )
        return any(any(ent.fs_models) or any(ent.fs_scripts) for ent in ents)


class _LaunchedManifestMetadata(t.NamedTuple):
    run_id: str
    exp_name: str
    exp_path: str
    launcher_name: str

    @property
    def exp_telemetry_subdirectory(self) -> pathlib.Path:
        return _format_exp_telemetry_path(self.exp_path)

    @property
    def run_telemetry_subdirectory(self) -> pathlib.Path:
        return _format_run_telemetry_path(self.exp_path, self.exp_name, self.run_id)

    @property
    def manifest_file_path(self) -> pathlib.Path:
        return self.exp_telemetry_subdirectory / _serialize.MANIFEST_FILENAME


@dataclass(frozen=True)
class LaunchedManifest(t.Generic[_T]):
    """Immutable manifest mapping launched entities or collections of launched
    entities to other pieces of external data. This is commonly used to map a
    launch-able entity to its constructed ``Step`` instance without assuming
    that ``step.name == job.name``.
    """

    metadata: _LaunchedManifestMetadata
    applications: t.Tuple[t.Tuple[Application, _T], ...]
    ensembles: t.Tuple[t.Tuple[Ensemble, t.Tuple[t.Tuple[Application, _T], ...]], ...]
    featurestores: t.Tuple[
        t.Tuple[FeatureStore, t.Tuple[t.Tuple[FSNode, _T], ...]], ...
    ]

    def map(self, func: t.Callable[[_T], _U]) -> "LaunchedManifest[_U]":
        def _map_entity_data(
            fn: t.Callable[[_T], _U],
            entity_list: t.Sequence[t.Tuple[_AtomicLaunchableT, _T]],
        ) -> t.Tuple[t.Tuple[_AtomicLaunchableT, _U], ...]:
            return tuple((entity, fn(data)) for entity, data in entity_list)

        return LaunchedManifest(
            metadata=self.metadata,
            applications=_map_entity_data(func, self.applications),
            ensembles=tuple(
                (ens, _map_entity_data(func, application_data))
                for ens, application_data in self.ensembles
            ),
            featurestores=tuple(
                (fs_, _map_entity_data(func, node_data))
                for fs_, node_data in self.featurestores
            ),
        )


@dataclass(frozen=True)
class LaunchedManifestBuilder(t.Generic[_T]):
    """A class comprised of mutable collections of SmartSim entities that is
    used to build a ``LaunchedManifest`` while going through the launching
    process.
    """

    exp_name: str
    exp_path: str
    launcher_name: str
    run_id: str = field(default_factory=_helpers.create_short_id_str)

    _applications: t.List[t.Tuple[Application, _T]] = field(
        default_factory=list, init=False
    )
    _ensembles: t.List[t.Tuple[Ensemble, t.Tuple[t.Tuple[Application, _T], ...]]] = (
        field(default_factory=list, init=False)
    )
    _featurestores: t.List[t.Tuple[FeatureStore, t.Tuple[t.Tuple[FSNode, _T], ...]]] = (
        field(default_factory=list, init=False)
    )

    @property
    def exp_telemetry_subdirectory(self) -> pathlib.Path:
        return _format_exp_telemetry_path(self.exp_path)

    @property
    def run_telemetry_subdirectory(self) -> pathlib.Path:
        return _format_run_telemetry_path(self.exp_path, self.exp_name, self.run_id)

    def add_application(self, application: Application, data: _T) -> None:
        self._applications.append((application, data))

    def add_ensemble(self, ens: Ensemble, data: t.Sequence[_T]) -> None:
        self._ensembles.append((ens, self._entities_to_data(ens.entities, data)))

    def add_feature_store(self, fs_: FeatureStore, data: t.Sequence[_T]) -> None:
        self._featurestores.append((fs_, self._entities_to_data(fs_.entities, data)))

    @staticmethod
    def _entities_to_data(
        entities: t.Sequence[_AtomicLaunchableT], data: t.Sequence[_T]
    ) -> t.Tuple[t.Tuple[_AtomicLaunchableT, _T], ...]:
        if not entities:
            raise ValueError("Cannot map data to an empty entity sequence")
        if len(entities) != len(data):
            raise ValueError(
                f"Cannot map data sequence of length {len(data)} to entity "
                f"sequence of length {len(entities)}"
            )
        return tuple(zip(entities, data))

    def finalize(self) -> LaunchedManifest[_T]:
        return LaunchedManifest(
            metadata=_LaunchedManifestMetadata(
                self.run_id,
                self.exp_name,
                self.exp_path,
                self.launcher_name,
            ),
            applications=tuple(self._applications),
            ensembles=tuple(self._ensembles),
            featurestores=tuple(self._featurestores),
        )


def _format_exp_telemetry_path(
    exp_path: t.Union[str, "os.PathLike[str]"]
) -> pathlib.Path:
    return pathlib.Path(exp_path, CONFIG.telemetry_subdir)


def _format_run_telemetry_path(
    exp_path: t.Union[str, "os.PathLike[str]"], exp_name: str, run_id: str
) -> pathlib.Path:
    return _format_exp_telemetry_path(exp_path) / f"{exp_name}/{run_id}"
