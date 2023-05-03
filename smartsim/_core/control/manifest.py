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

from ...database import Orchestrator
from ...entity import EntityList, SmartSimEntity
from ...error import SmartSimError
from ..utils.helpers import fmt_dict

# List of types derived from EntityList which require specific behavior
# A corresponding property needs to exist (like db for Orchestrator),
# otherwise they will not be accessible
entity_list_exception_types = [Orchestrator]


class Manifest:
    """This class is used to keep track of all deployables generated by an experiment.
    Different types of deployables (i.e. different `SmartSimEntity`-derived objects
    or `EntityList`-derived objects) can be accessed by using the corresponding accessor.

    Instances of ``Model``, ``Ensemble`` and ``Orchestrator``
    can all be passed as arguments
    """

    def __init__(self, *args):
        self._deployables = list(args)
        self._check_types(self._deployables)
        self._check_names(self._deployables)
        self._check_entity_lists_nonempty()

    @property
    def db(self):
        """Return Orchestrator instances in Manifest

        :raises SmartSimError: if user added to databases to manifest
        :return: orchestrator instances
        :rtype: Orchestrator
        """
        _db = None
        for deployable in self._deployables:
            if isinstance(deployable, Orchestrator):
                if _db:
                    raise SmartSimError(
                        "User attempted to create more than one Orchestrator"
                    )
                _db = deployable
        return _db

    @property
    def models(self):
        """Return Model instances in Manifest

        :return: model instances
        :rtype: List[Model]
        """
        _models = []
        for deployable in self._deployables:
            if isinstance(deployable, SmartSimEntity):
                _models.append(deployable)
        return _models

    @property
    def ensembles(self):
        """Return Ensemble instances in Manifest

        :return: list of ensembles
        :rtype: List[Ensemble]
        """
        _ensembles = []
        for deployable in self._deployables:
            if isinstance(deployable, EntityList):
                is_exceptional_type = False
                for exceptional_type in entity_list_exception_types:
                    if isinstance(deployable, exceptional_type):
                        is_exceptional_type |= True
                if not is_exceptional_type:
                    _ensembles.append(deployable)

        return _ensembles

    @property
    def all_entity_lists(self):
        """All entity lists, including ensembles and
        exceptional ones like Orchestrator

        :return: list of entity lists
        :rtype: List[EntityList]
        """
        _all_entity_lists = self.ensembles
        db = self.db
        if db is not None:
            _all_entity_lists.append(db)

        return _all_entity_lists

    def _check_names(self, deployables):
        used = []
        for deployable in deployables:
            name = getattr(deployable, "name", None)
            if not name:
                raise AttributeError(f"Entity has no name. Please set name attribute.")
            if name in used:
                raise SmartSimError("User provided two entities with the same name")
            used.append(name)

    def _check_types(self, deployables):
        for deployable in deployables:
            if not (
                isinstance(deployable, SmartSimEntity)
                or isinstance(deployable, EntityList)
            ):
                raise TypeError(
                    f"Entity has type {type(deployable)}, not SmartSimEntity or EntityList"
                )

    def _check_entity_lists_nonempty(self):
        """Check deployables for sanity before launching"""

        for entity_list in self.all_entity_lists:
            if len(entity_list) < 1:
                raise ValueError(f"{entity_list.name} is empty. Nothing to launch.")

    def __str__(self):
        s = ""
        e_header = "=== Ensembles ===\n"
        m_header = "=== Models ===\n"
        db_header = "=== Database ===\n"
        if self.ensembles:
            s += e_header

            all_ensembles = self.ensembles
            for ensemble in all_ensembles:
                s += f"{ensemble.name}\n"
                s += f"Members: {len(ensemble)}\n"
                s += f"Batch Launch: {ensemble.batch}\n"
                if ensemble.batch:
                    s += f"{str(ensemble.batch_settings)}\n"
            s += "\n"

        if self.models:
            s += m_header
            for model in self.models:
                s += f"{model.name}\n"
                if model.batch_settings:
                    s += f"{model.batch_settings}\n"
                s += f"{model.run_settings}\n"
                if model.params:
                    s += f"Parameters: \n{fmt_dict(model.params)}\n"
            s += "\n"

        if self.db:
            s += db_header
            s += f"Shards: {self.db.num_shards}\n"
            s += f"Port: {str(self.db.ports[0])}\n"
            s += f"Network: {self.db._interface}\n"
            s += f"Batch Launch: {self.db.batch}\n"
            if self.db.batch:
                s += f"{str(self.db.batch_settings)}\n"

        s += "\n"
        return s

    @property
    def has_db_objects(self):
        """Check if any entity has DBObjects to set"""

        def has_db_models(entity):
            if hasattr(entity, "_db_models"):
                return len(entity._db_models) > 0

        def has_db_scripts(entity):
            if hasattr(entity, "_db_scripts"):
                return len(entity._db_scripts) > 0

        has_db_objects = False
        for model in self.models:
            has_db_objects |= hasattr(model, "_db_models")

        # Check if any model has either a DBModel or a DBScript
        # we update has_db_objects so that as soon as one check
        # returns True, we can exit
        has_db_objects |= any(
            [has_db_models(model) | has_db_scripts(model) for model in self.models]
        )
        if has_db_objects:
            return True

        # If there are no ensembles, there can be no outstanding model
        # to check for DBObjects, return current value of DBObjects, which
        # should be False
        ensembles = self.ensembles
        if not ensembles:
            return has_db_objects

        # First check if there is any ensemble DBObject, if so, return True
        has_db_objects |= any(
            [
                has_db_models(ensemble) | has_db_scripts(ensemble)
                for ensemble in ensembles
            ]
        )
        if has_db_objects:
            return True
        for ensemble in ensembles:
            # Last case, check if any model within an ensemble has DBObjects attached
            has_db_objects |= any(
                [has_db_models(model) | has_db_scripts(model) for model in ensemble]
            )
            if has_db_objects:
                return True

        # `has_db_objects` should be False here
        return has_db_objects
