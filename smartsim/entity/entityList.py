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

if t.TYPE_CHECKING:
    # pylint: disable=unused-import
    import smartsim


class EntityList:
    """Abstract class for containers for SmartSimEntities"""

    def __init__(self, name: str, path: str, **kwargs: t.Any) -> None:
        self.name: str = name
        self.path: str = path
        self.entities: t.List["smartsim.entity.SmartSimEntity"] = []
        self._db_models: t.List["smartsim.entity.DBModel"] = []
        self._db_scripts: t.List["smartsim.entity.DBScript"] = []
        self._initialize_entities(**kwargs)

    def _initialize_entities(self, **kwargs: t.Any) -> None:
        """Initialize the SmartSimEntity objects in the container"""
        raise NotImplementedError

    @property
    def db_models(self) -> t.Iterable["smartsim.entity.DBModel"]:
        """Return an immutable collection of attached models"""
        return (model for model in self._db_models)

    @property
    def db_scripts(self) -> t.Iterable["smartsim.entity.DBScript"]:
        """Return an immutable collection of attached scripts"""
        return (script for script in self._db_scripts)

    @property
    def batch(self) -> bool:
        try:
            if not hasattr(self, "batch_settings"):
                return False

            if self.batch_settings:
                return True
            return False
        # local orchestrator cannot launch with batches
        except AttributeError:
            return False

    @property
    def type(self) -> str:
        """Return the name of the class"""
        return type(self).__name__

    def set_path(self, new_path: str) -> None:
        self.path = new_path
        for entity in self.entities:
            entity.path = new_path

    def __getitem__(self, name: str) -> t.Optional["smartsim.entity.SmartSimEntity"]:
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None

    def __iter__(self) -> t.Iterator["smartsim.entity.SmartSimEntity"]:
        for entity in self.entities:
            yield entity

    def __len__(self) -> int:
        return len(self.entities)
