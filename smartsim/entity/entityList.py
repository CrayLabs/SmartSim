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

import typing as t

from .entity import SmartSimEntity

if t.TYPE_CHECKING:
    # pylint: disable-next=unused-import
    import smartsim

_T = t.TypeVar("_T", bound=SmartSimEntity)
# Old style pyint from TF 2.6.x does not know about pep484 style ``TypeVar`` names
# pylint: disable-next=invalid-name
_T_co = t.TypeVar("_T_co", bound=SmartSimEntity, covariant=True)


class EntitySequence(t.Generic[_T_co]):
    """Abstract class for containers for SmartSimEntities"""

    def __init__(self, name: str, path: str, **kwargs: t.Any) -> None:
        self.name: str = name
        self.path: str = path

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # WARNING: This class cannot be made truly covariant until the
        #          following properties are made read-only. It is currently
        #          designed for in-house type checking only!!
        #
        # Despite the fact that these properties are type hinted as
        # ``Sequence``s, the underlying types must remain ``list``s as that is
        # what subclasses are expecting when implementing their
        # ``_initialize_entities`` methods.
        #
        # I'm leaving it "as is" for now as to not introduce a potential API
        # break in case any users subclassed the invariant version of this
        # class (``EntityList``), but a "proper" solution would be to turn
        # ``EntitySequence``/``EntityList`` into proper ``abc.ABC``s and have
        # the properties we expect to be initialized represented as abstract
        # properties. An additional benefit of this solution is would be that
        # users could actually initialize their entities in the ``__init__``
        # method, and it would remove the need for the cumbersome and
        # un-type-hint-able ``_initialize_entities`` method by returning all
        # object construction into the class' constructor.
        # ---------------------------------------------------------------------
        #
        self.entities: t.Sequence[_T_co] = []
        self._db_models: t.Sequence["smartsim.entity.DBModel"] = []
        self._db_scripts: t.Sequence["smartsim.entity.DBScript"] = []
        #
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

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

    def __getitem__(self, name: str) -> t.Optional[_T_co]:
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None

    def __iter__(self) -> t.Iterator[_T_co]:
        for entity in self.entities:
            yield entity

    def __len__(self) -> int:
        return len(self.entities)


class EntityList(EntitySequence[_T]):
    """An invariant subclass of an ``EntitySequence`` with mutable containers"""

    def __init__(self, name: str, path: str, **kwargs: t.Any) -> None:
        super().__init__(name, path, **kwargs)
        # Change container types to be invariant ``list``s
        self.entities: t.List[_T] = list(self.entities)
        self._db_models: t.List["smartsim.entity.DBModel"] = list(self._db_models)
        self._db_scripts: t.List["smartsim.entity.DBScript"] = list(self._db_scripts)

    def _initialize_entities(self, **kwargs: t.Any) -> None:
        """Initialize the SmartSimEntity objects in the container"""
        # Need to identically re-define this "abstract method" or pylint
        # complains that we are trying to define a concrete implementation of
        # an abstract class despite the fact that we want this class to also be
        # abstract.  All the more reason to turn both of these classes into
        # ``abc.ABC``s in my opinion.
        raise NotImplementedError
