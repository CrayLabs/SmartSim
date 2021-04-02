# BSD 2-Clause License
#
# Copyright (c) 2021, Hewlett Packard Enterprise
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

from ..database import Orchestrator
from ..entity import EntityList, SmartSimEntity
from ..error import SmartSimError


def separate_entities(args):
    """Given multiple entities to launch, separate
    by Class type

    :returns: entities, entity list, and orchestrator
    :rtype: tuple
    """
    _check_names(args)
    entities = []
    entity_lists = []
    db = None

    for arg in args:
        if isinstance(arg, Orchestrator):
            if db:
                raise SmartSimError("Separate_entities was given two orchestrators")
            db = arg
        elif isinstance(arg, SmartSimEntity):
            entities.append(arg)
        elif isinstance(arg, EntityList):
            entity_lists.append(arg)
        else:
            raise TypeError(
                f"Argument was of type {type(arg)}, not SmartSimEntity or EntityList"
            )

    return entities, entity_lists, db


def _check_names(args):
    used = []
    for arg in args:
        name = getattr(arg, "name", None)
        if not name:
            raise TypeError(
                f"Argument was of type {type(arg)}, not SmartSimEntity or EntityList"
            )
        if name in used:
            raise SmartSimError("User provided two entities with the same name")
        used.append(name)
