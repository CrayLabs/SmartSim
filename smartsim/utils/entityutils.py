

from ..database import Orchestrator
from ..entity import SmartSimEntity, Ensemble, Model, DBNode, EntityList
from ..error import SmartSimError


def seperate_entities(args):
    """Given multiple entities to launch, seperate
       by Class type

       :returns: entities, entity list, and orchestrator
       :rtype: tuple
    """
    # TODO add entity name detection
    i = 0
    ents = []
    elists = []
    orc = None

    while i < len(args):
        entity = args[i]
        if isinstance(entity, Orchestrator):
            if orc:
                raise SmartSimError("seperate_entities was given two orchestrators")
            orc = entity
            i += 1
        elif isinstance(entity, SmartSimEntity):
            ents.append(entity)
            i += 1
        elif isinstance(entity, EntityList):
            elists.append(entity)
            i += 1
        else:
            raise TypeError(
                f"Argument was of type {type(entity)}, not SmartSimEntity or EntityList")

    return ents, elists, orc
