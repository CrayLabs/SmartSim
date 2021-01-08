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
