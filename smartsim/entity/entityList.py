import abc


class EntityList(abc.ABC):
    """Abstract class for containers for SmartSimEntities"""

    def __init__(self, name, path, **kwargs):
        super().__init__()
        self.name = name
        self.path = path
        self.entities = []
        self._initialize_entities(**kwargs)

    @abc.abstractmethod
    def _initialize_entities(self, **kwargs):
        """Initialize the SmartSimEntity objects in the container"""
        pass

    def __getitem__(self, name):
        for entity in self.entities:
            if entity.name == name:
                return entity

    def __iter__(self):
        for entity in self.entities:
            yield entity

    def __len__(self):
        return len(self.entities)

    def __repr__(self):
        return self.name