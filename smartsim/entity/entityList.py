
class EntityList():
    """Abstract class for containers for SmartSimEntities"""

    def __init__(self, name, path, **kwargs):
        self.name = name
        self.path = path
        self.entities = []
        self._initialize_entities(**kwargs)

    def _initialize_entities(self, **kwargs):
        """Initialize the SmartSimEntity objects in the container"""
        raise NotImplementedError


    @property
    def batch(self):
        try:
            if self.batch_settings:
                return True
            else:
                return False
        # local orchestrator cannot launch with batches
        except AttributeError:
            return False

    @property
    def type(self):
        """Return the name of the class
        """
        return type(self).__name__

    def set_path(self, new_path):
        self.path = new_path
        for entity in self.entities:
            entity.path = new_path

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
