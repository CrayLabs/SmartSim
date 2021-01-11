from itertools import product

class Junction:
    """A Junction manages all the data endpoints of a SmartSim
    experiment and the database cluster that serves as the
    central hub of communication
    """

    def __init__(self):
        self.database_instances = []

    def store_db_addr(self, addr, port):
        """Register a database instance by its host id and port

        :param addr: Hostname on which the database instance was started
        :type addr:  str
        :param port: Port number of the initialized database instance
        :type port:  int
        """
        if not isinstance(port, list):
            port = [port]
        port = [str(p) for p in port if not isinstance(p, str)]

        for combine in product(addr, port):
            self.database_instances.append(":".join(combine))

    def get_connections(self, entity):
        """Retrieve all connections registered to this entity

        :param entity: The entity to retrieve connections from
        :type entity:  SmartSimEntity
        :returns: Dictionary whose keys are environment variables to be set
        :rtype: dict
        """
        connections = {}
        if self.database_instances:
            connections["SSDB"] = ",".join(self.database_instances)
            if entity.incoming_entities:
                connections["SSKEYIN"] = ",".join([in_entity.name for in_entity in entity.incoming_entities])
            if entity.query_key_prefixing():
                connections["SSKEYOUT"] = entity.name
        return connections
