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
            connections["SSDB"] = _env_safe_string(";".join(self.database_instances))
            if entity.incoming_entities:
                connections["SSKEYIN"] = _env_safe_string(
                    ";".join([in_entity.name for in_entity in entity.incoming_entities])
                )
            if entity.query_key_prefixing():
                connections["SSKEYOUT"] = entity.name
        return connections

    def __str__(self):
        """Return user-readable string form of Junction

        :returns: a user-readable string of Junction
        :rtype: str
        """
        junction_str = "\n   Connections \n"
        for sender, receivers in self.senders.items():
            receive_str = ", ".join(receivers)
            junction_str += " ".join(("    ", sender, " => ", receive_str, "\n"))
        junction_str += "\n"
        return junction_str

    def __repr__(self):
        """Return the representation value of Junction

        :returns: representation value of Junction
        :rtype: dict
        """
        return {"database_instances": self.database_instances}


def _env_safe_string(string):
    """Format a string to safely set as an environment variable

    This function formats a string that can be safely set
    as an environment variable by enclosing it in double quotation
    marks.

    :param string: The string value of an environment variable
    :type string:  str
    :returns string: The original string enclosed in double quotes
    """
    if string[0] != '"':
        string = '"' + string
    if string[-1] != '"':
        string += '"'
    return string
