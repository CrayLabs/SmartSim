

class Junction:
    """A Junction holds multiple registered connections. The connections are made
       through incrementing a database counter on a single database instance. Each
       time a new connection is registered, something happens
       TODO: write this better
    """

    def __init__(self):
        self._database_id = 0
        self.senders = {}  # senders[entity_name] = database_ids
        self.recievers = {} # recievers[entity_name] = database_ids
        self.records = {} # records[sender] = reciever

    def increment_db_id(self):
        self._database_id += 1

    def store_db_addr(self, addr, port):
        self.db_addr = addr
        self.db_port = str(port)

    def register(self, sender, reciever):
        """register a connection from on entity to another"""
        self.records[sender] = reciever
        if reciever in self.recievers:
            self.recievers[reciever].append(self._database_id)
        else:
            self.recievers[reciever] = [self._database_id]
        if sender in self.senders:
            self.senders[sender].append(self._database_id)
        else:
            self.senders[sender] = [self._database_id]
        self.increment_db_id()


    def get_connections(self, entity):
        """Collects all the connections and formats them into a dictionary of
           {'SSDB' : '127.0.0.1:6379',
            'SSDATAIN' : 1 2 3,
            'SSDATAOUT : 4 5 6
           }
            where the number indicates with database partition to communicate over
        """
        data_in, data_out = self._get_connections(entity)
        connections = {}
        def get_env_str(database_list):
            if database_list:
                env_str = ""
                for conn in database_list:
                    env_str += str(conn) + " "
                return env_str
            else:
                return ""
        connections["SSDATAIN"] = get_env_str(data_in)
        connections["SSDATAOUT"] = get_env_str(data_out)
        connections["SSDB"] = ":".join((self.db_addr, self.db_port))
        return connections

    def _get_connections(self, entity):
        """get the connections for a specific entity, returning None if there
           are no entities to connect or send to"""

        def get_connection(entity, conn_dict):
            if entity in conn_dict.keys():
                return conn_dict[entity]
            else:
                return None

        data_in = get_connection(entity, self.recievers)
        data_out = get_connection(entity, self.senders)
        return data_in, data_out


    def __str__(self):
        junction_str = "\n   Connections \n"
        for sender, reciever in self.records.items():
            junction_str += " ".join(("    ", sender, " => ", reciever, "\n"))
        junction_str += "\n"
        return junction_str