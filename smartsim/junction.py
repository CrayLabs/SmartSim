

class Junction:
    """A Junction holds multiple registered connections. The connections are made
       through incrementing a database counter on a single database instance. Each
       time a new connection is registered, something happens
       TODO: write this better
    """

    def __init__(self):
        self.senders = {}  # senders[entity_name] = receivers
        self.recievers = {} # recievers[entity_name] = senders

    def store_db_addr(self, addr, port):
        self.db_addr = addr
        self.db_port = str(port)

    def register(self, sender, reciever):
        """register a connection from on entity to another"""

        if reciever in self.recievers:
            self.recievers[reciever].append(sender)
        else:
            self.recievers[reciever] = [sender]

        if sender in self.senders:
            self.senders[sender].append(reciever)
        else:
            self.senders[sender] = [reciever]


    def get_connections(self, entity):
        """Collects all the connections and formats them into a dictionary of
           {'SSDB' : '127.0.0.1:6379',
            'SSDATAIN' : sim_one;sim_two
            'SSDATAOUT : node_one
           }
        """
        data_in = self._get_connections(entity)
        connections = {}
        def get_env_str(database_list):
            if database_list:
                env_str = ""
                for i, conn in enumerate(database_list):
                    if i == len(database_list)-1:
                        env_str += str(conn)
                    else:
                        env_str += str(conn) + ":"
                return env_str.strip()
            else:
                return ""
        connections["SSDATAIN"] = get_env_str(data_in)
        connections["SSDB"] = ":".join((self.db_addr, self.db_port))
        connections["SSNAME"] = entity
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
        return data_in


    def __str__(self):
        junction_str = "\n   Connections \n"
        for sender, recievers in self.senders.items():
            recieve_str = ", ".join(recievers)
            junction_str += " ".join(("    ", sender, " => ", recieve_str, "\n"))
        junction_str += "\n"
        return junction_str