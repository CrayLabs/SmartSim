
import redis


class Connection:
    """The class for SmartSimNodes to communicate with the central Redis(KeyDB)
       database to retrieve and send data from clients and other nodes"""
       
    def __init__(self):
        self.conn = None
       
    def connect(self, host, port, db):
        try:    
            self.conn = redis.Redis(host, port, db)
            if not self.connected():
                raise Exception("something didnt work")
        except redis.ConnectionError:
            raise Exception("No database setup")
        
    def connected(self):
        """pings server and returns true if connected"""
        response = self.conn.ping()
        return response
        
    def get(self, key):
        data = self.conn.get(key)
        return data
    
    def send(self, key, value):
        self.conn.set(key, value)
        
