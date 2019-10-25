
import redis
import time
from os import environ


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

    def get(self, key, wait=False, wait_interval=.5):
        # TODO put in a timeout limit
        data = self.conn.get(key)
        if not data and wait:
            time.sleep(wait_interval)
            return self.get(key, wait=wait)
        else:
            return data

    def send(self, key, value):
        self.conn.set(key, value)
