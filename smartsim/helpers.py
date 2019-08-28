
"""A number of functions used in various places throughout MPO, but belonging to
   no specific class or group."""

import sys
import logging

from subprocess import PIPE, Popen, CalledProcessError
from os import getcwd, environ, path



def get_SSHOME():
    """Retrieves SMARTSIMHOME env variable"""
    try:
        SS_HOME = environ["SMARTSIMHOME"]
        if not SS_HOME.endswith("/"):
            SS_HOME += "/"
        return SS_HOME
    except KeyError:
        print("SmartSim library environment not setup!")
        sys.exit()


def bfs(key, tree):
    """Retrieves a value from a toml file at an unspecified
       depth. Breadth first traversal of toml dict tree.

       Args
         Key (str): key being searched for
         tree (str): A config file to search through

       Returns
         Value associated with key or a KeyError if key cannot
         be found.
        """
    visited = []
    try:
        for k, v in tree.items():
            if k == key:
                return v
            else:
                if isinstance(v, dict):
                    visited.append(v)
            return _bfs(key, visited)
    except KeyError:
        return None

def _bfs(key, visited):
    if len(visited) == 0:
        raise KeyError
    else:
        cur_table = visited.pop()
        for k, v in cur_table.items():
            if k == key:
                return v
            else:
                if isinstance(v, dict):
                    visited.append(v)
            return _bfs(key, visited)



def execute_cmd(cmd_list, wd=getcwd(),  err_message=""):
    logging.info("Executing shell command: %s" % " ".join(cmd_list))
    proc = Popen(cmd_list, cwd=wd, stdout=PIPE, stderr=PIPE)
    try:
        out, err = proc.communicate()
    except CalledProcessError as e:
        logging.error("Exception while attempting to start a shell process")
        for o in out.decode("utf-8").split("\n"):
            print("Stdout: %s" % o)
        for er in err.decode("utf-8").split("\n"):
            print("Stderr: %s" % er)
        raise e

    if proc.returncode is not 0:
        logging.error("Command \"%s\" returned non-zero" % " ".join(cmd_list))
        for o in out.decode("utf-8").split("\n"):
            print("Stdout: %s" % o)
        for er in err.decode("utf-8").split("\n"):
            print("Stderr: %s" % er)
        print(err_message)
        raise ChildProcessError("Command (%s) returned non-zero!" % " ".join(cmd_list))

    return out.decode('utf-8'), err.decode('utf-8')


