
"""A number of functions used in various places throughout MPO, but belonging to
   no specific class or group."""

import sys
import toml
import logging

from subprocess import PIPE, Popen, CalledProcessError
from os import getcwd, environ


def read_config(config_name):
    try:
        fname = get_SSHOME() + config_name
        with open(fname, 'r', encoding='utf-8') as fp:
            parsed_toml = toml.load(fp)
            return parsed_toml
    except Exception as e:
        print("Could not parse/find configuration file: " + config_name)
        sys.exit()


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


def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar

    Args
        iteration (int): current iteration (required)
        total     (int): total iterations (required)
        prefix    (str): prefix string (optional)
        suffix    (str): suffix string (optional)
        decimals  (int): positive number of decimals in percent complete (optional)
        length    (int): character length of bar (optional)
        fill      (str): bar fill character (optional)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()
