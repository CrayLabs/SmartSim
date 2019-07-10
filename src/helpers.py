
"""A number of functions used in various places throughout MPO, but belonging to
   no specific class or group."""


import toml
import logging

from subprocess import PIPE, Popen, CalledProcessError
from os import getcwd, environ


def read_config():
    try:
        cwd = getcwd()
        fname = cwd + "/../ss-config.toml"
        with open(fname, 'r', encoding='utf-8') as fp:
            parsed_toml = toml.load(fp)
            return parsed_toml
    except Exception as e:
        raise Exception("Could not parse/find ss-config.toml")

def get_SSHOME():
    """Retrieves SMARTSIMHOME env variable"""
    try:
        SS_HOME = environ["SMARTSIMHOME"]
        return SS_HOME
    except KeyError:
        raise Exception("Smart Sim library not setup")

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
