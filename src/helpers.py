
"""A number of functions used in various places throughout MPO, but belonging to
   no specific class or group."""


import toml
import logging

from subprocess import PIPE, Popen, CalledProcessError
from os import  getcwd


def read_config():
    try:
        cwd = getcwd()
        fname = cwd + "/../mpo-config.toml"
        with open(fname, 'r', encoding='utf-8') as fp:
            parsed_toml = toml.load(fp)
            return parsed_toml
    except Exception as e:
        raise Exception("Could not parse/find mpo-config.toml")




def execute_cmd(cmd_list, wd=getcwd(), err_message=""):
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
