import os
from subprocess import Popen, PIPE, CalledProcessError
from datetime import datetime

from ..utils import get_logger
logger = get_logger(__name__)


def seq_to_str(seq, to_byte=False, encoding="utf-8", add_equal=False):
    """
    An auxiliary function to convert the commands in the sequence format to string format
    This is necessary based on the shell boolean used when we start a subprocess. the problem
    is the --option=value. Otherwise, a simple " ".join would suffice
    :param seq (string array)
    :param to_byte(bool) whether or not convert to byte stream
    """
    cmd_str = ""
    #if we do not translate --option,arg to --option=arg we only need to join with spaces
    if not add_equal:
        return " ".join(seq)

    for cmd in seq:
        # handling the slurm style of --option=argument format
        #@todo not nice! improve
        if cmd.startswith("--") and cmd != "--no-shell":
                cmd_str += cmd + "="
        else:
                cmd_str += cmd+ " "
    if to_byte:
        return cmd_str.encode(encoding)
    else:
        return cmd_str


def extract_line(output, key):
    """
    an auxiliary function to find a key in a multi-line string
    :returns the first line which contains the key
    """
    for line in output:
        if key in line:
            return line
    return None


def current_time_military(minute_add = 0):
    """
    returns the current time in format hhmm as a string
    """
    t_now = datetime.now()
    hour_int = t_now.hour
    minute_int = t_now.minute
    new_mins = minute_int + minute_add
    minute_int = new_mins % 60
    hour_int += new_mins // 60


    if hour_int < 10:
        hour_str = "0%d" % hour_int
    else:
        hour_str = str(hour_int)
    if minute_int < 10:
        minute_str = "0%d" % minute_int
    else:
        minute_str = str(minute_int)
    return hour_str + minute_str


def write_to_bash(cmd, name):
    with open(name, 'w') as destFile:
        for line in cmd:
            destFile.write("%s\n" % line)

def execute_cmd(cmd_list, err_message="", shell=False, cwd=None, verbose=False):
    """
        This is the function that runs a shell command and returns the output
        It will raise exceptions if the commands did not run successfully
        :param cmd_list: The command to be excuted
        :type cmd_list: List of str, optional str
        :param str err_message: Error message for logger in event of
                non-zero process return code
        :param shell: The shell argument (which defaults to False)
                specifies whether to use the shell as the program to execute.
                If shell is True, it is recommended to pass args
                as a string rather than as a sequence.
        :param cwd: The current working directory
        :type cwd: str
        :param verbose: Boolean for verbose output
        :type verbose: bool
        :raises: CalledProcessError
        :returns: tuple of str for output and error messages
    """
    if verbose:
        logger.info("Executing shell command: %s" % " ".join(cmd_list))
    # spawning the subprocess and connecting to its output
    proc = Popen(cmd_list, stdout=PIPE, stderr=PIPE, shell=shell, cwd=cwd)
    try:
        # waiting for the process to terminate and capture its output
        out, err = proc.communicate()
    except CalledProcessError as e:
        logger.error("Exception while attempting to start a shell process")
        raise e

    if proc.returncode is not 0:
        logger.error("Command \"%s\" returned non-zero" % " ".join(cmd_list))
        logger.error(err.decode('utf-8'))
        logger.error(err_message)
        # raise exception removed: no need to throw an exception here!

    # decoding the output and err and return as a string tuple
    return (out.decode('utf-8'), err.decode('utf-8'))

