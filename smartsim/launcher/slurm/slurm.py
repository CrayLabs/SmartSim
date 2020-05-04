
from ..shell import execute_cmd

def sstat(args):
    """Calls sstat with args
    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sstat
    """

    cmd = ["sstat"] + args
    returncode, out, error = execute_cmd(cmd)
    return out, error

def sacct(args):
    """Calls sacct with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sacct
    """
    cmd = ["sacct"] + args
    returncode, out, error = execute_cmd(cmd)
    return out, error

def salloc(args):
    """Calls slurm salloc with args
    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of salloc
    """

    cmd = ["salloc"] + args
    returncode, out, error = execute_cmd(cmd)
    return out, error

def sinfo(args):
    """Calls slurm sinfo with args
    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of sinfo
    """

    cmd = ["sinfo"] + args
    returncode, out, error = execute_cmd(cmd)
    return out, error

def scancel(args):
    """Calls slurm scancel with args. returncode is
       also supplied in this function.

    :param args: list of command arguments
    :type args: list of str
    :return: output and error
    :rtype: str
    """
    cmd = ["scancel"] + args
    returncode, out, error = execute_cmd(cmd)
    return returncode, out, error

