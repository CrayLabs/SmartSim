from ..util.shell import execute_cmd


def qstat(args):
    """Calls PBS qstat with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of qstat
    """
    cmd = ["qstat"] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def qsub(args):
    """Calls PBS qsub with args

    :param args: List of command arguments
    :type args: List of str
    :returns: Output and error of salloc
    """
    cmd = ["qsub"] + args
    _, out, error = execute_cmd(cmd)
    return out, error


def qdel(args):
    """Calls PBS qdel with args.

    returncode is also supplied in this function.

    :param args: list of command arguments
    :type args: list of str
    :return: output and error
    :rtype: str
    """
    cmd = ["qdel"] + args
    returncode, out, error = execute_cmd(cmd)
    return returncode, out, error
