import subprocess, os, io

def _convert_to_fd(filearg, from_dir, mode="a"):
    filearg = _get_path(filearg, from_dir)

    return open(filearg, mode)

_hack=object()

def run_cmd(cmd, input_str=None, from_dir=None, verbose=None,
            arg_stdout=_hack, arg_stderr=_hack, env=None,
            combine_output=False, timeout=None, executable=None):
    """
    Wrapper around subprocess to make it much more convenient to run shell commands

    >>> run_cmd('ls file_i_hope_doesnt_exist')[0] != 0
    True
    """

    # Real defaults for these value should be subprocess.PIPE
    if arg_stdout is _hack:
        arg_stdout = subprocess.PIPE
    elif isinstance(arg_stdout, str):
        arg_stdout = _convert_to_fd(arg_stdout, from_dir)

    if arg_stderr is _hack:
        arg_stderr = subprocess.STDOUT if combine_output else subprocess.PIPE
    elif isinstance(arg_stderr, str):
        arg_stderr = _convert_to_fd(arg_stdout, from_dir)

    if verbose:
        print("RUN: {}\nFROM: {}".format(cmd, os.getcwd() if from_dir is None else from_dir))

    if (input_str is not None):
        stdin = subprocess.PIPE
    else:
        stdin = None

    proc = subprocess.Popen(cmd,
                            shell=True,
                            stdout=arg_stdout,
                            stderr=arg_stderr,
                            stdin=stdin,
                            cwd=from_dir,
                            executable=executable,
                            env=env)

    output, errput = proc.communicate(input_str)

    # In Python3, subprocess.communicate returns bytes. We want to work with strings
    # as much as possible, so we convert bytes to string (which is unicode in py3) via
    # decode. 
    if output is not None:
        try:
            output = output.decode('utf-8', errors='ignore')
        except AttributeError:
            pass
    if errput is not None:
        try:
            errput = errput.decode('utf-8', errors='ignore')
        except AttributeError:
            pass

    # Always strip outputs
    if output:
        output = output.strip()
    if errput:
        errput = errput.strip()

    stat = proc.wait()
    if isinstance(arg_stdout, io.IOBase):
        arg_stdout.close() # pylint: disable=no-member
    if isinstance(arg_stderr, io.IOBase) and arg_stderr is not arg_stdout:
        arg_stderr.close() # pylint: disable=no-member


    if verbose:
        if stat != 0:
            print("  stat: {:d}\n".format(stat))
        if output:
            print("  output: {}\n".format(output))
        if errput:
            print("  errput: {}\n".format(errput))

    return stat, output, errput
