
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




def execute_cmd(cmd_list, wd=getcwd(),  err_message="", progress=False):
    logging.info("Executing shell command: %s" % " ".join(cmd_list))
    proc = Popen(cmd_list, cwd=wd, stdout=PIPE, stderr=PIPE)
    try:
        if progress:
            while True:
                nextline = proc.stdout.readline().decode("utf-8")
                if "statistics" in nextline:
                    break
                parse_for_progress("Day", nextline)
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

def parse_for_progress(leader, textfile):
    show = False
    for t in textfile.split():
        if show == True:
            print_progress_bar(float(t), 31, prefix="     Day " + str(float(t)))
            show = False
        if t == leader:
            show = True


def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
