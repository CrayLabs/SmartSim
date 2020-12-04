import socket
from datetime import datetime

from ..utils import get_logger
logger = get_logger(__name__)


class ComputeNode():
    """The ComputeNode class holds resource information
    about a physical compute node
    """
    def __init__(self, node_name=None, node_ppn=None):
        """Initialize a ComputeNode

        :param node_name: the name of the node
        :type node_name: str
        :param node_ppn: the number of ppn
        :type node_ppn: int
        """
        self.name = node_name
        self.ppn = node_ppn

    def _is_valid_node(self):
        """Check if the node is complete

        Currently, validity is judged by name
        and ppn being not None.

        :returns: True if valid, false otherwise
        :rtype: bool
        """
        if self.name is None:
            return False
        if self.ppn is None:
            return False

        return True

class Partition():
    """The partition class holds information about
    a system partition.
    """
    def __init__(self):
        """Initialize a system partition
        """
        self.name = None
        self.min_ppn = None
        self.nodes = set()

    def _is_valid_partition(self):
        """Check if the partition is valid

        Currently, validity is judged by name
        and each ComputeNode being valid

        :returns: True if valid, false otherwise
        :rtype: bool
        """
        if self.name is None:
            return False
        if len(self.nodes)<=0:
            return False
        for node in self.nodes:
            if not node._is_valid_node():
                return False

        return True


def get_ip_from_host(host):
    """Return the IP address for the interconnect.

    :param str host: hostname of the compute node e.g. nid00004
    :returns: ip of host
    :rtype: str
    """
    ip_address = socket.gethostbyname(host)
    return ip_address


def seq_to_str(seq, to_byte=False, encoding="utf-8", add_equal=False):
    """Convert a sequence to a string

    An auxiliary function to convert the commands in the sequence
    format to string format.   This is necessary based on the
    shell boolean used when we start a subprocess. the problem
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
    """Find a key in a multiline string

    An auxiliary function to find a key in a multi-line string

    :returns the first line which contains the key
    :rtype: str or None
    """
    for line in output:
        if key in line:
            return line
    return None


def current_time_military(minute_add = 0):
    """Convert current time to military time

    :returns: the current time in format hhmm as a string
    :rtype: str
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

