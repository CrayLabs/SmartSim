"""
Launcher interface
Taken from CrayAI

Interface for implementing unique launchers on distributed
systems. Examples: slurm, pbs pro, urika-xc, etc
"""

import abc


class Launcher(abc.ABC):

  def __init__(self, def_nodes=1, def_ppn=1, def_alt_opts=[], def_partition=None, def_queue=None):
    """ __init__

    :param def_nodes: Default number of nodes to allocation
    :param def_ppn: Default processes per node
    :param def_alt_opts: Default alternative options for the launcher
    :param def_partition: Default partition to select
    :param def_queue: Default queue
    """
    self.def_nodes = def_nodes
    self.def_ppn = def_ppn
    self.def_alt_opts = def_alt_opts
    self.def_partition = def_partition
    self.def_queue = def_queue
    self.alloc_id = None
    super().__init__()

  @abc.abstractmethod
  def validate(self, nodes=None, ppn=None, partition=None):
    """Validate the functionality of the launcher and availability of resources on the system

    :param nodes: Override the number of nodes to validate
    :param ppn: Override the processes per node to validate
    :param partition: Override the partition to validate
    :param verbose: Define verbosity
    :return:
    """
    pass

  @abc.abstractmethod
  def get_alloc(self, nodes=None, ppn=None, partition=None, add_opts=None):
    """Get an allocation on the current system using the given launcher interface

    :param nodes: Override the number of nodes to allocate
    :param ppn: Override the number of processes per node to allocate
    :param partition: Override the partition to allocation on
    :param add_opts: Additional options to add tot eh allocation command, e.g salloc, qsub, etc
    :return (int): The allocation id
    """
    pass

  @abc.abstractmethod
  def free_alloc(self, alloc_id=None):
    """Free the default allocation or the allocation represented by the provided alloc_id

    :param alloc_id: Int representing an allocation id
    :return:
    """
    pass

  @abc.abstractmethod
  def run(self, cmd=[], nodes=None, ppn=None, add_opts=""):
    """Run a command on the allocation. Fails if allocation was never created.

    :param cmd: Command to run in the form of a list of strings
    :param nodes: Override the number of nodes to run the command on
    :param ppn: Override the number of processes per node to run the command on
    :param add_opts: Any additional options to pass to the run command
    :return:
    """
    pass


