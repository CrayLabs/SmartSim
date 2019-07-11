from .launcher import Launcher
from helpers import execute_cmd
import logging
from os import getcwd

class SlurmLauncher(Launcher):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Saved separately so that run is aware of alterations to the default partition
    self.alloc_partition = None

  # Validate SLURM on the system and that requested resources are available
  def validate(self, nodes=None, ppn=None, partition=None):
    """Validate slurm on the system and check for sufficient resources using 'sinfo'

    :param nodes: Override the default node count to validate
    :param ppn: Override the default processes per node to validate
    :param partition: Override the default partition to validate
    :param verbose: Set verbosity
    :return:
    """
    if nodes is None:
      nodes = self.def_nodes
    if ppn is None:
      ppn = self.def_ppn
    if partition is None:
      partition = self.def_partition

    sinfo_cmd = ["sinfo", "--noheader", "--format", "%R %D %c"]
    err_mess = "Failed to validate slurm!"

    out_str, err_str = execute_cmd(sinfo_cmd, err_message=err_mess)

    # Parse the output string to extract the available resources for the specified partition
    # Example output from sinfo --noheader --format %R %D %c:
    # bdw18 32 72
    # bdw10 9 40
    # npl24 2 96
    # npl32 6 128
    # npl24V20 2 48
    # ...
    tot_nodes = 0
    for line in out_str.split("\n"):
      partition_details = line.split(" ")
      if partition is None or partition_details[0] == partition:
        # Strip + character from core count if detected
        if not partition_details[2][-1].isnumeric():
          partition_details[2] = partition_details[2][:-1]
        if int(partition_details[2]) >= ppn:
          tot_nodes += int(partition_details[1])

    logging.debug("Found %d nodes that match the constraints provided" % tot_nodes)

    if tot_nodes < nodes:
      raise SystemError("Could not find enough nodes with the specified constraints: \n"
                        "Nodes Requested: %i\n"
                        "Nodes Detected: %i" % (nodes, tot_nodes))

    logging.info("Successfully Validated Slurm with sufficient resources")

  @staticmethod
  def parse_salloc(out_str):
    for line in out_str.split("\n"):
      if line.startswith("salloc: Granted job allocation"):
        return line.split()[-1]

  def get_alloc_cmd(self, nodes=None, ppn=None, partition=None, add_opts=[], shell=False):

    if nodes is None:
      nodes = self.def_nodes
    if ppn is None:
      ppn = self.def_ppn
    if partition is None:
      partition = self.def_partition
    self.alloc_partition = partition
    ntasks = ppn * nodes

    # Build command:
    if shell:
      salloc = ["salloc", "--nodes", str(nodes), "--ntasks", str(ntasks)] + add_opts
    else:
      salloc = ["salloc", "--no-shell", "--nodes", str(nodes), "--ntasks", str(ntasks)] + add_opts
    if partition is not None:
      salloc += ["--partition", partition]

    return salloc

  def get_alloc(self, nodes=None, ppn=None, partition=None, add_opts=[]):
    """ Get the allocation using salloc and specified constraints.

    :param nodes: Override the default node count to allocate
    :param ppn: Override the default processes per node to allocate
    :param partition: Override the default partition to allocate from
    :param add_opts: Additional options to pass to salloc
    :return:
    """
    salloc = self.get_alloc_cmd(nodes, ppn, partition, add_opts)

    out, err = execute_cmd(salloc)

    self.alloc_id = self.parse_salloc(err)
    if self.alloc_id is not None:
      logging.info("Allocation Successful with Job ID: %s" % self.alloc_id)
    else:
      raise SystemError("Failed to parse Job ID from call to `salloc`")

    return self.alloc_id

  def free_alloc(self, alloc_id=None):
    """ Free the current or provided allocation using scancel

    :param alloc_id: Option to explicitly state the allocation ID to free, otherwise just target the current allocation
    :return:
    """
    if alloc_id is None:
      alloc_id = self.alloc_id

    if alloc_id is None:
      logging.debug("No allocation ID was specified and there is no current allocation being managed.\n"
                    "Exiting early from SlurmLauncher.free_alloc()")
      return

    scancel = ["scancel", alloc_id]
    err_mess = "Unable to revoke your allocation for jobid %s\n"\
               "The job may have already timed out, or you may need to cancel the job manually" % alloc_id

    _, err = execute_cmd(scancel, err_message=err_mess)

    if not err is None:
      logging.debug("Unable to revoke your allocation for jobid %s" % alloc_id)
      logging.debug("The job may have already timed out, or you may need to cancel the job manually")
      logging.debug(err)

    logging.info("Successfully Freed Allocation %s" % alloc_id)
    self.alloc_id = None

  def run(self, cmd=[], cwd=getcwd(), nodes=None, ppn=None, add_opts=[]):
    """Run a command using srun

    :param cmd: The command to be run in the form of a list of strings
    :param nodes: Override the default node count for the run
    :param ppn: Override the default processes per node for the run
    :param add_opts: Additional options to be passed on to srun
    :return:
    """
    if nodes is None:
      nodes = self.def_nodes
    if ppn is None:
      ppn = self.def_ppn

    if self.alloc_id is None:
      self.get_alloc()

    ntasks = ppn * nodes

    srun = ["srun", "--jobid", self.alloc_id,
            "--nodes", str(nodes),
            "--ntasks", str(ntasks)]

    if len(add_opts) > 0:
      srun.append(add_opts)
    if self.alloc_partition is not None:
      srun += ["--partition", self.alloc_partition]

    srun += cmd

    out, err = execute_cmd(srun, wd=cwd)
    logging.info(out)
    #logging.info(err)


class PBSLauncher(Launcher):

  def __init__(self):
    super.__init__()
    raise NotImplementedError("PBS Launcher is not yet available!")


class UrikaLauncher(Launcher):
  """Launcher class intended to leverage Urika-XC/Urika-CS to either launch batch jobs with run_training or analytics
  jobs with start_analytics (using the --run-cmd option). The UrikaLauncher is a wrapper to a standard launcher, as the
  Urika packages require an active allocation in order to run as expected.
  """

  def __init__(self, launcher, start_analytics=False, *args, **kwargs):
    """__init__

    :param launcher: A Slurm or PBS based launcher to manage job allocation for the Urika package
    :param start_analytics: bool that switches between start_analytics and run_training as launching mechanisms
    """
    super().__init__(*args, **kwargs)
    self.launcher = launcher
    self.start_analytics = start_analytics

  def validate(self, nodes=None, ppn=None, partition=None, verbose=True):
    """Validate both the underlying workload manager based launcher and the analytics module. The Analytics module will
    be validated by simply running a 'start_analytics --help' call.

    :param nodes: Override the default node count to validate
    :param ppn: Override the default processes per node to validate
    :param partition: Override the default partition to validate
    :param verbose: Set verbosity
    :return:
    """
    # Validate the workload manager launcher
    self.launcher.validate(verbose=False)
    # Ensure analytics module is loaded
    start_analytics_help = ["start_analytics", "--help"]
    err_mess = "start_analytics command failed. Have you loaded the analytics module?"
    execute_cmd(start_analytics_help, err_message=err_mess)
    logging.info("Successfully Validated the Urika module")

  def run(self, cmd=[], nodes=None, ppn=None, partition=None, add_opts=[], urika_opts=[]):
    """Run the provided command with either run_training or start_analytics to leverage the Urika OSA stack

    :param cmd: The command to be run
    :param nodes: Override the number of nodes to run on
    :param ppn: Override the number of processes per node to run on
    :param partition: Override the partition to run on
    :param add_opts: Any additional options to pass into the underlying launcher
    :param urika_opts: Additional options to pass into the urika command (run_training/start_analytics)
    :return:
    """
    # Build run command
    run_cmd = self.launcher.get_alloc_cmd(nodes, ppn, partition, add_opts+self.def_alt_opts, shell=True)
    if self.start_analytics:
      run_cmd += ["start_analytics"] + urika_opts + ["--run-cmd"]
      run_cmd += [" ".join(cmd)]
    else:
      run_cmd = ["run_training"]
      if nodes is not None:
        run_cmd.append("-N")
        run_cmd.append(str(nodes))
      if ppn is not None:
        run_cmd.append("--ppn")
        run_cmd.append(str(ppn))
      run_cmd = run_cmd + urika_opts
    logging.info("Running shell process: %s" % " ".join(run_cmd))
    out, err = execute_cmd(run_cmd)

    for o in out:
      logging.info("StdOut: %s" % o)
    for e in err:
      logging.error("StdErr: %s" % e)
