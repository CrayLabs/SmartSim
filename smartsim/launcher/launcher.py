"""
Interface for implementing unique launchers on distributed
systems.
    - Examples: Slurm, pbs pro, Urika-xc, etc

"""
import os
import abc
import time
import atexit
import zmq
import pickle
from subprocess import PIPE, Popen, CalledProcessError, TimeoutExpired, run

from .alloc import AllocManager
from .launcherUtil import seq_to_str
from ..error import LauncherError, SmartSimError

from rediscluster import RedisCluster
from rediscluster.exceptions import ClusterDownError

from ..utils import get_logger, get_env
logger = get_logger(__name__)


class Launcher(abc.ABC):
    cancel_err_mess = "Unable to revoke your allocation for jobid %s\n"
    run_err_mess = "An error occurred while trying to run the command:\n"

    def __init__(self,
                 def_nodes=1,
                 def_ppn=1,
                 def_partition=None,
                 def_queue=None,
                 def_duration="1:00:00",
                 remote=False,
                 cmd_center_addr="127.0.0.1",
                 cmd_center_port="5555"):
        """ :param def_nodes: Default number of nodes to allocation
            :param def_ppn: Default processes per node
            :param def_partition: Default partition to select
            :param def_queue: Default queue
            :param def_duration: is the default walltime HH:MM:SS
            :param bool remote: run commands on a remote server
            :param str cmd_center_addr: tcp address of the command center
            :param str cmd_center_port: port of the command center
        """
        self.def_nodes = def_nodes
        self.def_ppn = def_ppn
        self.def_partition = def_partition
        self.def_queue = def_queue
        self.def_duration = def_duration
        self.remote = remote
        self.cmd_center_addr = cmd_center_addr
        self.cmd_center_port = cmd_center_port
        self.alloc_manager = AllocManager()
        super().__init__()

    #-------------- Abstract Methods --------------
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
    def _get_alloc_cmd(self,
                       nodes,
                       ppn,
                       partition,
                       start_time,
                       duration,
                       add_opts,
                       to_string=False,
                       debug=False):
        """
        A method to translate the requested resources into a proper command for making the reservation
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
    def run_on_alloc(self,
                     cmd,
                     nodes=None,
                     ppn=None,
                     duration="",
                     wait=True,
                     add_opts=None,
                     partition=None,
                     wd=""):
        """
        Runs a command on an allocation.
        The user is expected to have made an allocation before calling this function
        throws badOpsException is the allocation is not made
        For instance the user may have reserved 4 nodes and 10 tasks
        she can submit aprun/mpirun commands and use a parts of her reservation for each command
        """
        pass

    @abc.abstractmethod
    def _get_free_cmd(self, alloc_id):
        """
        returns the workload manager dependant command to release the allocation
        This is used in the free_alloc function which is implemented below.

        :param alloc_id (string): The allocation id to be released.
        """
        pass

    @abc.abstractmethod
    def stop(self, job_id):
        """
        Stops the job with specified job_id.

        :param str job_id: The job indentifier
        """
        pass

    def free_alloc(self, alloc_id):
        """if there is an active researvation, or if alloc_id is specified it gets cancelled"""

        if alloc_id not in self.alloc_manager().keys():
            raise LauncherError("Allocation id, " + str(alloc_id) +
                                " not found.")

        (cancel_cmd, cancel_err_mess) = self._get_free_cmd(alloc_id)
        returncode, _, err = self.execute_cmd(cancel_cmd)

        if returncode != 0:
            logger.info("Unable to revoke your allocation for jobid %s" % alloc_id)
            logger.info(
                "The job may have already timed out, or you may need to cancel the job manually")
            raise LauncherError("Unable to revoke your allocation for jobid %s" % alloc_id)

        self.alloc_manager.remove_alloc(alloc_id)
        logger.info("Successfully freed allocation %s" % alloc_id)

    def ping_host(self, hostname):
        """Ping a specific hostname and return the output

           :param str hostname: hostname of the compute node
        """
        try:
            returncode, out, err = self.execute_cmd(["ping -c 1 " + hostname], shell=True)
            return out
        except LauncherError as e:
            logger.error("Communication with database nodes failed")
            raise SmartSimError("Could not ping database nodes for cluster creation") from e

    def create_cluster(self, nodes, ports):
        """Create a KeyDB cluster on the specifed nodes at port. This method
            is called using the KeyDB-cli tool and is called after all of the
            database nodes have been launched.

            :param nodes: the nodes the database instances were launched on
            :type nodes: list of strings
            :param ports: ports the database nodes were launched on
            :type ports: list of ints
            :raises: SmartSimError if cluster creation fails
        """
        cluster_str = ""
        for node in nodes:
            node_ip = self.get_ip_from_host(node)
            for port in ports:
                full_ip = ":".join((node_ip, str(port) + " "))
                cluster_str += full_ip

        # call cluster command
        smartsimhome = get_env("SMARTSIMHOME")
        keydb_cli = os.path.join(smartsimhome, "third-party/KeyDB/src/keydb-cli")
        cmd = " ".join((keydb_cli, "--cluster create", cluster_str, "--cluster-replicas 0"))
        returncode, out, err = self.execute_cmd_with_input([cmd], "yes")

        if returncode != 0:
            logger.error(err)
            raise LauncherError("KeyDB '--cluster create' command failed")
        else:
            logger.debug(out)
            logger.info("KeyDB Cluster has been created with %s nodes" % str(len(nodes)))

    def get_ip_from_host(self, host):
        """Return the IP address for the interconnect.

        :param str host: hostname of the compute node e.g. nid00004
        :returns: ip of host
        :rtype: str
        """
        ping_out = self.ping_host(host)
        found = False

        for item in ping_out.split():
            if found:
                return item.split("(")[1].split(")")[0]
            if item == host:
                found = True

    @staticmethod
    def check_cluster_status(nodes, ports):
        """Check the status of the cluster and ensure that all nodes are up and running"""
        node_list = []
        for node in nodes:
            for port in ports:
                node_dict = dict()
                node_dict["host"] = node
                node_dict["port"] = port
                node_list.append(node_dict)

        trials = 10
        while trials > 0:
            try:
                redis_tester = RedisCluster(startup_nodes=node_list)
                redis_tester.set("__test__", "__test__")
                redis_tester.delete("__test__")
                break
            except ClusterDownError:
                logger.debug("Caught a cluster down error")
                time.sleep(5)
                trials -= 1
        if trials == 0:
            raise LauncherError("Cluster setup could not be verified")

    def execute_cmd(self, cmd_list, shell=False, cwd=None, verbose=False, is_async=False, timeout=None):
        """
            This is the function that runs a shell command and returns the output
            It will raise exceptions if the commands did not run successfully

            :param cmd_list: The command to be excuted
            :type cmd_list: List of str, optional str
            :param shell: The shell argument (which defaults to False)
                    specifies whether to use the shell as the program to execute.
                    If shell is True, it is recommended to pass args
                    as a string rather than as a sequence.
            :param cwd: The current working directory
            :type cwd: str
            :param verbose: Boolean for verbose output
            :type verbose: bool
            :param bool is_asnyc: run asynchronously
            :returns: tuple of process returncode, output and error messages
                      if is_async is true, return 1 for successful launch and
                      -1 for unsuccessful launch
        """
        if self.remote: # run the process remotely
            return self.execute_remote_cmd(cmd_list, shell=shell, is_async=is_async, cwd=cwd)

        if verbose:
            logger.info("Executing shell command: %s" % " ".join(cmd_list))
        if is_async:
            return self.run_async_command(cmd_list, cwd)

        # spawning the subprocess and connecting to its output
        proc = Popen(cmd_list, stdout=PIPE, stderr=PIPE, shell=shell, cwd=cwd)
        try:
            # waiting for the process to terminate and capture its output
            if timeout:
                out, err = proc.communicate(timeout=timeout)
            else:
                out, err = proc.communicate()
        except TimeoutExpired as e:
            proc.kill()
            output, errs = proc.communicate()
            logger.error("Timeout for command execution exceeded")
            raise LauncherError("Failed to execute command: " + " ".join(cmd_list))
        except CalledProcessError as e:
            logger.error("Exception while attempting to start a shell process")
            raise LauncherError("Failed to execute command: " + " ".join(cmd_list))

        # decoding the output and err and return as a string tuple
        return proc.returncode, out.decode('utf-8'), err.decode('utf-8')

    def execute_cmd_with_input(self, cmd_list, cmd_input, shell=True, encoding="utf-8"):
        """Run a command using the subprocess.run with input. Default to executing with
           the shell and utf-8 encoding.

           :param cmd_list: cmd to run
           :type cmd_list: list of str
           :param str cmd_input: input for the run command after cmd execution
           :param bool shell: toggle run in shell
           :param str encoding: encoding for the parsed process information
           :returns: tuple of process returncode, output and error messages
        """
        if isinstance(cmd_list, str):
            cmd_list = [cmd_list]
        if self.remote:
            return self.execute_remote_cmd(cmd_list, cmd_input=cmd_input, shell=True)
        else:
            proc = run(cmd_list,
                    input=cmd_input,
                    encoding=encoding,
                    capture_output=True,
                    shell=shell)
            return proc.returncode, proc.stdout, proc.stderr

    def execute_remote_cmd(self, cmd_list, cmd_input=None, shell=False, is_async=False, cwd=None):
        """Send a command to a remote server to be run. Message is sent over a zmq
           connection to the server where the command should be run. This connection
           is set up ahead of time when intializing the experiment and launcher.

           :param cmd_list: cmd to run
           :type cmd_list: list of str
           :param bool shell: toggle run in shell
           :param bool is_asnyc: toggle running the command without waiting for output
           :param str cwd: current working directory to run the command in
           :returns: tuple of process returncode, output and error messages
                      if is_async is true, return 1 for successful launch and
                      -1 for unsuccessful launch
        """
        cmd = pickle.dumps((cmd_list, shell, cmd_input, cwd, is_async))
        address = "tcp://" + ":".join((self.cmd_center_addr, str(self.cmd_center_port)))

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.LINGER, 0) # immediately fail if connection fails
        socket.connect(address)
        try:
            socket.send(cmd)

            poller = zmq.Poller()
            poller.register(socket, zmq.POLLIN)
            if poller.poll(10*1000): # wait 10 seconds
                rep = socket.recv()
                if not is_async:
                    returncode, out, err = pickle.loads(rep)
                    return returncode, out, err
                else:
                    status = pickle.loads(rep)
                    return status
            else:
                raise SmartSimError(
                    f"Communication failed with command center at {address}")
        except zmq.error.Again:
            raise SmartSimError(
                    f"Communication failed with command center at {address}")
        finally:
            socket.close()
            context.term()

    def run_async_command(self, cmd, cwd):
        try:
            popen_obj = Popen(cmd, cwd=cwd, shell=True)
        except OSError as err:
            logger.error(err)
            return -1
        return 1
