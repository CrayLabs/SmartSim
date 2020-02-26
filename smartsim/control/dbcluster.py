import time
from subprocess import Popen, TimeoutExpired, run, SubprocessError, PIPE
from rediscluster import RedisCluster
from rediscluster.exceptions import ClusterDownError

from ..error import SmartSimError
from ..utils import get_logger
logger = get_logger(__name__)


def ping_host(hostname):
    """Ping a specific hostname and return the output

       :param str hostname: hostname of the compute node
    """
    proc = Popen("ping -c 1 " + hostname, stderr=PIPE, stdout=PIPE, shell=True)
    try:
        output, errs = proc.communicate(timeout=15)
        return output.decode("utf-8")
    except TimeoutExpired:
        proc.kill()
        output, errs = proc.communicate()
        logger.error("Communication with database nodes failed")
        raise SmartSimError("Could not ping database nodes for cluster creation")

def get_ip_from_host(host):
    """Return the IP address for the interconnect.

       :param str host: hostname of the compute node e.g. nid00004
    """
    ping_out = ping_host(host)
    found = False

    for item in ping_out.split():
        if found:
            return item.split("(")[1].split(")")[0]
        if item == host:
            found = True

def create_cluster(nodes, port):
    """Create a KeyDB cluster on the specifed nodes at port. This method
       is called using the KeyDB-cli tool and is called after all of the
       database nodes have been launched.

       :param nodes: the nodes the database instances were launched on
       :type nodes: list of strings
       :param int port: port the database nodes were launched on
       :raises: SmartSimError if cluster creation fails
    """
    cluster_str = ""
    for node in nodes:
        node_ip = get_ip_from_host(node)
        node_ip += ":" + str(port)
        cluster_str += node_ip + " "

    # call cluster command
    keydb_cli = "keydb-cli"
    cmd = " ".join((keydb_cli, "--cluster create", cluster_str, "--cluster-replicas 0"))
    proc = run([cmd],
                input="yes",
                encoding="utf-8",
                capture_output=True,
                shell=True)
    out = proc.stdout
    err = proc.stderr

    if proc.returncode != 0:
        logger.error(err)
        raise SmartSimError("KeyDB '--cluster create' command failed")
    else:
        logger.debug(out)
        logger.info("KeyDB Cluster has been created with %s nodes" % str(len(nodes)))

def check_cluster_status(nodes, port):
    """Check the status of the cluster and ensure that all nodes are up and running"""
    node_list = []
    for node in nodes:
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
        raise SmartSimError("Cluster setup could not be verified")
