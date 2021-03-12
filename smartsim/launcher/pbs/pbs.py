import json
from ...error import LauncherError
from .pbsCommands import qstat
from ...utils import get_logger
logger = get_logger(__name__)


def validate( nodes=None, ppn=None, partition=None):
    """Check that there are sufficient resources in the provided PBS
    partitions.

    :param str partition: partition to validate
    :param nodes: Override the default node count to validate
    :type nodes: int
    :param ppn: Override the default processes per node to validate
    :type ppn: int
    :raises: LauncherError
    """
    sys_partitions = _get_system_partition_info()

    n_avail_nodes = 0

    if not nodes:
        nodes = 1
    if not ppn:
        ppn = 1

    p_name = partition
    if p_name is None or p_name == "default":
        p_name = _get_default_partition()

    if p_name not in sys_partitions.keys():
        raise LauncherError("Partition {0} is not found on this system".format(p_name))

    queue = sys_partitions[p_name]

    n_avail_cpus = queue["cpus"]
    if n_avail_cpus == 0:
        logger.info("Max ppn count not specified for partition {}, assuming it is sufficiently large.".format(p_name))

    if n_avail_cpus == 0 or n_avail_cpus >= ppn:
        n_avail_nodes = queue["nodes"]
        if n_avail_nodes == 0:
            logger.info("Max node count not specified for partition {}, assuming it is sufficiently large.".format(p_name))
            logger.info("Successfully validated PBS with sufficient resources")
        elif n_avail_nodes >= nodes:
            logger.debug("Found {0} nodes that match the constraints provided".format(n_avail_nodes))
            logger.info("Successfully validated PBS with sufficient resources")
        else:
            raise LauncherError("{0} nodes are not available on the specified partitions.  Only "\
                                "{1} nodes available.".format(nodes,n_avail_nodes))
    else:
        raise LauncherError("{0} ppn are not available on the specified partitions.  Only "\
                            "{1} ppn available.".format(ppn,n_avail_cpus))



def _get_system_partition_info():
    """Build a dictionary of PBS queues
        :returns: dict of queue dictionaries
        :rtype: dict
    """

    qstat_output, _ = qstat(["-Qf", "-F", "json"])
    qstat_json = json.loads(qstat_output)

    queues = qstat_json["Queue"]

    partitions = {}
    for p_name, p_info in queues.items():

        partition = {"name": p_name}
        p_node = 0
        p_ppn = 0

        if "resources_max" in p_info:
            res_max = p_info["resources_max"]
            if "ncpus" in res_max:
                p_ppn = int(res_max["ncpus"])
            if "nodect" in res_max:
                p_node = int(res_max["nodect"])

        partition["nodes"] = p_node
        partition["cpus"] = p_ppn
        partitions[p_name] = partition


    return partitions

def _get_default_partition():
    """Returns the default partition from slurm which

    This default partition is assumed to be the partition named Default
    (or something similar), otherwise, the first listed partition is
    chosen.
    :returns: the name of the default partition
    :rtype: str
    """
    qstat_output, _ = qstat(["-Qf", "-F", "json"])
    qstat_json = json.loads(qstat_output)
    queues = qstat_json["Queue"]

    if len(queues) == 0:
        raise LauncherError("Could not find any PBS queue!")
    default = next(iter(queues))

    for queue in queues.keys():
        if queue.lower() == "default":
            return queue

    return default
