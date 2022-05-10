import os


def get_hosts():
    hosts = []
    if "PBS_NODEFILE" in os.environ:
        node_file = os.environ["PBS_NODEFILE"]
        with open(node_file, "r") as f:
            for line in f.readlines():
                host = line.split(".")[0]
                hosts.append(host)
        # account for mpiprocs causing repeats in PBS_NODEFILE
        return list(set(hosts))
    raise Exception("Could not parse interactive allocation nodes from PBS_NODEFILE")


def get_queue():
    if "PBS_QUEUE" in os.environ:
        return os.environ.get("PBS_QUEUE")
    raise Exception("Could not parse queue from SLURM_JOB_PARTITION")


def get_tasks():
    if "PBS_NP" in os.environ:
        return os.environ.get("PBS_NP")
    raise Exception("Could not parse number of requested tasks from PBS_NP")


def get_tasks_per_node():
    if "PBS_NUM_PPN" in os.environ:
        return os.environ.get("PBS_NUM_PPN")
    raise Exception("Could not parse tasks per node from PBS_NUM_PPN")
