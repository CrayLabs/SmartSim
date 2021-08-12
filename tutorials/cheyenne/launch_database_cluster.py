import os
import numpy as np

from smartsim import Experiment
from smartsim.database import PBSOrchestrator

from smartredis import Client


"""
Launch a distributed, in memory database cluster and use the
SmartRedis python client to send and receive some numpy arrays.

This example runs in an interactive allocation with at least three
nodes and 1 processor per node.

i.e. qsub -l select=3:ncpus=1 -l walltime=00:10:00 -A <account> -q premium -I
"""

def collect_db_hosts(num_hosts):
    """A simple method to collect hostnames because we are using
       openmpi. (not needed for aprun(ALPS), Slurm, etc.
    """

    hosts = []
    if "PBS_NODEFILE" in os.environ:
        node_file = os.environ["PBS_NODEFILE"]
        with open(node_file, "r") as f:
            for line in f.readlines():
                host = line.split(".")[0]
                hosts.append(host)
    else:
        raise Exception("could not parse interactive allocation nodes from PBS_NODEFILE")

    # account for mpiprocs causing repeats in PBS_NODEFILE
    hosts = list(set(hosts))

    if len(hosts) >= num_hosts:
        return hosts[:num_hosts]
    else:
        raise Exception(f"PBS_NODEFILE had {len(hosts)} hosts, not {num_hosts}")


def launch_cluster_orc(experiment, hosts, port):
    """Just spin up a database cluster, check the status
       and tear it down"""

    print(f"Starting Orchestrator on hosts: {hosts}")
    # batch = False to launch on existing allocation
    db = PBSOrchestrator(port=port,
                         db_nodes=3,
                         batch=False,
                         interface="ib0",
                         run_command="mpirun",
                         hosts=hosts)

    # generate directories for output files
    # pass in objects to make dirs for
    experiment.generate(db, overwrite=True)

    # start the database on interactive allocation
    experiment.start(db, block=True)

    # get the status of the database
    statuses = experiment.get_status(db)
    print(f"Status of all database nodes: {statuses}")

    return db

# create the experiment and specify PBS because cheyenne is a PBS system
exp = Experiment("launch_cluster_db", launcher="pbs")

db_port = 6780
db_hosts = collect_db_hosts(3)
# start the database
db = launch_cluster_orc(exp, db_hosts, db_port)


# test sending some arrays to the database cluster
# the following functions are largely the same across all the
# client languages: C++, C, Fortran, Python

# only need one address of one shard of DB to connect client
db_address = ":".join((db_hosts[0], str(db_port)))
client = Client(address=db_address, cluster=True)

# put into database
test_array = np.array([1,2,3,4])
print(f"Array put in database: {test_array}")
client.put_tensor("test", test_array)

# get from database
returned_array = client.get_tensor("test")
print(f"Array retrieved from database: {returned_array}")

# shutdown the database because we don't need it anymore
exp.stop(db)


