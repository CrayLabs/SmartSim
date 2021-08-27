import os
import numpy as np

from smartsim import Experiment
from smartsim.database import LSFOrchestrator

from smartredis import Client


"""
Launch a distributed, in memory database cluster and use the
SmartRedis python client to send and receive some numpy arrays.

This example runs in an interactive allocation with at least three
nodes and 1 processor per node.

i.e. bsub -Is -W 01:00 -J SmartSim-int -nnodes 3 -P <project> -alloc_flags smt1 $SHELL
"""

def launch_cluster_orc(experiment, port):
    """Just spin up a database cluster, check the status
       and tear it down"""

    # batch = False to launch on existing allocation

    db = LSFOrchestrator(port=port,
                        db_per_host=2,
                        db_nodes=6,
                        batch=False,
                        cpus_per_shard=21,
                        gpus_per_shard=3,
                        interface="ib0")


    # generate directories for output files
    # pass in objects to make dirs for
    experiment.generate(db, overwrite=True)

    # start the database on interactive allocation
    experiment.start(db, block=True)

    # get the status of the database
    statuses = experiment.get_status(db)
    print(f"Status of all database nodes: {statuses}")

    return db

# create the experiment and specify LSF because Summit is a LSF system
exp = Experiment("launch_cluster_db", launcher="lsf")

db_port = 6780
# start the database
db = launch_cluster_orc(exp, db_port)


# test sending some arrays to the database cluster
# the following functions are largely the same across all the
# client languages: C++, C, Fortran, Python

# only need one address of one shard of DB to connect client
db_address = ":".join((db._hosts[0], str(db_port)))
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


