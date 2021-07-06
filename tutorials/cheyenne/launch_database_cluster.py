#!/usr/bin/env python3
#PBS -N smartsimtest
#PBS  -r n
#PBS  -j oe
#PBS  -V
#PBS  -l walltime=00:10:00
#PBS  -A P93300606
#PBS  -q regular
#PBS  -V
#PBS  -S /bin/bash
#PBS  -l select=4:ncpus=36:mpiprocs=36:ompthreads=1:nodetype=largemem

import os
import socket
import numpy as np

from smartsim import Experiment, constants
from smartsim.database import PBSOrchestrator

from smartredis import Client


"""
Launch a distributed, in memory database cluster and use the
SmartRedis python client to send and recieve some numpy arrays.

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
        with open(os.path.basename(node_file), "w") as f:
            for line in hosts[num_hosts:]:
                f.write(line)
                print("host is {}".format(line))
        os.environ["PBS_NODEFILE"] = os.path.basename(node_file)
    else:
        raise Exception("PBS_NODEFILE {} had {} hosts, not {}".format(node_file, len(hosts),num_hosts))


def launch_cluster_orc(exp, db_hosts, port):
    """Just spin up a database cluster, check the status
       and tear it down"""

    print(f"Starting Orchestrator on hosts: {db_hosts}")
    # batch = False to launch on existing allocation
    db = PBSOrchestrator(port=port, db_nodes=3, batch=False,
                          run_command="mpirun", hosts=db_hosts)

    # generate directories for output files
    # pass in objects to make dirs for
    exp.generate(db, overwrite=True)

    # start the database on interactive allocation
    exp.start(db, block=True)

    # get the status of the database
    statuses = exp.get_status(db)
    print(f"Status of all database nodes: {statuses}")

    return db

print("before PBS_NODEFILE is {}".format(os.getenv("PBS_NODEFILE")))

# create the experiment and specify PBS because cheyenne is a PBS system
exp = Experiment("launch_cluster_db", launcher="pbs")

db_port = 6780
db_hosts = collect_db_hosts(3)
# start the database
db = launch_cluster_orc(exp, db_hosts, db_port)


## test sending some arrays to the database cluster
## the following functions are largely the same across all the
## client languages: C++, C, Fortran, Python
#
## only need one address of one shard of DB to connect client
db_address = ":".join((socket.gethostbyname(db_hosts[0]), str(db_port)))
print("db_address is {}".format(db_address))
print("after PBS_NODEFILE is {}".format(os.getenv("PBS_NODEFILE")))
#client = Client(address=db_address, cluster=True)
#
## put into database
#test_array = np.array([1,2,3,4])
#print(f"Array put in database: {test_array}")
#client.put_tensor("test", test_array)
#
## get from database
#returned_array = client.get_tensor("test")
#print(f"Array retrieved from database: {returned_array}")
#
## shutdown the database because we don't need it anymore
exp.stop(db)
