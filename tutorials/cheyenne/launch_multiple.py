import os

from smartsim import Experiment
from smartsim.database import PBSOrchestrator
from smartsim.settings import MpirunSettings


"""
Launch a distributed, in memory database cluster and a model that
sends data to the database.

This example runs in an interactive allocation with at least three
nodes and 2 processors per node. be sure to include mpirprocs in you
allocation.

i.e. qsub -l select=3:ncpus=2:mpiprocs:2 -l walltime=00:20:00 -A <account> -q premium -I
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
    db_cluster = PBSOrchestrator(port=port,
                                db_nodes=3,
                                batch=False,
                                interface="ib0",
                                run_command="mpirun",
                                hosts=hosts)

    # generate directories for output files
    # pass in objects to make dirs for
    experiment.generate(db_cluster, overwrite=True)

    # start the database on interactive allocation
    experiment.start(db_cluster, block=True)

    # get the status of the database
    statuses = experiment.get_status(db_cluster)
    print(f"Status of all database nodes: {statuses}")

    return db_cluster

def create_producer(experiment):

    mpirun = MpirunSettings(exe="python",
                            exe_args="producer.py")
    mpirun.set_tasks(1)
    producer = experiment.create_model("producer", mpirun)

    # create directories for the output files and copy
    # scripts to execution location inside newly created dir
    # only necessary if its not an executable (python is executable here)
    producer.attach_generator_files(to_copy="./producer.py")
    experiment.generate(producer, overwrite=True)
    return producer

# create the experiment and specify PBS because cheyenne is a PBS system
exp = Experiment("launch_multiple", launcher="pbs")

db_port = 6780
db_hosts = collect_db_hosts(3)
# start the database
db = launch_cluster_orc(exp, db_hosts, db_port)

model = create_producer(exp)
exp.start(model, block=True, summary=True)

# shutdown the database because we don't need it anymore
exp.stop(db)

print(exp.summary())


