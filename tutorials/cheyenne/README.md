
# Cheyenne Tutorials

Cheyenne is the supercomputer at the National Center for Atmopheric Research. The
following tutorials are meant to aide users on Cheyenne with getting used to the
different types of workflows that are possible with SmartSim.


## Prerequisites

Since SmartSim does not currently support the Message Passing Toolkit (MPT), Cheyenne
users of SmartSim will need to utilize OpenMPI.

The following module commands were utilized to run the examples

```bash
module purge
module load ncarenv/1.3 gnu/8.3.0 ncarcompilers/0.5.0 netcdf/4.7.4 openmpi/4.0.5
```

With this environment loaded, users will need to build and install both SmartSim and
SmartRedis through pip. Usually we recommend users installing or loading miniconda and
using the pip that comes with that installation.

 1. Activate a Python environment
 2. ``pip install smartsim``
 3. ``smart --device cpu``  (May take a couple minutes)

If you run into trouble with the installation, please consult the installation
documentation [here](https://www.craylabs.org/docs/installation.html)

## Examples

Three of the examples utilize interactive allocations, which is the preferred method of
launching SmartSim.

When utilizing OpenMPI (as opposed to other run commands like ``srun``) to launch the
Orchestrator database, SmartSim needs to be informed of the nodes the user would like
the database to be launched on.

This can be automated, and code for the automation of hostname aquisition is included in
most of the files. This recipe can be followed for launching the Orchestrator with
OpenMPI on PBS systems.


```python
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
```

----------

### 1. launch_distributed_model.py

Launch a distributed model with OpenMPI through SmartSim. This could represent
a simulation or other workload that contains the SmartRedis clients and commuicates
with the Orchestrator.

This example runs in an interactive allocation with at least three
nodes and 20 processors per node. be sure to include mpiprocs in your
allocation.

```bash
# fill in account and queue parameters
qsub -l select=3:ncpus=20:mpiprocs=20 -l walltime=00:20:00 -A <account> -q <queue> -I
```

After obtaining the allocation, make sure to module load your conda or python environment
with SmartSim and SmartRedis installed, as well as module load OpenMPI and gcc 8.3 as
specified at the top of this README.

Compile the simple hello world MPI program.

```bash
mpicc hello.c -o hello
```

Run the model through SmartSim in the interactive allocation

```bash
python launch_distributed_model.py
```

Instead of using an interactive allocation, SmartSim jobs can also be
launched through batch files. This is helpful when waiting a long time
for queued jobs.

The following gives an example of how you could launch the MPI
model above through a batch script instead of an interactive allocation.

```bash
#!/bin/bash

#PBS -l select=3:ncpus=20:mpiprocs=20
#PBS -l walltime=00:10:00
#PBS -A NCGD0048
#PBS -q economy
#PBS -N SmartSim

# activate conda env if needed
python launch_distributed_model.py
```
---------

### 2. launch_database_cluster.py

This file shows how to launch a distributed ``Orchestrator`` (database cluster) and
utilize the SmartRedis Python client to communicate with it. This example is meant
to provide an example of how users can interact with the database in an interactive
fashion, possibly in a medium like a jupyter notebook.

This example runs in an interactive allocation with at least three
nodes and 2 processors per node. be sure to include mpiprocs in your
allocation.

```bash
# fill in account and queue parameters
qsub -l select=3:ncpus=1 -l walltime=00:20:00 -A <account> -q <queue> -I
```
After obtaining the allocation, make sure to module load your conda or python environment
with SmartSim and SmartRedis installed, as well as module load OpenMPI and gcc 8.3 as
specified at the top of this README.

Run the workflow with

```bash
python launch_database_cluster.py
```
----------
### 3. launch_multiple.py

Launch an Orchestrator database in a cluster across three nodes and a data producer
that will put and get data from the Orchestrator using the SmartRedis Python client.

This example shows how a user can take the previous example a step further by
launching the application which communicates with the Orchestrator through SmartSim
as well.

It is important to note in this example that the database and producer are running
a converged workflow - that is, the database and application are placed on the same
nodes. Add a node(s) to the interactive allocation line if you wish for the data
producer to run on a seperate node.

```bash
# fill in account and queue parameters
qsub -l select=3:ncpus=2:mpiprocs:2 -l walltime=00:20:00 -A <account> -q <queue> -I
```
After obtaining the allocation, make sure to module load your conda or python environment
with SmartSim and SmartRedis installed, as well as module load OpenMPI and gcc 8.3 as
specified at the top of this README.

run the workflow with

```bash
python launch_multiple.py
```
-----------
### 4. launch_ensemble_batch.py

Launch a ensemble of hello world models in a batch created by SmartSim. This
file can be launched on a head node and will create a batch file for the all
the jobs to be launched.

The higher level batch capabilities of SmartSim allow users to create many
batch jobs of differing content without needing to write each one. As well,
SmartSim acts as a batch process manager in Python allowing interactivity
with the batch system to create pipelines, dependants, and conditions.

In this case, we create three replicas of the same model through the
``Experiment.create_ensemble()`` function. ``QsubBatchSettings`` are created
to specify resources for the entire batch. ``MpirunSettings`` are created
to specify how each member within the batch should be launched.

Before running the example, be sure to change the ``account`` number in the
file and any other batch settings for submission.

Then, compile the simple hello world MPI program.

```bash
mpicc hello.c -o hello
```

and run the workflow with

```bash
python launch_ensemble_batch.py
```



