

<div align="center">
    <a href="https://github.com/CrayLabs/SmartSim"><img src="https://raw.githubusercontent.com/CrayLabs/SmartSim/master/doc/images/SmartSim_Large.png" width="90%"><img></a>
    <br />
    <br />
<div display="inline-block">
    <a href="https://github.com/CrayLabs/SmartSim"><b>Home</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.craylabs.org/docs/installation_instructions/basic.html"><b>Install</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://www.craylabs.org/docs/overview.html"><b>Documentation</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://github.com/CrayLabs"><b>Cray Labs</b></a>&nbsp;&nbsp;&nbsp;
    <a href="mailto:craylabs@hpe.com"><b>Contact</b></a>&nbsp;&nbsp;&nbsp;
    <a href="https://join.slack.com/t/craylabs/shared_invite/zt-nw3ag5z5-5PS4tIXBfufu1bIvvr71UA"><b>Join us on Slack!</b></a>&nbsp;&nbsp;&nbsp;
  </div>
    <br />
    <br />
</div>


<div align="center">

[![License](https://img.shields.io/github/license/CrayLabs/SmartSim)](https://github.com/CrayLabs/SmartSim/blob/master/LICENSE.md)
![GitHub last commit](https://img.shields.io/github/last-commit/CrayLabs/SmartSim)
![GitHub deployments](https://img.shields.io/github/deployments/CrayLabs/SmartSim/github-pages?label=doc%20build)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/smartsim)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/smartsim)
![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/CrayLabs/SmartSim)
![Language](https://img.shields.io/github/languages/top/CrayLabs/SmartSim)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/CrayLabs/SmartSim/branch/develop/graph/badge.svg?token=96HFI2F45E)](https://codecov.io/gh/CrayLabs/SmartSim)
[![Downloads](https://static.pepy.tech/personalized-badge/smartsim?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/smartsim)

</div>

------------

# SmartSim

SmartSim is made up of two parts
  1. SmartSim Infrastructure Library (This repository)
  2. [SmartRedis](https://github.com/CrayLabs/SmartRedis)

The two library components are designed to work together, but can also be used
independently.

*SmartSim* is a workflow library that makes it easier to use common Machine Learning (ML)
libraries, like PyTorch and TensorFlow, in High Performance Computing (HPC) simulations
and applications. SmartSim launches ML infrastructure on HPC systems alongside user
workloads.

*SmartRedis* provides an API to connect HPC workloads, particularly (MPI + X) simulations,
to the ML infrastructure, namely the The [Orchestrator database](https://www.craylabs.org/docs/orchestrator.html),
launched by SmartSim.

Applications integrated with the SmartRedis clients, written in Fortran, C, C++ and Python,
can send data to and remotely request SmartSim infrastructure to execute ML models and scripts
on GPU or CPU. The distributed Client-Server paradigm allows for data to be seamlessly
exchanged between applications at runtime without the utilization of MPI.

----------

**Table of Contents**
- [SmartSim](#smartsim)
- [Quick Start](#quick-start)
- [SmartSim Infrastructure Library](#smartsim-infrastructure-library)
  - [Experiments](#experiments)
    - [Hello World](#hello-world)
    - [Hello World MPI](#hello-world-mpi)
  - [Experiments on HPC Systems](#experiments-on-hpc-systems)
    - [Interactive Launch Example](#interactive-launch-example)
    - [Batch Launch Examples](#batch-launch-examples)
- [Infrastructure Library Applications](#infrastructure-library-applications)
  - [Redis + RedisAI](#redis--redisai)
    - [Local Launch](#local-launch)
    - [Interactive Launch](#interactive-launch)
    - [Batch Launch](#batch-launch)
- [SmartRedis](#smartredis)
  - [Tensors](#tensors)
  - [Datasets](#datasets)
- [SmartSim + SmartRedis Tutorials](#smartsim--smartredis-tutorials)
  - [Run the Tutorials](#run-the-tutorials)
  - [Online Analysis](#online-analysis)
      - [Lattice Boltzmann Simulation](#lattice-boltzmann-simulation)
  - [Online Processing](#online-processing)
    - [Singular Value Decomposition](#singular-value-decomposition)
  - [Online Inference](#online-inference)
    - [PyTorch CNN Example](#pytorch-cnn-example)
- [Publications](#publications)
- [Cite](#cite)
  - [bibtex](#bibtex)

----

# Quick Start


The documentation has a number of tutorials that make it easy to get used to SmartSim locally
before using it on your system. Each tutorial is a Jupyter notebook that can be run through the
[SmartSim Tutorials docker image](https://github.com/orgs/CrayLabs/packages?repo_name=SmartSim)
which will run a jupyter lab with the tutorials, SmartSim, and SmartRedis installed.

```bash
docker pull ghcr.io/craylabs/smartsim-tutorials:latest
docker run -p 8888:8888 ghcr.io/craylabs/smartsim-tutorials:latest
# click on link to open jupyter lab
```

# SmartSim Infrastructure Library

The Infrastructure Library (IL), the ``smartsim`` python package,
facilitates the launch of Machine Learning and simulation
workflows. The Python interface of the IL creates, configures, launches and monitors
applications.

## Experiments

The [Experiment](https://www.craylabs.org/docs/api/smartsim_api.html#experiment) object
is the main interface of SmartSim. Through the [Experiment](https://www.craylabs.org/docs/api/smartsim_api.html#experiment)
users can create references to user applications called ``Models``.
### Hello World

Below is a simple example of a workflow that uses the IL to launch hello world
program using the local launcher which is designed for laptops and single nodes.

```python
from smartsim import Experiment

exp = Experiment("simple", launcher="local")

settings = exp.create_run_settings("echo", exe_args="Hello World")
model = exp.create_model("hello_world", settings)

exp.start(model, block=True)
print(exp.get_status(model))
```

### Hello World MPI

The [Experiment.create_run_settings](https://www.craylabs.org/docs/api/smartsim_api.html#smartsim.experiment.Experiment.create_run_settings) method returns a ``RunSettings`` object which
defines how a model is launched. There are many types of ``RunSettings`` [supported by
SmartSim](https://www.craylabs.org/docs/api/smartsim_api.html#settings).

 - ``RunSettings``
 - ``MpirunSettings``
 - ``SrunSettings``
 - ``AprunSettings``
 - ``JsrunSettings``

The following example launches a hello world MPI program using the local launcher
for single compute node, workstations and laptops.

```Python
from smartsim import Experiment

exp = Experiment("hello_world", launcher="local")
mpi_settings = exp.create_run_settings(exe="echo",
                                       exe_args="Hello World!",
                                       run_command="mpirun")
mpi_settings.set_tasks(4)

mpi_model = exp.create_model("hello_world", mpi_settings)

exp.start(mpi_model, block=True)
print(exp.get_status(model))
```

If an argument of `run_command="auto"` (the default) is passed to
`Experiment.create_run_settings`, SmartSim will attempt to find a run command on the
system with which it has a corresponding `RunSettings` class. If one can be found,
`Experiment.create_run_settings` will instance and return an object of that type.


-----------
## Experiments on HPC Systems

SmartSim integrates with common HPC schedulers providing batch and interactive
launch capabilities for all applications:

 - Slurm
 - LSF
 - PBSPro
 - Local (for laptops/single node, no batch)

In addition, on Slurm and PBS systems, [Dragon](https://dragonhpc.github.io/dragon/doc/_build/html/index.html)
can be used as a launcher. Please refer to the documentation for instructions on
how to insall it on your system and use it in SmartSim.


### Interactive Launch Example

The following launches the same ``hello_world`` model in an interactive allocation.

```bash
# get interactive allocation (Slurm)
salloc -N 3 --ntasks-per-node=20 --ntasks 60 --exclusive -t 00:10:00

# get interactive allocation (PBS)
qsub -l select=3:ncpus=20 -l walltime=00:10:00 -l place=scatter -I -q <queue>

# get interactive allocation (LSF)
bsub -Is -W 00:10 -nnodes 3 -P <project> $SHELL
```

This same script will run on a SLURM, PBS, or LSF system as the ``launcher``
is set to `auto` in the [Experiment](https://www.craylabs.org/docs/api/smartsim_api.html#experiment)
initialization. The run command like ``mpirun``,
``aprun`` or ``srun`` will be automatically detected from what is available on the
system.

```python
# hello_world.py
from smartsim import Experiment

exp = Experiment("hello_world_exp", launcher="auto")
run = exp.create_run_settings(exe="echo", exe_args="Hello World!")
run.set_tasks(60)
run.set_tasks_per_node(20)

model = exp.create_model("hello_world", run)
exp.start(model, block=True, summary=True)

print(exp.get_status(model))
```
```bash
# in interactive terminal
python hello_world.py
```


This script could also be launched in a batch file instead of an
interactive terminal. For example, for Slurm:

```bash
#!/bin/bash
#SBATCH --exclusive
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:10:00

python /path/to/hello_world.py
```
```bash
# on Slurm system
sbatch run_hello_world.sh
```


### Batch Launch Examples

SmartSim can also launch workloads in a batch directly from Python, without the need
for a batch script. Users can launch groups of ``Model`` instances in a ``Ensemble``.

The following launches 4 replicas of the the same ``hello_world`` model.

```python
# hello_ensemble.py
from smartsim import Experiment

exp = Experiment("hello_world_batch", launcher="auto")

# define resources for all ensemble members
batch = exp.create_batch_settings(nodes=4, time="00:10:00", account="12345-Cray")
batch.set_queue("premium")

# define how each member should run
run = exp.create_run_settings(exe="echo", exe_args="Hello World!")
run.set_tasks(60)
run.set_tasks_per_node(20)

ensemble = exp.create_ensemble("hello_world",
                               batch_settings=batch,
                               run_settings=run,
                               replicas=4)
exp.start(ensemble, block=True, summary=True)

print(exp.get_status(ensemble))
```

```bash
python hello_ensemble.py
```

Similar to the interactive example, this same script will run on a SLURM, PBS,
or LSF system as the ``launcher`` is set to `auto` in the
[Experiment](https://www.craylabs.org/docs/api/smartsim_api.html#experiment)
initialization. Local launching does not support batch workloads.


--------

# Infrastructure Library Applications
 - Orchestrator - In-memory data store and Machine Learning Inference (Redis + RedisAI)

## Redis + RedisAI

The ``Orchestrator`` is an in-memory database that utilizes Redis and RedisAI to provide
a distributed database and access to ML runtimes from Fortran, C, C++ and Python.

SmartSim provides classes that make it simple to launch the database in many
configurations and optionally form a distributed database cluster. The examples
below will show how to launch the database. Later in this document we will show
how to use the database to perform ML inference and processing.


### Local Launch

The following script launches a single database using the local launcher.

[Experiment.create_database](https://www.craylabs.org/docs/api/smartsim_api.html#smartsim.experiment.Experiment.create_database)
will initialize an ``Orchestrator`` instance corresponding to the specified launcher.

```python
# run_db_local.py
from smartsim import Experiment

exp = Experiment("local-db", launcher="local")
db = exp.create_database(port=6780,       # database port
                         interface="lo")  # network interface to use

# by default, SmartSim never blocks execution after the database is launched.
exp.start(db)

# launch models, analysis, training, inference sessions, etc
# that communicate with the database using the SmartRedis clients

# stop the database
exp.stop(db)
```

### Interactive Launch

The ``Orchestrator``, like ``Ensemble`` instances, can be launched locally, in interactive
allocations, or in a batch.

The following example launches a distributed (3 node) database cluster
in an interactive allocation.


```bash
# get interactive allocation (Slurm)
salloc -N 3 --ntasks-per-node=1 --exclusive -t 00:10:00

# get interactive allocation (PBS)
qsub -l select=3:ncpus=1 -l walltime=00:10:00 -l place=scatter -I -q queue

# get interactive allocation (LSF)
bsub -Is -W 00:10 -nnodes 3 -P project $SHELL

```

```python
# run_db.py
from smartsim import Experiment

# auto specified to work across launcher types
exp = Experiment("db-on-slurm", launcher="auto")
db_cluster = exp.create_database(db_nodes=3,
                                 db_port=6780,
                                 batch=False,
                                 interface="ipogif0")
exp.start(db_cluster)

print(f"Orchestrator launched on nodes: {db_cluster.hosts}")
# launch models, analysis, training, inference sessions, etc
# that communicate with the database using the SmartRedis clients

exp.stop(db_cluster)
```
```bash
# in interactive terminal
python run_db.py
```

### Batch Launch

The ``Orchestrator`` can also be launched in a batch without the need for an interactive allocation.
SmartSim will create the batch file, submit it to the batch system, and then wait for the database
to be launched. Users can hit CTRL-C to cancel the launch if needed.

```Python
# run_db_batch.py
from smartsim import Experiment

exp = Experiment("batch-db-on-pbs", launcher="auto")
db_cluster = exp.create_database(db_nodes=3,
                                 db_port=6780,
                                 batch=True,
                                 time="00:10:00",
                                 interface="ib0",
                                 account="12345-Cray",
                                 queue="cl40")

exp.start(db_cluster)

print(f"Orchestrator launched on nodes: {db_cluster.hosts}")
# launch models, analysis, training, inference sessions, etc
# that communicate with the database using the SmartRedis clients

exp.stop(db_cluster)
```

```bash
python run_db_batch.py
```

------
# SmartRedis

The SmartSim IL Clients ([SmartRedis](https://github.com/CrayLabs/SmartRedis))
are implementations of Redis clients that implement the RedisAI
API with additions specific to scientific workflows.

SmartRedis clients are available in Fortran, C, C++, and Python.
Users can seamlessly pull and push data from the Orchestrator from different languages.

## Tensors

Tensors are the fundamental data structure for the SmartRedis clients. The Clients
use the native array format of the language. For example, in Python, a tensor is
a NumPy array while the C/C++ clients accept nested and contiguous arrays.

When stored in the database, all tensors are stored in the same format. Hence,
any language can receive a tensor from the database no matter what supported language
the array was sent from. This enables applications in different languages to communicate
numerical data with each other at runtime.

For more information on the tensor data structure, see
[the documentation](https://www.craylabs.org/docs/sr_data_structures.html#tensor)

## Datasets

Datasets are collections of Tensors and associated metadata. The ``Dataset`` class
is a user space object that can be created, added to, sent to, and retrieved from
the Orchestrator.

For an example of how to use the ``Dataset`` class, see the [Online Analysis example](#online-analysis)

For more information on the API, see the
[API documentation](https://www.craylabs.org/docs/sr_data_structures.html#dataset)

# SmartSim + SmartRedis Tutorials

SmartSim and SmartRedis were designed to work together. When launched through
SmartSim, applications using the SmartRedis clients are directly connected to
any Orchestrator launched in the same [Experiment](https://www.craylabs.org/docs/api/smartsim_api.html#experiment).

In this way, a SmartSim [Experiment](https://www.craylabs.org/docs/api/smartsim_api.html#experiment) becomes a driver for coupled ML and Simulation
workflows. The following are simple examples of how to use SmartSim and SmartRedis
together.

## Run the Tutorials

Each tutorial is a Jupyter notebook that can be run through the
[SmartSim Tutorials docker image](https://github.com/orgs/CrayLabs/packages?repo_name=SmartSim)
which will run a jupyter lab with the tutorials, SmartSim, and SmartRedis installed.

```bash
docker pull ghcr.io/craylabs/smartsim-tutorials:latest
docker run -p 8888:8888 ghcr.io/craylabs/smartsim-tutorials:latest
```
Each of the following examples can be found in the
[SmartSim documentation](https://www.craylabs.org/docs/tutorials/getting_started/getting_started.html).

## Online Analysis

Using SmartSim, HPC applications can be monitored in real time by streaming data
from the application to the database. SmartRedis clients can retrieve the
data, process, analyze it, and finally store any updated data back to the database for
use by other clients.

The following is an example of how a user could monitor and analyze a simulation.
The example here uses the Python client; however, SmartRedis clients are also available
for C, C++, and Fortran. All SmartRedis clients implement the same API.

The example will produce [this visualization](https://user-images.githubusercontent.com/13009163/127622717-2c9e4cfd-50f4-4d94-88c4-8c05fa2fa616.mp4) while the simulation is running.

#### Lattice Boltzmann Simulation

Using a [Lattice Boltzmann Simulation](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods),
this example demonstrates how to use the SmartRedis ``Dataset`` API to stream
data over the Orchestrator deployed by SmartSim.

The simulation will be composed of two parts: `fv_sim.py` which will generate data from
the Simulation and store it in the Orchestrator, and `driver.py`
which will launch the Orchestrator, start `fv_sim.py` and check for data posted to the
Orchestrator to plot updates in real-time.

The following code highlights the sections of `fv_sim.py` that are responsible for
transmitting the data needed to plot timesteps of the simulation to the Orchestrator.

```Python
# fv_sim.py
from smartredis import Client
import numpy as np

# initialization code omitted

# save cylinder location to database
cylinder = (X - x_res/4)**2 + (Y - y_res/2)**2 < (y_res/4)**2 # bool array
client.put_tensor("cylinder", cylinder.astype(np.int8))

for time_step in range(steps): # simulation loop
    for i, cx, cy in zip(idxs, cxs, cys):
        F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
        F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

    bndryF = F[cylinder,:]
    bndryF = bndryF[:,[0,5,6,7,8,1,2,3,4]]

    rho = np.sum(F, 2)
    ux  = np.sum(F * cxs, 2) / rho
    uy  = np.sum(F * cys, 2) / rho

    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(idxs, cxs, cys, weights):
        Feq[:,:,i] = rho * w * ( 1 + 3*(cx*ux+cy*uy)  + 9*(cx*ux+cy*uy)**2/2 - 3*(ux**2+uy**2)/2 )
    F += -(1.0/tau) * (F - Feq)
    F[cylinder,:] = bndryF

    # Create a SmartRedis dataset with vorticity data
    dataset = Dataset(f"data_{str(time_step)}")
    dataset.add_tensor("ux", ux)
    dataset.add_tensor("uy", uy)

    # Put Dataset in db at key "data_{time_step}"
    client.put_dataset(dataset)
```

The driver script, `driver.py`, launches the Orchestrator database and runs
the simulation in a non-blocking fashion. The driver script then uses the SmartRedis
client to pull the DataSet and plot the vorticity while the simulation is running.

```Python
# driver.py
time_steps, seed = 3000, 42

exp = Experiment("finite_volume_simulation", launcher="local")

db = exp.create_database(port=6780,        # database port
                         interface="lo")   # network interface db should listen on

# create the lb simulation Model reference
settings = exp.create_run_settings("python",
                                   exe_args=["fv_sim.py",
                                             f"--seed={seed}",
                                             f"--steps={time_steps}"])
model = exp.create_model("fv_simulation", settings)
model.attach_generator_files(to_copy="fv_sim.py")
exp.generate(db, model, overwrite=True)

exp.start(db)
client = Client(address=db.get_address()[0], cluster=False)

# start simulation (non-blocking)
exp.start(model, block=False, summary=True)

# poll until simulation starts and then retrieve data
client.poll_key("cylinder", 200, 100)
cylinder = client.get_tensor("cylinder").astype(bool)

for i in range(0, time_steps):
    client.poll_key(f"data_{str(i)}", 10, 1000)
    dataset = client.get_dataset(f"data_{str(i)}")
    ux, uy = dataset.get_tensor("ux"), dataset.get_tensor("uy")

    # analysis/plotting code omitted

exp.stop(db)
```

For more examples of how to use SmartSim and SmartRedis together to perform
online analysis, please see the
[online analsysis tutorial section](https://www.craylabs.org/docs/tutorials/online_analysis/lattice/online_analysis.html) of the
SmartSim documentation.

## Online Processing

Each of the SmartRedis clients can be used to remotely execute
[TorchScript](https://pytorch.org/docs/stable/jit.html) code on data
stored within the database. The scripts/functions are executed in the Torch
runtime linked into the database.

Any of the functions available in the
[TorchScript builtins](https://pytorch.org/docs/stable/jit_builtin_functions.html#builtin-functions)
can be saved as "script" or "functions" in the database and used directly by
any of the SmartRedis Clients.

### Singular Value Decomposition

For example, the following code sends the built-in
[Singular Value Decomposition](https://pytorch.org/docs/stable/generated/torch.svd.html)
to the database and execute it on a dummy tensor.

```python
import numpy as np
from smartredis import Client

# don't even need to import torch
def calc_svd(input_tensor):
    return input_tensor.svd()


# connect a client to the database
client = Client(cluster=False)

# get dummy data
tensor = np.random.randint(0, 100, size=(5, 3, 2)).astype(np.float32)

client.put_tensor("input", tensor)
client.set_function("svd", calc_svd)

client.run_script("svd", "calc_svd", "input", ["U", "S", "V"])
# results are not retrieved immediately in case they need
# to be fed to another function/model

U = client.get_tensor("U")
S = client.get_tensor("S")
V = client.get_tensor("V")
print(f"U: {U}, S: {S}, V: {V}")
```

The processing capabilities make it simple to form computational pipelines of
functions, scripts, and models.

See the full [TorchScript Language Reference](https://pytorch.org/docs/stable/jit.html#torchscript-language)
documentation for more information on available methods, functions, and how
to create your own.

## Online Inference

SmartSim supports the following frameworks for querying Machine Learning models
from C, C++, Fortran and Python with the SmartRedis Clients:

<table>
  <thead>
    <tr>
      <th style="text-align:center">RedisAI Version</th>
      <th style="text-align:center">Libraries</th>
      <th style="text-align:center">Supported Version</th>
    </tr>
  </thead>
  <tbody style="text-align:center">
    <tr>
      <td rowspan="3">1.2.7</td>
      <td>PyTorch</td>
      <td>2.1.0</td>
    </tr>
    <tr>
      <td>TensorFlow\Keras</td>
      <td>2.15.0</td>
    </tr>
    <tr>
      <td>ONNX</td>
      <td>1.14.1</td>
    </tr>
  </tbody>
</table>

A [number of other libraries](https://github.com/onnx/onnxmltools) are
supported through ONNX, like [SciKit-Learn](https://github.com/onnx/sklearn-onnx/)
and [XGBoost](https://github.com/onnx/onnxmltools/tree/master/tests/xgboost).

**Note:** It's important to remember that SmartSim utilizes a client-server model. To run
experiments that utilize the above frameworks, you must first start the Orchestrator
database with SmartSim.

### PyTorch CNN Example

The example below shows how to spin up a database with SmartSim and
invoke a PyTorch CNN model using the SmartRedis clients.

```python
# simple_torch_inference.py
import io
import torch
import torch.nn as nn
from smartredis import Client
from smartsim import Experiment

exp = Experiment("simple-online-inference", launcher="local")
db = exp.create_database(port=6780, interface="lo")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3)

    def forward(self, x):
        return self.conv(x)

torch_model = Net()
example_forward_input = torch.rand(1, 1, 3, 3)
module = torch.jit.trace(torch_model, example_forward_input)
model_buffer = io.BytesIO()
torch.jit.save(module, model_buffer)

exp.start(db, summary=True)

address = db.get_address()[0]
client = Client(address=address, cluster=False)

client.put_tensor("input", example_forward_input.numpy())
client.set_model("cnn", model_buffer.getvalue(), "TORCH", device="CPU")
client.run_model("cnn", inputs=["input"], outputs=["output"])
output = client.get_tensor("output")
print(f"Prediction: {output}")

exp.stop(db)
```

The above python code can be run like any normal python script:
```bash
python simple_torch_inference.py
```

For more examples of how to use SmartSim and SmartRedis together to perform
online inference, please see the
[online inference tutorials section](https://www.craylabs.org/docs/tutorials/ml_inference/Inference-in-SmartSim.html) of the
SmartSim documentation.

--------

# Publications

The following are public presentations or publications using SmartSim

 - [Collaboration with NCAR - CGD Seminar](https://www.youtube.com/watch?v=2e-5j427AS0)
 - [SmartSim: Using Machine Learning in HPC Simulations](https://arxiv.org/abs/2104.09355)
 - [SmartSim: Online Analytics and Machine Learning for HPC Simulations](https://www.youtube.com/watch?v=JsSgq-fq44w&list=PLuQQBBQFfpgq0OvjKbjcYgTDzDxTqtwua&index=11)
 - [PyTorch Ecosystem Day Poster](https://assets.pytorch.org/pted2021/posters/J8.png)


--------
# Cite

Please use the following citation when referencing SmartSim, SmartRedis, or any SmartSim related work:

Partee et al., “Using Machine Learning at Scale in HPC Simulations with SmartSim:
An Application to Ocean Climate Modeling”, Journal of Computational Science, Volume 62, 2022, 101707, ISSN 1877-7503

Available: https://doi.org/10.1016/j.jocs.2022.101707.

## bibtex


```latex
@article{PARTEE2022101707,
    title = {Using Machine Learning at scale in numerical simulations with SmartSim: An application to ocean climate modeling},
    journal = {Journal of Computational Science},
    volume = {62},
    pages = {101707},
    year = {2022},
    issn = {1877-7503},
    doi = {https://doi.org/10.1016/j.jocs.2022.101707},
    url = {https://www.sciencedirect.com/science/article/pii/S1877750322001065},
    author = {Sam Partee and Matthew Ellis and Alessandro Rigazzi and Andrew E. Shao and Scott Bachman and Gustavo Marques and Benjamin Robbins},
    keywords = {Deep learning, Numerical simulation, Climate modeling, High performance computing, SmartSim},
}
```
