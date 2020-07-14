
***********
Experiments
***********

The primary user-facing object within the SmartSim library is the SmartSim Experiment,
encompassing all external entities (e.g. numerical simulation, data analysis tools, or ML packages)
as well as internal SmartSim objects that coordinate the setup and execution of the experiment.
Users configure the experiment in Python, either in a script or Jupyter notebook.
The simulation and analysis entities communicate with each other through an in-memory database
using an API, referred to as a SmartSim client, with support for C, C++, Fortran, and Python.

.. |SmartSim Architecture| image:: images/SmartSim_Architecture.png
  :width: 700
  :alt: Alternative text

|SmartSim Architecture|


Entities
========

Experiments are like configuration files that specify exactly how a set
of entities should be run within SmartSim. Entities are the specific tasks
a user wants to run in their workflow. There are four entities within SmartSim:

1. Model
2. Ensemble
3. Node
4. Orchestrator


Model
-----
SmartSim model objects are created by the user to add simulation
applications into the SmartSim experiment.  The SmartSim model
object contains run settings for the simulation such as the number
of computational nodes for the simulation, the number of processes per
node, the executable, and executable arguments. For more information
on how to configure the runtime settings of a Model or any entity,
see the `launcher documentation <launchers.html>`_

Instances of numerical models or "simulation" models. Models can
be created through a call to ``Experiment.create_model()`` and though
the creation of an Ensemble.

Ensemble
--------
In addition to a single model, SmartSim has the ability to launch an
ensemble(s) of simulations simultaneously. An ensemble can be manually
constructed through API calls or SmartSim can be used to generate an
ensemble of model realizations by copying and modifying input files.
For the latter approach, at run-time, user-defined character tags in
the simulation's configuration files (e.g. Fortran namelists or XML)
are replaced by SmartSim with specific parameter values. Users can
specify ranges for each of these parameters with the ensemble of
realizations run using a number of preset strategies or implementing
a custom strategy. There are multiple ways of generating ensemble members;
see the `generation documentation <generate.html>`_

Node
----
Nodes run processes adjacent to the simulation. Nodes can be used
for anything from analysis, training, inference, etc. Nodes are the
most flexible entity with no requirements on language or framework.
Nodes are commonly used for acting on data being streamed out of a
simulation model through the orchestrator

Orchestrator
------------
The Orchestrator is an in-memory database, clustered or standalone, that
is launched prior to the simulation. The Orchestrator can be used
to store data from another entity in memory during the course of
an experiment. In order to stream data into the orchestrator or
receive data from the orchestrator, one of the SmartSim clients
has to be used within a NumModel or SmartSimNode.

The use of an in-memory, distributed database to store data is one
of the key components of SmartSim that allows for scalable simulation
and analysis workloads. The inclusion of an in-memory database to the
in-transit framework provides data persistence so that the data can
be accessed at any time during or after the SmartSim experiment.
A distributed framework enables the database to be scaled to the needs
of a particular use case, which may exceed the resources of a single
node for even modest simulations.

SmartSim can use KeyDB or Redis for data staging. We default to KeyDB
due to its inherent clustering capability, performance, and
compatibility with the widely-used Redis database APIs.
KeyDB stores data in key-value pairs that can be set, retrieved,
and manipulated using the aforementioned Redis database APIs.
Every instance of KeyDB can handle concurrent database requests
by taking advantage of multiple threads. KeyDB can be used in a cluster
configuration to scale across multiple compute nodes, and with the
assistance of SmartSim, also host multiple shards of the cluster per
compute node.

Some of the additional features provided in the community
version of KeyDB include multi-master, active replicas, rollover,
and database backup to disk, and KeyDB Pro offers additional features
such as persistent FLASH support, MVCC support, and non-blocking queries.



Creating an Experiment
======================

Experiments can be run as a part of a larger python codebase, or as
a standalone tool. When running a SmartSim script, users can call
``experiment.generate()`` which will create a file structure for the
experiment and all entities within the experiment. This helps
label and organize the various outputs from each of the various
entities. For more information on generation see the `generation
documentation <generate.html>`_


Launching an Experiment
=======================

SmartSim supports launching simulations, databases, and analysis packages on
heterogeneous, computational resources with users specifying hardware groups
on which SmartSim entities are launched. On execution, SmartSim will create
the orchestrator (database) and then execute the models and nodes.  The launching of the
SmartSim experiment is non-blocking, and as a result, the user is free to
execute other commands or launch additional experiments in the same Python script.
If the user would like to wait for the experiment to complete, the status of the
SmartSim models and nodes can be monitored with a blocking poll command through the SmartSim API.


Monitoring Experiments
======================


Stopping Experiments
====================
Because the SmartSim experiment uses an in-memory database, the simulation data is
accessible for as long as the system allocation remains active.  However,
if the user would like to stop the experiment, the API includes the ability to stop
all or specified models, nodes, and database.  Similarly, the API allows the user
to release the system allocations(s) requested by SmartSim if that allocation is not
to be reused by follow-on experiments or for additional data analysis.
