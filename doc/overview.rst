

************
Introduction
************

SmartSim is a Cray/HPE library that enables the convergence of large scale simulations and AI
workloads on heterogeneous architectures. SmartSim enables a data science first approach to
simulation research by overcoming systemic impediments including lack of inter-operability
between programming languages and dependence on file I/O. SmartSim shares data between
C, C++, Fortran, and Python clients using an in-memory, distributed database cluster.

.. |SmartSim Architecture| image:: images/SmartSim_Architecture.png
  :width: 700
  :alt: Alternative text

|SmartSim Architecture|


SmartSim manages hardware allocations, orchestrates simulation and
analytics workloads, and automates model ensembling. The creation of the in-memory database,
the management of HPC system resource allocations, and management of application input and output
is all automated through the SmartSim API allowing users to launch asynchronous, interconnected
jobs within a single Python script.

Prior to SmartSim, operating on simulation model data was a slow and iterative
process. Users were often required to write large simulation outputs to file
and perform operations like analysis and visualization after the simulation
was finished. This manual process has been replaced in SmartSim with a
data pipeline that allows users to conduct analysis, visualization, ML
training and much more, while the simulation is running. The flexibility
of our data model makes it so that any simulation can write output
that works with a number of language ecosystems enabling the use of
tools and libraries that were not easily used before.

With SmartSim, users can stream data out of their models and operate
directly on the data at discrete time intervals decided by the user. The
framework is flexible and can handle largely any process from
distributed machine learning to more typical data anaylsis and visualization.



Using SmartSim
==============

There are a number of ways SmartSim can be used, but the two mediums
that SmartSim was intended for are:

  1. Jupyter Environment
  2. Python script/shell

Jupyter Enivronment
-------------------

Jupyter notebooks have become the medium of choice for many data scientists
and Python programmers because they allow for self-documentation and quick
iteration. The latter is crucial for data analysis, data processing, and
prototyping of ML/DL models. SmartSim was designed with Jupyter in mind because
of these reasons. SmartSim allows for the simulation expert to never leave the
comfort of the Jupyter environment. Users can configure, run, and analyize
simulation data all within a Jupyter notebook. One can
easily document and share Jupyter notebooks with other SmartSim users
so that all model parameters, run settings (e.g. number of processors) and
configuration for any analytical or machine learning tool is documented
and reproducible. The latter point is of paramount importance when conducting
machine learning experiments for simulations that may be used for
critical applications like weather forecasting.


SmartSim Scripts
----------------

SmartSim scripts are meant to read like a configuration file for an experiment
using a simulation model. The configuration for each model and tool used in the
experiment should be captured in the script (e.g. number of compute nodes for the
simulation) such that reproduction of experiments is as easy as sharing a
python script.

The infrastructure library was designed specifically with reproducibility in mind.
A common problem for domain experts that utilize simulation models is knowing
the exact parameters for the model and the workload manager given a certain
machine, and model configuration. Similar to the Jupyter environment, having
all of the configurations in one script allows scientists to share their
work without having to provide multiple documents describing how to run
their various models and tools.



Library Design
==============

There are two components of the SmartSim library:

  1. Infrastructure Library
  2. Client libraries


Infrastructure Library
======================

The infrastructure library (IL) makes up the bulk of SmartSim's codebase. The IL
provides interfaces to automate the process of deploying the infrastructure
necessary to run complex experiments. This includes interaction with the file
system, workload manager, system level APIs, external libraries, and
simulation models.

An important responsibility of the IL is to interact with the workload manager
of a system. This includes but is not limited to: obtaining, tracking and
releasing allocations; launching, stopping and monitoring jobs; and creating
communication channels between jobs.

The primary reason for implementing the IL is that many workflows in the HPC
space require complex configuration that reduces reproducibility and
productivity. The abtractions of the IL provide a simple mechanism in Python
for workflow configuration which enables reproducibility accross users,
machines, and sites as well as decreases time to solution.


Clients
=======

Traditional workflows on an HPC system lack flexibility and lock applications
into using only packages that are compatible with the software stack being
used. We recognized this when conducting research on how to connect simulation
models to tools and libraries more commonly used in the space of Data Science.
A user with a traditional MPI workload was forced to either make their application
work with MPI, configure inflexible, complex RDMA communication, or even worse,
write their output to the file system.

We are not the first library in the HPC space to aide with these inflexible
application coupling and communication paradigms. Libraries like SENSEI and
GLEAN, and DataSpaces offer useful methods to mitigate this inflexibility
especially for large scale visualization. These libraries, however, often
lock users into using specific data models or formats that are not helpful
when conducting analysis or data processing for machine learning.

For the above reasons, we decided to create our own client libraries that
would loosely integrate with the IL and tightly integrate with the
simulations themselves. The client libraries allow users to stream their
data out of their simulations, and recieve that data in an environment
and data format that is more suited for the type of analytical and processing
workflows done for machine learning. A user of the SmartSim clients
can decide where in their model data is to be communicated without changing
their data format internally and with only a few function
calls. The clients were designed to be minimally intrusive to the simulation
codebase and flexible enough to be fully integrated if need be.

In a nutshell, the clients make it possible to have a Fortran, C, or C++ program
communicating with a Python environment while the simulation is running. Combining
the IL with the clients, users don't ever have to leave a Jupyter notebook to
configure, launch, monitor, analyze and create learning systems from their
simulations.