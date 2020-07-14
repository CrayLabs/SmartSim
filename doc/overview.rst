

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

  1. Client libraries
  2. Infrastructure Library


Clients
=======

information on client

Infrastructure Library
======================

notes on the base library itself
