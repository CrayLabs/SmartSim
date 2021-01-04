

************
Introduction
************

SmartSim is an HPE library that enables the convergence of numerical simulations
and AI workloads on heterogeneous architectures. SmartSim enables a data-science
first approach to simulation research by overcoming systemic impediments; including
lack of inter-operability between programming languages and dependence on file I/O.
SmartSim is portable and can be run on various platforms from local development on OSX/Linux
to large cluster and supercomputer systems with Slurm or PBS workload managers.

The primary user-facing object within the SmartSim library is the SmartSim Experiment,
encompassing all external entities (e.g. numerical simulations, data analysis tools,
or machine learning packages), as well as internal SmartSim objects that coordinate
the setup and execution of the SmartSim Experiment. SmartSim Experiment entities
share data through communication with an in-memory key-value store (Redis or KeyDB)
referred to as the Orchestrator. SmartSim includes clients in Python, C, C++, and Fortran
so that applications written in these languages can connect to the orchestrator at runtime.
The Orchestrator can be distributed on compute plane nodes (scaled horizontally) and
replicated on-node (scaled vertically) to increase throughput.

Furthermore, SmartSim is capable of co-locating inference-ready AI/ML models in popular
Deep Learning frameworks like Pytorch and TensorFlow with a simulation model or HPC workload.
AI models can be stored inside the Orchestrator and queried with the aforementioned clients.
The clients utilize a custom Tensor and Dataset format that allows for batches of n-dimensional
tensors to be sent to and from the Orchestrator for inference. Thus, the Orchestrator is
capable of providing distributed, online inference at runtime in C, C++, Fortran, and Python.

The figure below shows how a SmartSim experiment can connect single models as well as ensembles
to the Orchestrator for data storage and online inference. In addition to simulations, additional
applications can be launched together with simulations at the start of or during runtime for
online analysis, training, and visualization. Users configure the SmartSim Experiment
in Python, either in a script or with a Jupyter notebook.

.. |SmartSim Architecture| image:: images/SmartSim_Architecture.png
  :width: 700
  :alt: Alternative text

|SmartSim Architecture|


Library Design
==============

There are two core components of the SmartSim library:

  1. Infrastructure Library
  2. Client libraries (SILC)

The two libraries can either be used in conjunction, or separately depending
on the needs of the user.


Infrastructure Library
----------------------

The infrastructure library (IL) provides an API to automate the process of
deploying the infrastructure necessary to run complex experiments on a number
of different platforms. This includes interaction with the file system, workload
manager, system level APIs, operating system, external libraries, and simulation
models. The easiest way to conceptualize the IL is as a constructor of microservices
that facilitate coupling HPC workloads to modern data science tooling.

The IL is responsible for communicating with the operating system and optionally
a workload manager (like Slurm and PBS), if it's running on a supercomputer or
cluster system. This includes but is not limited to: obtaining, tracking and
releasing allocations; launching, stopping and monitoring jobs; and creating
communication channels between jobs.

The primary reason for implementing the IL is that many workflows in the HPC space
require complex configuration that reduces reproducibility and productivity.
The abstractions of the IL provide a simple mechanism for workflow configuration
which enables reproducibility across users, machines, and sites as well as increases productivity.

Clients
-------

Traditional workflows on an HPC system often lack flexibility and lock applications
into using only packages that are compatible with the software stack being used.
The SmartSim development team recognized this when conducting research on how to
connect simulation models to tools and libraries more commonly used in the domain
space of Data Science. A user with a traditional MPI workload was forced to either
make their application work with MPI, configure inflexible, complex RDMA communication,
or write output to the file system.

SmartSim is not the first library in the HPC space to aide with these inflexible application
coupling and communication paradigms. Libraries like SENSEI and GLEAN, and DataSpaces offer
useful methods to mitigate this inflexibility especially for large scale visualization.
SENSEI, in particular, offers adaptors that can hook into visualization tools like Paraview
for in-situ or in-transit visualization. These libraries, however, often force applications
into specific data formats (VTK, PNetCDF) that, while beneficial for visualization,
are not conducive for coupling simulations and machine learning.

For the above reasons, SmartSim has its own clients that loosely integrate with the IL
and tightly integrate with the simulations or HPC workloads themselves. The client
libraries allow users to stream their data out of their workload and receive that data
in an environment and data format that is suited for the type of analytical and processing
workflows done for machine learning. In addition, the clients enable remote execution
of AI/ML models in popular frameworks like TensorFlow, and Pytorch. The clients were
designed to be minimally intrusive to the simulation codebase requiring few function
calls to perform data transmission, and inference.

An important aspect of SmartSim in general is that it was not designed to be a replacement
of traditional I/O like ADIOS or GLEAN Rather, SmartSim was created as an augmentation
that allows for users couple their workloads with AI/ML techniques and send only the
data that needs to be analyzed, processed, visualized (in the sense of matplotlib rather than Paraview),
or inferred from.

Using SmartSim
==============

There are a number of ways SmartSim can be used, but the two mediums
that SmartSim was intended for are:

  1. Jupyter Environment
  2. Python script/shell

Jupyter Environment
-------------------

Jupyter notebooks have become the medium of choice for many data scientists
and Python programmers because they allow for self-documentation and quick
iteration. The latter is crucial for data analysis, data processing, and
prototyping of ML/DL models. SmartSim was designed with Jupyter in mind because
of these reasons. SmartSim allows for the simulation expert to never leave the
comfort of the Jupyter environment. Users can configure, run, and analyze
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



