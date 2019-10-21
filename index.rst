.. SmartSim documentation master file, created by
   sphinx-quickstart on Sat Sep 14 15:07:14 2019.
   
SmartSim Documentation
======================

SmartSim is a library that aides in the convergence of simulation and machine learning.
The goal of the library is to provide users with the ability to easily and flexibly
conduct machine learning experiments on simulation model data.

The library was built with three goals in mind. First, the process of generating
simulation data accross various models is manual and slow. Each model has their own
types of configuration files and not all of them are easily parsed. With SmartSim,
any text based configuration file can be parsed and modified programmatically such
that many different configurations of a single model can be generated efficently.
Efficient configuration and generation creates opportunies to explore and optimize
models through effiecent search algorithms and techniques like Model Parameter
Optimization (MPO).

The latter two goals of the library work in tandem: Online training and inference.
Online training is the process of training a machine learning model on the
data being produced by a simulation model as the simulation is progressing. Inference
follows the same logic, but adds a step where data is being passed back into the
simulation. (fill in more when this is finished)


SmartSim utilizes a project structure called an ``experiment`` that holds all of the
necessary files and data to run multiple simulations. Within each experiment are two
types of objects that allow a user complete control over how the simulation data is
generated and/or used. The first is called a ``target``. Targets are groups of models
with similar configurations. Within each target are the actual models. The models
correspond to directories that house everything from configuration files to simulation
output.

A basic project structure with one target and one model looks as follows:

.. code-block:: text

  lammps_atm           # experiment
  └── atm              # target
      ├── atm_0        # model
      │   └── in.atm   # model files
      └── atm_1        # model 
          └── in.atm   # model files

SmartSim is broken down into multiple modules that each have a specific purpose, and
can be combined in many different ways to form unique workflows for specific use cases.

Each module must be given an instance of the State class. The state class allows for
each module to know where each part of the experiment are located, retrieve user
configurations, log to a cental logging system and other functionality. For more
information on the state class, please see x


.. toctree::
   :maxdepth: 2
   :caption: General

   doc/interfaces
   doc/generate
   doc/simulate
   doc/developers

.. toctree::
   :maxdepth: 2
   :caption: Examples

   doc/examples/mom6
   doc/examples/cp2k
   doc/examples/mpo
   doc/examples/online-training


.. toctree::
   :maxdepth: 2
   :caption: Modules

   doc/api/state
   doc/api/simModule
   doc/api/generator
   doc/api/controller
   
   

Indices and tables
==================

* :ref:`search`
