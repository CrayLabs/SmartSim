.. SmartSim documentation master file, created by
   sphinx-quickstart on Sat Sep 14 15:07:14 2019.
   
SmartSim Documentation
======================

SmartSim is a library that aides in the convergence of simulation and machine learning.
The goal of the library is to provide users with the ability to easily and flexibly
conduct machine learning experiments on simulation model data.


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

   doc/overview


.. toctree::
   :maxdepth: 2
   :caption: Modules

   doc/api/state
   doc/api/simModule
   doc/api/generator
   doc/api/controller
   doc/api/processor
   

.. toctree::
   :maxdepth: 2
   :caption: Examples

   
   

Indices and tables
==================

* :ref:`search`
