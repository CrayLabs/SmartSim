.. SmartSim documentation master file, created by
   sphinx-quickstart on Sat Sep 14 15:07:14 2019.

SmartSim
========

SmartSim is a library that aides in the convergence of simulation and data science.
The goal of the library is to provide users with the ability to
conduct data science and machine learning experiments on simulation
model data in a flexible, online fashion.

Prior to SmartSim, operating on simulation model data was a slow and iterative
process. Users were often required to write large simulation outputs to file
and perform operations like analysis and visualization after the simulation
was finished. This manual process has been replaced in SmartSim with a
data pipeline that allows users to conduct analysis, visualization, ML
training and much more, while the simulation is running.

With SmartSim, users can stream data out of their models and operate
directly on the data at discrete time intervals decided by the user. The
framework is extremely flexible and can handle largely any process from
distributed machine learning to more tpyical data anaylsis and visualization.

SmartSim utilizes a project structure called an ``experiment`` that holds all of the
necessary files and data to run multiple simulations. Within each experiment are two
types of objects that allow a user complete control over how the simulation data is
generated and/or used. The first is called a ``ensemble``. ensembles are groups of models
with similar configurations. Within each ensemble are the actual models. The models
correspond to directories that house everything from configuration files to simulation
output.

A basic project structure with one ensemble and one model looks as follows:

.. code-block:: text

  lammps_atm           # experiment
  └── atm              # ensemble
      ├── atm_0        # model
      │   └── in.atm   # model files
      └── atm_1        # model
          └── in.atm   # model files


TODO
 - explain pipeline
 - KeyDB
 - nodes


.. toctree::
   :maxdepth: 2
   :caption: General

   doc/generate
   doc/simulate
   doc/developers

.. toctree::
   :maxdepth: 2
   :caption: Examples

   doc/examples/cp2k
   doc/examples/online-training


.. toctree::
   :maxdepth: 2
   :caption: API

   doc/api/state
   doc/api/simModule
   doc/api/generator
   doc/api/controller
   doc/api/client
   doc/api/utilities


Indices and tables
==================

* :ref:`search`
