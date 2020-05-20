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
distributed machine learning to more typical data anaylsis and visualization.

.. toctree::
   :maxdepth: 2
   :caption: General

   doc/experiment
   doc/generate
   doc/launchers
   doc/cmdservers
   doc/developers

.. toctree::
   :maxdepth: 2
   :caption: Examples

   doc/examples/basic
   doc/examples/ensembles
   doc/examples/jupyter
   doc/examples/urika
   doc/examples/MPO

.. toctree::
   :maxdepth: 2
   :caption: Clients

   doc/clients/fortran
   doc/clients/c-plus
   doc/clients/c
   doc/clients/python

.. toctree::
   :maxdepth: 2
   :caption: API

   doc/api/experiment
   doc/api/client
   doc/api/remote
   doc/api/utilities


Indices and tables
==================

* :ref:`search`
