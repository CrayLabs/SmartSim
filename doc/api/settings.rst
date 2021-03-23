
Settings
========

.. currentmodule:: smartsim.settings

RunSettings
-----------

The base RunSettings class can be used for parameterizing
launch for local jobs.

.. autoclass:: RunSettings
   :show-inheritance:
   :inherited-members:
   :undoc-members:
   :members:

SrunSettings
------------

SrunSettings can be used for running on existing allocations,
running jobs in interactive allocations, and for adding srun
steps to a batch

.. autoclass:: SrunSettings
   :show-inheritance:
   :inherited-members:
   :undoc-members:
   :members:

SbatchSettings
--------------

SbatchSettings are used for launching batches onto Slurm
WLM systems.

.. autoclass:: SbatchSettings
   :show-inheritance:
   :inherited-members:
   :undoc-members:
   :members:

AprunSettings
-------------

AprunSettings can be used on any system that suppports the
Cray ALPS layer. Currently ALPS is only tested on PBSpro
within SmartSim.

AprunSettings can be used in interactive session (on allocation)
and within batch launches (QsubBatchSettings)

.. autoclass:: AprunSettings
   :show-inheritance:
   :inherited-members:
   :undoc-members:
   :members:

QsubBatchSettings
-----------------

QsubBatchSettings are used to configure jobs that should
be launched as a batch on PBSPro systems.

.. autoclass:: QsubBatchSettings
   :show-inheritance:
   :inherited-members:
   :undoc-members:
   :members:
