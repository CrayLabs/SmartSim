========
Settings
========

.. currentmodule:: smartsim.settings

Settings are provided to ``Model`` and ``Ensemble`` objects
to provide parameters for how a job should be executed. Some
are specifically meant for certain launchers like ``SbatchSettings``
is solely meant for system using Slurm as a workload manager.
``MpirunSettings`` for OpenMPI based jobs is supported by Slurm,
PBSPro, and Cobalt.


Types of Settings:

.. autosummary::

   RunSettings
   SrunSettings
   SbatchSettings
   AprunSettings
   QsubBatchSettings
   CobaltBatchSettings
   MpirunSettings


Local
=====

When running SmartSim on laptops and single node workstations,
the base ``RunSettings`` object is used to parameterize jobs.
``RunSettings`` include a ``run_command`` parameter for local
launches that utilize a parallel launch binary like
``mpirun``, ``mpiexec``, and others.

.. autosummary::

   RunSettings.add_exe_args
   RunSettings.update_env

.. autoclass:: RunSettings
   :inherited-members:
   :undoc-members:
   :members:

------------------------------------------


Slurm
=====

SrunSettings
------------

``SrunSettings`` can be used for running on existing allocations,
running jobs in interactive allocations, and for adding srun
steps to a batch.

.. autosummary::

   SrunSettings.set_cpus_per_task
   SrunSettings.set_hostlist
   SrunSettings.set_nodes
   SrunSettings.set_tasks
   SrunSettings.set_tasks_per_node
   SrunSettings.add_exe_args
   SrunSettings.format_run_args
   SrunSettings.format_env_vars
   SrunSettings.update_env

.. autoclass:: SrunSettings
   :inherited-members:
   :undoc-members:
   :members:

SbatchSettings
--------------

``SbatchSettings`` are used for launching batches onto Slurm
WLM systems.

.. autosummary::

   SbatchSettings.set_account
   SbatchSettings.set_batch_command
   SbatchSettings.set_nodes
   SbatchSettings.set_hostlist
   SbatchSettings.set_partition
   SbatchSettings.set_walltime
   SbatchSettings.format_batch_args

.. autoclass:: SbatchSettings
   :inherited-members:
   :undoc-members:
   :members:

------------------------------------------

PBSPro
======

AprunSettings
-------------

``AprunSettings`` can be used on any system that suppports the
Cray ALPS layer. Currently ALPS is only tested on PBSpro
within SmartSim.

``AprunSettings`` can be used in interactive session (on allocation)
and within batch launches (``QsubBatchSettings``)

.. autosummary::

   AprunSettings.set_cpus_per_task
   AprunSettings.set_hostlist
   AprunSettings.set_tasks
   AprunSettings.set_tasks_per_node
   AprunSettings.make_mpmd
   AprunSettings.add_exe_args
   AprunSettings.format_run_args
   AprunSettings.format_env_vars
   AprunSettings.update_env

.. autoclass:: AprunSettings
   :inherited-members:
   :undoc-members:
   :members:


QsubBatchSettings
-----------------

``QsubBatchSettings`` are used to configure jobs that should
be launched as a batch on PBSPro systems.

.. autosummary::

   QsubBatchSettings.set_account
   QsubBatchSettings.set_batch_command
   QsubBatchSettings.set_nodes
   QsubBatchSettings.set_ncpus
   QsubBatchSettings.set_queue
   QsubBatchSettings.set_resource
   QsubBatchSettings.set_walltime
   QsubBatchSettings.format_batch_args


.. autoclass:: QsubBatchSettings
   :inherited-members:
   :undoc-members:
   :members:


------------------------------------------


Cobalt
======

CobaltBatchSettings
-------------------

.. autosummary::

   CobaltBatchSettings.set_account
   CobaltBatchSettings.set_batch_command
   CobaltBatchSettings.set_nodes
   CobaltBatchSettings.set_queue
   CobaltBatchSettings.set_walltime
   CobaltBatchSettings.format_batch_args

.. autoclass:: CobaltBatchSettings
   :inherited-members:
   :undoc-members:
   :members:

------------------------------------------


General
=======

The following are run setting types that are supported on multiple
launchers

MpirunSettings
--------------

``MpirunSettings`` are for launching with OpenMPI. ``MpirunSettings`` are
supported on Slurm, PBSpro, and Cobalt.

.. autosummary::

   MpirunSettings.set_cpus_per_task
   MpirunSettings.set_hostlist
   MpirunSettings.set_tasks
   MpirunSettings.set_task_map
   MpirunSettings.make_mpmd
   MpirunSettings.add_exe_args
   MpirunSettings.format_run_args
   MpirunSettings.format_env_vars
   MpirunSettings.update_env

.. autoclass:: MpirunSettings
   :inherited-members:
   :undoc-members:
   :members:
