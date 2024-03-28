
*************
SmartSim API
*************


Experiment
==========

.. _experiment_api:

.. currentmodule:: smartsim.experiment

.. autosummary::

   Experiment.__init__
   Experiment.start
   Experiment.stop
   Experiment.create_ensemble
   Experiment.create_model
   Experiment.create_database
   Experiment.create_run_settings
   Experiment.create_batch_settings
   Experiment.generate
   Experiment.poll
   Experiment.finished
   Experiment.get_status
   Experiment.reconnect_orchestrator
   Experiment.summary

.. autoclass:: Experiment
   :show-inheritance:
   :members:


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
    AprunSettings
    MpirunSettings
    JsrunSettings
    SbatchSettings
    QsubBatchSettings
    CobaltBatchSettings
    BsubBatchSettings


RunSettings
-----------

.. _rs-api:

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


SrunSettings
------------

.. _srun_api:

``SrunSettings`` can be used for running on existing allocations,
running jobs in interactive allocations, and for adding srun
steps to a batch.


.. autosummary::

    SrunSettings.set_nodes
    SrunSettings.set_tasks
    SrunSettings.set_tasks_per_node
    SrunSettings.set_walltime
    SrunSettings.set_hostlist
    SrunSettings.set_excluded_hosts
    SrunSettings.set_cpus_per_task
    SrunSettings.add_exe_args
    SrunSettings.format_run_args
    SrunSettings.format_env_vars
    SrunSettings.update_env

.. autoclass:: SrunSettings
    :inherited-members:
    :undoc-members:
    :members:



AprunSettings
-------------

.. _aprun_api:

``AprunSettings`` can be used on any system that supports the
Cray ALPS layer. SmartSim supports using ``AprunSettings``
on PBSPro and Cobalt WLM systems.

``AprunSettings`` can be used in interactive session (on allocation)
and within batch launches (e.g., ``QsubBatchSettings``)


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




JsrunSettings
-------------

.. _jsrun_api:

``JsrunSettings`` can be used on any system that supports the
IBM LSF launcher.

``JsrunSettings`` can be used in interactive session (on allocation)
and within batch launches (i.e. ``BsubBatchSettings``)


.. autosummary::

    JsrunSettings.set_num_rs
    JsrunSettings.set_cpus_per_rs
    JsrunSettings.set_gpus_per_rs
    JsrunSettings.set_rs_per_host
    JsrunSettings.set_tasks
    JsrunSettings.set_tasks_per_rs
    JsrunSettings.set_binding
    JsrunSettings.make_mpmd
    JsrunSettings.set_mpmd_preamble
    JsrunSettings.update_env
    JsrunSettings.set_erf_sets
    JsrunSettings.format_env_vars
    JsrunSettings.format_run_args


.. autoclass:: JsrunSettings
    :inherited-members:
    :undoc-members:
    :members:


MpirunSettings
--------------

.. _openmpi_api:

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


------------------------------------------



SbatchSettings
--------------

.. _sbatch_api:

``SbatchSettings`` are used for launching batches onto Slurm
WLM systems.


.. autosummary::

    SbatchSettings.set_account
    SbatchSettings.set_batch_command
    SbatchSettings.set_nodes
    SbatchSettings.set_hostlist
    SbatchSettings.set_partition
    SbatchSettings.set_queue
    SbatchSettings.set_walltime
    SbatchSettings.format_batch_args

.. autoclass:: SbatchSettings
    :inherited-members:
    :undoc-members:
    :members:


QsubBatchSettings
-----------------

.. _qsub_api:

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



CobaltBatchSettings
-------------------

.. _cqsub_api:

``CobaltBatchSettings`` are used to configure jobs that should
be launched as a batch on Cobalt Systems. They closely mimic
that of the ``QsubBatchSettings`` for PBSPro.


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



BsubBatchSettings
-----------------

.. _bsub_api:

``BsubBatchSettings`` are used to configure jobs that should
be launched as a batch on LSF systems.


.. autosummary::

    BsubBatchSettings.set_walltime
    BsubBatchSettings.set_smts
    BsubBatchSettings.set_project
    BsubBatchSettings.set_nodes
    BsubBatchSettings.set_expert_mode_req
    BsubBatchSettings.set_hostlist
    BsubBatchSettings.set_tasks
    BsubBatchSettings.format_batch_args


.. autoclass:: BsubBatchSettings
    :inherited-members:
    :undoc-members:
    :members:


Orchestrator
============

.. currentmodule:: smartsim.database


Orchestrator
------------

.. _orc_api:

.. autoclass:: Orchestrator
   :members:
   :inherited-members:
   :undoc-members:


Model
=====

.. currentmodule:: smartsim.entity.model

.. autosummary::

   Model.__init__
   Model.attach_generator_files
   Model.colocate_db
   Model.params_to_args
   Model.register_incoming_entity
   Model.enable_key_prefixing
   Model.disable_key_prefixing
   Model.query_key_prefixing

.. autoclass:: Model
   :members:
   :show-inheritance:
   :inherited-members:


Ensemble
========

.. currentmodule:: smartsim.entity.ensemble

.. autosummary::

   Ensemble.__init__
   Ensemble.models
   Ensemble.add_model
   Ensemble.attach_generator_files
   Ensemble.register_incoming_entity
   Ensemble.enable_key_prefixing
   Ensemble.query_key_prefixing

.. autoclass:: Ensemble
   :members:
   :show-inheritance:
   :inherited-members:


Machine Learning
================

.. _ml_api:

SmartSim includes built-in utilities for supporting TensorFlow, Keras, and Pytorch.

TensorFlow
----------

.. _smartsim_tf_api:

SmartSim includes built-in utilities for supporting TensorFlow and Keras in training and inference.

.. currentmodule:: smartsim.ml.tf.utils

.. autosummary::

    freeze_model

.. automodule:: smartsim.ml.tf.utils
    :members:


.. currentmodule:: smartsim.ml.tf

.. autoclass:: StaticDataGenerator
   :show-inheritance:
   :inherited-members:
   :members:

.. autoclass:: DynamicDataGenerator
   :members:
   :show-inheritance:
   :inherited-members:

PyTorch
----------

.. _smartsim_torch_api:

SmartSim includes built-in utilities for supporting PyTorch in training and inference.

.. currentmodule:: smartsim.ml.torch

.. autoclass:: StaticDataGenerator
   :members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: DynamicDataGenerator
   :members:
   :show-inheritance:
   :inherited-members:

.. autoclass:: DataLoader
   :members:
   :show-inheritance:
   :inherited-members:

Slurm
=====

.. _slurm_module_api:


.. currentmodule:: smartsim.slurm

.. autosummary::

    get_allocation
    release_allocation

.. automodule:: smartsim.slurm
    :members:


Ray
===

.. currentmodule:: smartsim.exp.ray

.. _ray_api:

``RayCluster`` is used to launch a Ray cluster
 and can be launched as a batch or in an interactive allocation.

.. autoclass:: RayCluster
    :show-inheritance:
    :members:
    :inherited-members:
    :undoc-members:
    :exclude-members: batch set_path type
