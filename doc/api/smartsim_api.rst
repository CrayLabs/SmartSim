*************
SmartSim API
*************

.. _experiment_api:

Experiment
==========

.. currentmodule:: smartsim.experiment

.. _exp_init:
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
   Experiment.preview
   Experiment.summary
   Experiment.telemetry

.. autoclass:: Experiment
   :show-inheritance:
   :members:


.. _settings-info:

Settings
========

.. currentmodule:: smartsim.settings

Settings are provided to ``Model`` and ``Ensemble`` objects
to provide parameters for how a job should be executed. Some
are specifically meant for certain launchers like ``SbatchSettings``
is solely meant for system using Slurm as a workload manager.
``MpirunSettings`` for OpenMPI based jobs is supported by Slurm
and PBSPro.


Types of Settings:

.. autosummary::

    RunSettings
    SrunSettings
    AprunSettings
    MpirunSettings
    MpiexecSettings
    OrterunSettings
    JsrunSettings
    DragonRunSettings
    SbatchSettings
    QsubBatchSettings
    BsubBatchSettings

Settings objects can accept a container object that defines a container
runtime, image, and arguments to use for the workload. Below is a list of
supported container runtimes.

Types of Containers:

.. autosummary::

    Singularity


.. _rs-api:

RunSettings
-----------


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


.. _srun_api:

SrunSettings
------------


``SrunSettings`` can be used for running on existing allocations,
running jobs in interactive allocations, and for adding srun
steps to a batch.


.. autosummary::

    SrunSettings.set_nodes
    SrunSettings.set_node_feature
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


.. _aprun_api:

AprunSettings
-------------


``AprunSettings`` can be used on any system that supports the
Cray ALPS layer. SmartSim supports using ``AprunSettings``
on PBSPro WLM systems.

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


.. _dragonsettings_api:

DragonRunSettings
-----------------

``DragonRunSettings`` can be used on systems that support Slurm or
PBS, if Dragon is available in the Python environment (see `_dragon_install`
for instructions on how to install it through ``smart``).

``DragonRunSettings`` can be used in interactive sessions (on allcation)
and within batch launches (i.e. ``SbatchSettings`` or ``QsubBatchSettings``,
for Slurm and PBS sessions, respectively).

.. autosummary::
    DragonRunSettings.set_nodes
    DragonRunSettings.set_tasks_per_node

.. autoclass:: DragonRunSettings
    :inherited-members:
    :undoc-members:
    :members:


.. _jsrun_api:

JsrunSettings
-------------


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

.. _openmpi_run_api:

MpirunSettings
--------------


``MpirunSettings`` are for launching with OpenMPI. ``MpirunSettings`` are
supported on Slurm and PBSpro.


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

.. _openmpi_exec_api:

MpiexecSettings
---------------


``MpiexecSettings`` are for launching with OpenMPI's ``mpiexec``. ``MpirunSettings`` are
supported on Slurm and PBSpro.


.. autosummary::

    MpiexecSettings.set_cpus_per_task
    MpiexecSettings.set_hostlist
    MpiexecSettings.set_tasks
    MpiexecSettings.set_task_map
    MpiexecSettings.make_mpmd
    MpiexecSettings.add_exe_args
    MpiexecSettings.format_run_args
    MpiexecSettings.format_env_vars
    MpiexecSettings.update_env

.. autoclass:: MpiexecSettings
    :inherited-members:
    :undoc-members:
    :members:

.. _openmpi_orte_api:

OrterunSettings
---------------


``OrterunSettings`` are for launching with OpenMPI's ``orterun``. ``OrterunSettings`` are
supported on Slurm and PBSpro.


.. autosummary::

    OrterunSettings.set_cpus_per_task
    OrterunSettings.set_hostlist
    OrterunSettings.set_tasks
    OrterunSettings.set_task_map
    OrterunSettings.make_mpmd
    OrterunSettings.add_exe_args
    OrterunSettings.format_run_args
    OrterunSettings.format_env_vars
    OrterunSettings.update_env

.. autoclass:: OrterunSettings
    :inherited-members:
    :undoc-members:
    :members:


------------------------------------------


.. _sbatch_api:

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
    SbatchSettings.set_queue
    SbatchSettings.set_walltime
    SbatchSettings.format_batch_args

.. autoclass:: SbatchSettings
    :inherited-members:
    :undoc-members:
    :members:

.. _qsub_api:

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


.. _bsub_api:

BsubBatchSettings
-----------------


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

.. _singularity_api:

Singularity
-----------


``Singularity`` is a type of ``Container`` that can be passed to a
``RunSettings`` class or child class to enable running the workload in a
container.

.. autoclass:: Singularity
    :inherited-members:
    :undoc-members:
    :members:

.. _orc_api:

Orchestrator
============

.. currentmodule:: smartsim.database

.. autosummary::

   Orchestrator.__init__
   Orchestrator.db_identifier
   Orchestrator.num_shards
   Orchestrator.db_nodes
   Orchestrator.hosts
   Orchestrator.reset_hosts
   Orchestrator.remove_stale_files
   Orchestrator.get_address
   Orchestrator.is_active
   Orchestrator.set_cpus
   Orchestrator.set_walltime
   Orchestrator.set_hosts
   Orchestrator.set_batch_arg
   Orchestrator.set_run_arg
   Orchestrator.enable_checkpoints
   Orchestrator.set_max_memory
   Orchestrator.set_eviction_strategy
   Orchestrator.set_max_clients
   Orchestrator.set_max_message_size
   Orchestrator.set_db_conf
   Orchestrator.telemetry
   Orchestrator.checkpoint_file
   Orchestrator.batch

Orchestrator
------------

.. _orchestrator_api:

.. autoclass:: Orchestrator
   :members:
   :inherited-members:
   :undoc-members:

.. _model_api:

Model
=====

.. currentmodule:: smartsim.entity.model

.. autosummary::

   Model.__init__
   Model.attach_generator_files
   Model.colocate_db
   Model.colocate_db_tcp
   Model.colocate_db_uds
   Model.colocated
   Model.add_ml_model
   Model.add_script
   Model.add_function
   Model.params_to_args
   Model.register_incoming_entity
   Model.enable_key_prefixing
   Model.disable_key_prefixing
   Model.query_key_prefixing

Model
-----

.. autoclass:: Model
   :members:
   :show-inheritance:
   :inherited-members:

Ensemble
========

.. currentmodule:: smartsim.entity.ensemble

.. autosummary::

   Ensemble.__init__
   Ensemble.add_model
   Ensemble.add_ml_model
   Ensemble.add_script
   Ensemble.add_function
   Ensemble.attach_generator_files
   Ensemble.enable_key_prefixing
   Ensemble.models
   Ensemble.query_key_prefixing
   Ensemble.register_incoming_entity

Ensemble
--------

.. _ensemble_api:

.. autoclass:: Ensemble
   :members:
   :show-inheritance:
   :inherited-members:

.. _ml_api:

Machine Learning
================


SmartSim includes built-in utilities for supporting TensorFlow, Keras, and Pytorch.

.. _smartsim_tf_api:

TensorFlow
----------

SmartSim includes built-in utilities for supporting TensorFlow and Keras in training and inference.

.. currentmodule:: smartsim.ml.tf.utils

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

.. _smartsim_torch_api:

PyTorch
----------

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

.. _slurm_module_api:

Slurm
=====

.. currentmodule:: smartsim.wlm.slurm

.. autosummary::

    get_allocation
    release_allocation
    validate
    get_default_partition
    get_hosts
    get_queue
    get_tasks
    get_tasks_per_node

.. automodule:: smartsim.wlm.slurm
    :members:
