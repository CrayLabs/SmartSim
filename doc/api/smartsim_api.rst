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
   Experiment.get_status
   Experiment.wait
   Experiment.preview
   Experiment.summary
   Experiment.stop
   Experiment.telemetry

.. autoclass:: Experiment
   :show-inheritance:
   :members:


.. _settings-info:

Settings
========

.. currentmodule:: smartsim.settings

Settings are provided to ``Application`` and ``Ensemble`` objects
to provide parameters for how a job should be executed. Some
are specifically meant for certain launchers like ``SlurmBatchArguments``
is solely meant for system using Slurm as a workload manager.
``MpirunLaunchArguments`` for OpenMPI based jobs is supported by Slurm
and PBSPro.


Types of Settings:

.. autosummary::

    LaunchSettings
    SlurmLaunchArguments
    
    DragonLaunchArguments
    PalsMpiexecLaunchArguments
    AprunLaunchArguments
    LocalLaunchArguments
    MpiexecLaunchArguments
    MpirunLaunchArguments
    OrterunLaunchArguments
    JsrunLaunchArguments
    SlurmBatchArguments
    BsubBatchArguments
    QsubBatchArguments
    SgeQsubBatchSettings
   

Settings objects can accept a container object that defines a container
runtime, image, and arguments to use for the workload. Below is a list of
supported container runtimes.

Types of Containers:

.. autosummary::

    Singularity


.. _rs-api:

LaunchSettings
-----------


When running SmartSim on laptops and single node workstations,
the base ``LaunchSettings`` object is used to parameterize jobs.
``LaunchSettings`` include a ``run_command`` parameter for local
launches that utilize a parallel launch binary like
``mpirun``, ``mpiexec``, and others.


.. autosummary::

    LaunchSettings.launcher
    LaunchSettings.launch_args
    LaunchSettings.env_vars
    LaunchSettings.update_env

.. autoclass:: LaunchSettings
    :inherited-members:
    :undoc-members:
    :members:


.. _srun_api:

SlurmLaunchArguments
------------


``SlurmLaunchArguments`` can be used for running on existing allocations,
running jobs in interactive allocations, and for adding srun
steps to a batch.


.. autosummary::

    SlurmLaunchArguments.launcher_str
    SlurmLaunchArguments.set_nodes
    SlurmLaunchArguments.set_hostlist
    SlurmLaunchArguments.set_hostlist_from_file
    SlurmLaunchArguments.set_excluded_hosts
    SlurmLaunchArguments.set_cpus_per_task
    SlurmLaunchArguments.set_tasks
    SlurmLaunchArguments.set_tasks_per_node
    SlurmLaunchArguments.set_cpu_bindings
    SlurmLaunchArguments.set_memory_per_node
    SlurmLaunchArguments.set_executable_broadcast
    SlurmLaunchArguments.set_node_feature
    SlurmLaunchArguments.set_walltime
    SlurmLaunchArguments.set_het_group
    SlurmLaunchArguments.set_verbose_launch
    SlurmLaunchArguments.set_quiet_launch
    SlurmLaunchArguments.format_launch_args
    SlurmLaunchArguments.format_env_vars
    SlurmLaunchArguments.format_comma_sep_env_vars
    SlurmLaunchArguments.set


.. autoclass:: SlurmLaunchArguments
    :inherited-members:
    :undoc-members:
    :members:


.. _aprun_api:

AprunLaunchArguments
-------------


``AprunLaunchArguments`` can be used on any system that supports the
Cray ALPS layer. SmartSim supports using ``AprunLaunchArguments``
on PBSPro WLM systems.

``AprunLaunchArguments`` can be used in interactive session (on allocation)
and within batch launches (e.g., ``QsubBatchSettings``)


.. autosummary::

    AprunLaunchArguments.launcher_str
    AprunLaunchArguments.set_cpus_per_task
    AprunLaunchArguments.set_tasks
    AprunLaunchArguments.set_tasks_per_node
    AprunLaunchArguments.set_hostlist
    AprunLaunchArguments.set_hostlist_from_file
    AprunLaunchArguments.set_excluded_hosts
    AprunLaunchArguments.set_cpu_bindings
    AprunLaunchArguments.set_memory_per_node
    AprunLaunchArguments.set_walltime
    AprunLaunchArguments.set_verbose_launch
    AprunLaunchArguments.set_quiet_launch
    AprunLaunchArguments.format_env_vars
    AprunLaunchArguments.format_launch_args
    AprunLaunchArguments.set

.. autoclass:: AprunLaunchArguments
    :inherited-members:
    :undoc-members:
    :members:


.. _dragonsettings_api:

DragonLaunchArguments
-----------------

``DragonLaunchArguments`` can be used on systems that support Slurm or
PBS, if Dragon is available in the Python environment (see `_dragon_install`
for instructions on how to install it through ``smart``).

``DragonLaunchArguments`` can be used in interactive sessions (on allcation)
and within batch launches (i.e. ``SbatchSettings`` or ``QsubBatchSettings``,
for Slurm and PBS sessions, respectively).

.. autosummary::

    DragonLaunchArguments.launcher_str
    DragonLaunchArguments.set_nodes
    DragonRunSettings.set_tasks_per_node
    DragonLaunchArguments.set
    DragonLaunchArguments.set_node_feature
    DragonLaunchArguments.set_cpu_affinity
    DragonLaunchArguments.set_gpu_affinity

.. autoclass:: DragonRunSettings
    :inherited-members:
    :undoc-members:
    :members:


.. _jsrun_api:

JsrunLaunchArguments
-------------


``JsrunLaunchArguments`` can be used on any system that supports the
IBM LSF launcher.

``JsrunLaunchArguments`` can be used in interactive session (on allocation)
and within batch launches (i.e. ``BsubBatchSettings``)


.. autosummary::

    JsrunLaunchArguments.launcher_str
    JsrunLaunchArguments.set_tasks
    JsrunLaunchArguments.set_binding
    JsrunLaunchArguments.format_env_vars
    JsrunLaunchArguments.format_launch_args
    JsrunLaunchArguments.set

.. autoclass:: JsrunSettings
    :inherited-members:
    :undoc-members:
    :members:


.. _palsmpiexec_api:

 PalsMpiexecLaunchArguments
-------------


``PalsMpiexecLaunchArguments`` 


.. autosummary::

    PalsMpiexecLaunchArguments.launcher_str
    PalsMpiexecLaunchArguments.set_cpu_binding_type
    PalsMpiexecLaunchArguments.set_tasks
    PalsMpiexecLaunchArguments.set_executable_broadcast
    PalsMpiexecLaunchArguments.set_tasks_per_node
    PalsMpiexecLaunchArguments.set_hostlist
    PalsMpiexecLaunchArguments.format_env_vars
    PalsMpiexecLaunchArguments.format_launch_args
    PalsMpiexecLaunchArguments.set

.. autoclass:: PalsMpiexecLaunchArguments
    :inherited-members:
    :undoc-members:
    :members:



.. _openmpi_run_api:

MpirunLaunchArguments
--------------


``MpirunLaunchArguments`` are for launching with OpenMPI. ``BMpirunLaunchArguments`` are
supported on Slurm and PBSpro.


.. autosummary::

    MpirunLaunchArguments.set_task_map
    MpirunLaunchArguments.set_cpus_per_task
    MpirunLaunchArguments.set_executable_broadcast
    MpirunLaunchArguments.set_cpu_binding_type
    MpirunLaunchArguments.set_tasks_per_node
    MpirunLaunchArguments.set_tasks
    MpirunLaunchArguments.set_hostlist
    MpirunLaunchArguments.set_hostlist_from_file
    MpirunLaunchArguments.set_verbose_launch
    MpirunLaunchArguments.set_walltime
    MpirunLaunchArguments.set_quiet_launch
    MpirunLaunchArguments.format_env_vars
    MpirunLaunchArguments.format_launch_args
    MpirunLaunchArguments.set


.. autoclass:: MpirunLaunchArguments
    :inherited-members:
    :undoc-members:
    :members:

.. _openmpi_exec_api:

MpiexecLaunchArguments
---------------


``MpiexecLaunchArguments`` are for launching with OpenMPI's ``mpiexec``. ``MpirunLaunchArguments`` are
supported on Slurm and PBSpro.


.. autosummary::

    MpiexecLaunchArguments.launcher_str
    MpiexecLaunchArguments.set_task_map
    MpiexecLaunchArguments.set_cpus_per_task
    MpiexecLaunchArguments.set_executable_broadcast
    MpiexecLaunchArguments.set_cpu_binding_type
    MpiexecLaunchArguments.set_tasks_per_node
    MpiexecLaunchArguments.set_tasks
    MpiexecLaunchArguments.set_hostlist
    MpiexecLaunchArguments.set_hostlist_from_file
    MpiexecLaunchArguments.set_verbose_launch
    MpiexecLaunchArguments.set_walltime
    MpiexecLaunchArguments.set_quiet_launch
    MpiexecLaunchArguments.format_env_vars
    MpiexecLaunchArguments.format_launch_args
    MpiexecLaunchArguments.set


.. autoclass:: MpiexecLaunchArguments
    :inherited-members:
    :undoc-members:
    :members:

.. _openmpi_orte_api:

OrterunLaunchArguments
---------------


``OrterunLaunchArguments`` are for launching with OpenMPI's ``orterun``. ``OrterunLaunchArguments`` are
supported on Slurm and PBSpro.


.. autosummary::

    OrterunLaunchArguments.launcher_str
    OrterunLaunchArguments.set_task_map
    OrterunLaunchArguments.set_cpus_per_task
    OrterunLaunchArguments.set_executable_broadcast
    OrterunLaunchArguments.set_cpu_binding_type
    OrterunLaunchArguments.set_tasks_per_node
    OrterunLaunchArguments.set_tasks
    OrterunLaunchArguments.set_hostlist
    OrterunLaunchArguments.set_hostlist_from_file
    OrterunLaunchArguments.set_verbose_launch
    OrterunLaunchArguments.set_walltime
    OrterunLaunchArguments.set_quiet_launch
    OrterunLaunchArguments.format_env_vars
    OrterunLaunchArguments.format_launch_args
    OrterunLaunchArguments.set

.. autoclass:: OrterunSettings
    :inherited-members:
    :undoc-members:
    :members:


------------------------------------------


.. _sbatch_api:

SlurmBatchArguments
--------------


``SlurmBatchArguments`` are used for launching batches onto Slurm
WLM systems.


.. autosummary::

    SlurmBatchArguments.scheduler_str
    SlurmBatchArguments.set_walltime
    SlurmBatchArguments.set_nodes
    SlurmBatchArguments.set_account
    SlurmBatchArguments.set_partition
    SlurmBatchArguments.set_queue
    SlurmBatchArguments.set_cpus_per_task
    SlurmBatchArguments.set_hostlist
    SlurmBatchArguments.format_batch_args
    SlurmBatchArguments.set
    

.. autoclass:: SlurmBatchArguments
    :inherited-members:
    :undoc-members:
    :members:

.. _qsub_api:

QsubBatchArguments
-----------------


``QsubBatchArguments`` are used to configure jobs that should
be launched as a batch on PBSPro systems.


.. autosummary::

    QsubBatchArguments.scheduler_str
    QsubBatchArguments.set_nodes
    QsubBatchArguments.set_hostlist
    QsubBatchArguments.set_walltime
    QsubBatchArguments.set_queue
    QsubBatchArguments.set_ncpus
    QsubBatchArguments.set_account
    QsubBatchArguments.format_batch_args
    QsubBatchArguments.set


.. autoclass:: QsubBatchArguments
    :inherited-members:
    :undoc-members:
    :members:


.. _bsub_api:

BsubBatchArguments
-----------------


``BsubBatchArguments`` are used to configure jobs that should
be launched as a batch on LSF systems.


.. autosummary::

    BsubBatchArguments.scheduler_str
    BsubBatchArguments.set_walltime
    BsubBatchArguments.set_smts
    BsubBatchArguments.set_project
    BsubBatchArguments.set_account
    BsubBatchArguments.set_nodes
    BsubBatchArguments.set_hostlist
    BsubBatchArguments.set_tasks
    BsubBatchArguments.set_queue
    BsubBatchArguments.format_batch_args
    BsubBatchArguments.set


.. autoclass:: BsubBatchArguments
    :inherited-members:
    :undoc-members:
    :members:


.. _orc_api:

FeatureStore
============

.. currentmodule:: smartsim.database

.. autosummary::

   FeatureStore.__init__
   FeatureStore.fs_identifier
   FeatureStore.num_shards
   FeatureStore.fs_nodes
   FeatureStore.hosts
   FeatureStore.telemetry
   FeatureStore.reset_hosts
   FeatureStore.remove_stale_files
   FeatureStore.get_address
   FeatureStore.is_active
   FeatureStore.checkpoint_file
   FeatureStore.set_cpus
   FeatureStore.set_walltime
   FeatureStore.set_hosts
   FeatureStore.set_batch_arg
   FeatureStore.set_run_arg
   FeatureStore.enable_checkpoints
   FeatureStore.set_max_memory
   FeatureStore.set_eviction_strategy
   FeatureStore.set_max_clients
   FeatureStore.set_max_message_size
   FeatureStore.set_fs_conf

FeatureStore
------------

.. _FeatureStore_api:

.. autoclass:: FeatureStore
   :members:
   :inherited-members:
   :undoc-members:

.. _Application_api:

Application
=====

.. currentmodule:: smartsim.entity.application

.. autosummary::

   Application.__init__
   Application.exe
   Application.exe_args
   Application.add_exe_args
   Application.files
   Application.file_parameters
   Application.incoming_entities
   Application.key_prefixing_enabled
   Application.as_executable_sequence
   Application.attach_generator_files
   Application.attached_files_table
   Application.print_attached_files


Application
-----

.. autoclass:: Application
   :members:
   :show-inheritance:
   :inherited-members:

Ensemble
========

.. currentmodule:: smartsim.builders.ensemble

.. autosummary::

   Ensemble.__init__
   Ensemble.exe
   Ensemble.exe_args
   Ensemble.exe_arg_parameters
   Ensemble.files
   Ensemble.file_parameters
   Ensemble.permutation_strategy
   Ensemble.max_permutations
   Ensemble.replicas
   Ensemble.build_jobs


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

    fmt_walltime
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
