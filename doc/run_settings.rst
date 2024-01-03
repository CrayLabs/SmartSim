************
Run Settings
************

========
Overview
========
SmartSim supports configuring run settings for entities through the
``RunSettings`` object which acts as the base class for launching jobs locally or in
parallel using binaries like mpirun and mpiexec. This ``RunSettings`` base class extends
to specialized subclasses designed for HPC systems and Workload Managers (WLM) to support
job launches with environment-specific binaries.

- Navigate to :ref:`Local<Local>` section to configure run settings locally
- Navigate to :ref:`HPC<HPC>` section to configure run settings for a HPC

A run settings object is initialized through the ``Experiment.create_run_settings()`` function.
This function accepts a `run_command` argument: the command to run the executable.
If `run_command` is set to `auto`, SmartSim will attempt to match a run command on the
system with a ``RunSettings`` class. If found, the class corresponding to
that `run_command` will be created and returned.
If the `run_command` is passed a recognized run command (e.g. 'srun') the ``RunSettings``
instance will be a child class such as ``SrunSettings``.
If not supported by SmartSim, the base ``RunSettings`` class will be
created and returned with the specified `run_command` and `run_args` evaluated literally.
You may also specify `mpirun`, `mpiexec` or `orterun` to the `run_command` argument. This will return
the associated child class.

Once a ``RunSettings`` object is initialized, you have access to feature set that
further configures the run settings of a job.

=====
Local
=====
When running SmartSim on laptops and single node workstations, the base ``RunSettings`` object is
used to parameterize jobs. RunSettings include a `run_command` parameter for local launches
that utilize a parallel launch binary like `mpirun`, `mpiexec`, and others. If no `run_command`
is specified, the executable will be launched locally.

The local `launcher` supports the base :ref:`RunSettings API <rs-api>` which can be used to
run executables as well as run executables with arbitrary launch binaries like `mpirun` and `mpiexec`.
The local launcher is the default launcher for all ``Experiment`` instances.

When the user initializes the ``Experiment`` at the beginning of the Python driver script,
a `launcher` argument may be specified. SmartSim will register or detect the `launcher` and return the supported class
upon a call to ``Experiment.create_run_settings()``. A user may also specify the launcher when initializing
the run settings object. Below we demonstrate creating and configuring the base ``RunSettings`` object for local launches
by specifying the launcher during run settings creation. We show an example for each run command that may be provided.

Initialize and configure a run settings object with no run command specified:

.. code-block:: python

      # Initialize a RunSettings object
      run_settings = exp.create_run_settings(launcher="local", "echo", exe_args="Hello World", run_command=None)
      # Set the number of nodes
      run_settings.set_nodes(2)
      # Set the number of cpus to use per task
      run_settings.set_cpus_per_task(2)
      # Set the number of tasks for this job
      run_settings.set_tasks(10)
      # Set the number of tasks for this job
      run_settings.set_tasks_per_node(5)

Initialize and configure a run settings object with the `mpirun` run command specified:

.. code-block:: python

      # Initialize a RunSettings object
      run_settings = exp.create_run_settings(launcher="local", "echo", exe_args="Hello World", run_command="mpirun")
      # Set the number of nodes
      run_settings.set_nodes(2)
      # Set the number of cpus to use per task
      run_settings.set_cpus_per_task(2)
      # Set the number of tasks for this job
      run_settings.set_tasks(10)
      # Set the number of tasks for this job
      run_settings.set_tasks_per_node(5)

Initialize and configure a run settings object with the `mpiexec` run command specified:

.. code-block:: python

      # Initialize a RunSettings object
      run_settings = exp.create_run_settings(launcher="local", "echo", exe_args="Hello World", run_command="mpiexec")
      # Set the number of nodes
      run_settings.set_nodes(2)
      # Set the number of cpus to use per task
      run_settings.set_cpus_per_task(2)
      # Set the number of tasks for this job
      run_settings.set_tasks(10)
      # Set the number of tasks for this job
      run_settings.set_tasks_per_node(5)

===
HPC
===
To configure an entity for launch on an HPC, SmartSim offers ``RunSettings`` child classes.
Each WLM `launcher` supports different ``RunSettings`` child classes.
When the user initializes the ``Experiment`` at the beginning of the Python driver script,
a `launcher` argument may be specified. SmartSim will register or detect the `launcher` and return the supported class
upon a call to ``Experiment.create_run_settings()``. A user may also specify the launcher when initializing
the run settings object. Below we demonstrate creating and configuring the base ``RunSettings`` object for HPC launches
by specifying the launcher during run settings creation. We show an example for each run command that may be provided.

.. tabs::

    .. group-tab:: Slurm

      The Slurm `launcher` supports the :ref:`SrunSettings API <srun_api>` as well as the :ref:`MpirunSettings API <openmpi_run_api>`,
      :ref:`MpiexecSettings API <openmpi_exec_api>` and :ref:`OrterunSettings API <openmpi_orte_api>` that each can be used to run executables
      with arbitrary launch binaries like `mpirun`, `mpiexec` and `orterun`.

      **SrunSettings**

      Run a job with `srun` command on a Slurm based system. Any arguments passed in the `run_args` dict will be converted into `srun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a SrunSettings object
            run_settings = exp.create_run_settings(launcher="slurm", exe="echo", exe_args="Hello World", run_command="srun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **OrterunSettings**

      Run a job with `orterun` command (MPI-standard) on a Slurm based system. Any arguments passed in the `run_args` dict will be converted into `orterun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a OrterunSettings object
            run_settings = exp.create_run_settings(launcher="slurm", exe="echo", exe_args="Hello World", run_command="orterun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **MpirunSettings**

      Run a job with `mpirun` command (MPI-standard) on a Slurm based system. Any arguments passed in the `run_args` dict will be converted into `mpirun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a MpirunSettings object
            run_settings = exp.create_run_settings(launcher="slurm", exe="echo", exe_args="Hello World", run_command="mpirun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **MpiexecSettings**

      Run a job with `mpiexec` command (MPI-standard) on a Slurm based system. Any arguments passed in the `run_args` dict will be converted into `mpiexec` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a MpiexecSettings object
            run_settings = exp.create_run_settings(launcher="slurm", exe="echo", exe_args="Hello World", run_command="mpiexec")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

    .. group-tab:: PBSpro
      The PBSpro `launcher` supports the :ref:`AprunSettings API <aprun_api>` as well as the :ref:`MpirunSettings API <openmpi_run_api>`,
      :ref:`MpiexecSettings API <openmpi_exec_api>` and :ref:`OrterunSettings API <openmpi_orte_api>` that each can be used to run executables
      with arbitrary launch binaries like `mpirun`, `mpiexec` and `orterun`.

      **AprunSettings**

      Run a job with `aprun` command on a PBSpro based system. Any arguments passed in the `run_args` dict will be converted into `aprun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a AprunSettings object
            run_settings = exp.create_run_settings(launcher="pbs", exe="echo", exe_args="Hello World", run_command="aprun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **OrterunSettings**

      Run a job with `orterun` command on a PBSpro based system. Any arguments passed in the `run_args` dict will be converted into `orterun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a OrterunSettings object
            run_settings = exp.create_run_settings(launcher="pbs", exe="echo", exe_args="Hello World", run_command="orterun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **MpirunSettings**

      Run a job with `mpirun` command on a PBSpro based system. Any arguments passed in the `run_args` dict will be converted into `mpirun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a MpirunSettings object
            run_settings = exp.create_run_settings(launcher="pbs", exe="echo", exe_args="Hello World", run_command="mpirun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **MpiexecSettings**

      Run a job with `mpiexec` command on a PBSpro based system. Any arguments passed in the `run_args` dict will be converted into `mpiexec` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a MpiexecSettings object
            run_settings = exp.create_run_settings(launcher="pbs", exe="echo", exe_args="Hello World", run_command="mpiexec")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

    .. group-tab:: LSF
      The LSF `launcher` supports the :ref:`JsrunSettings API <jsrun_api>` as well as the :ref:`MpirunSettings API <openmpi_run_api>`,
      :ref:`MpiexecSettings API <openmpi_exec_api>` and :ref:`OrterunSettings API <openmpi_orte_api>` that each can be used to run executables
      with arbitrary launch binaries like `mpirun`, `mpiexec` and `orterun`.

      **JsrunSettings**

      Run a job with `jsrun` command on a LSF based system. Any arguments passed in the `run_args` dict will be converted into `jsrun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a JsrunSettings object
            run_settings = exp.create_run_settings(launcher="lsf", exe="echo", exe_args="Hello World", run_command="jsrun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **OrterunSettings**

      Run a job with `orterun` command on a LSF based system. Any arguments passed in the `run_args` dict will be converted into `orterun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a OrterunSettings object
            run_settings = exp.create_run_settings(launcher="lsf", exe="echo", exe_args="Hello World", run_command="orterun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **MpirunSettings**

      Run a job with `mpirun` command on a LSF based system. Any arguments passed in the `run_args` dict will be converted into `mpirun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a MpirunSettings object
            run_settings = exp.create_run_settings(launcher="lsf", exe="echo", exe_args="Hello World", run_command="mpirun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **MpiexecSettings**

      Run a job with `mpiexec` command on a LSF based system. Any arguments passed in the `run_args` dict will be converted into `mpiexec` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            # Initialize a MpiexecSettings object
            run_settings = exp.create_run_settings(launcher="lsf", exe="echo", exe_args="Hello World", run_command="mpiexec")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

.. note::
      SmartSim will look for a allocation by accessing the associated WLM job ID environment variable. If an allocation
      is present, the entity will be launched on the reserved compute resources. A user may also specify the allocation ID
      when initializing a run settings object via the `alloc` argument. If an allocation is specified, the entity receiving
      these run parameters will launch on that allocation.