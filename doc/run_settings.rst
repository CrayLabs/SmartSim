.. _run_settings_doc:

************
Run Settings
************
========
Overview
========
``RunSettings`` are used in the SmartSim API to define how ``Model`` and ``Ensemble`` jobs
should be executed.

In general, ``RunSettings`` define:

- the executable
- the arguments to pass to the executable
- necessary environment variables at runtime
- the required compute resources

The base ``RunSettings`` class is utilized for local task launches,
while its derived child classes offer specialized functionality for HPC workload managers (WLMs).
Each SmartSim `launcher` interfaces with a specific ``RunSettings`` subclass tailored to an HPC job scheduler.

- Navigate to the :ref:`Local<run_settings_local_ex>` section to configure run settings locally
- Navigate to the :ref:`HPC Systems<run_settings_hpc_ex>` section to configure run settings for HPC

A ``RunSettings`` object is initialized through the ``Experiment.create_run_settings`` function.
This function accepts a `run_command` argument: the command to run the executable.

If `run_command` is set to `"auto"`, SmartSim will attempt to match a run command on the
system with a ``RunSettings`` class. If found, the class corresponding to
that `run_command` will be created and returned.

If the `run_command` is passed a recognized run command (e.g. `"srun"`) the ``RunSettings``
instance will be a child class such as ``SrunSettings``. You may also specify `"mpirun"`,
`"mpiexec"`, `"aprun"`, `"jsrun"` or `"orterun"` to the `run_command` argument.
This will return the associated child class.

If the run command is not supported by SmartSim, the base ``RunSettings`` class will be created and returned
with the specified `run_command` and `run_args` evaluated literally.

After creating a ``RunSettings`` instance, users gain access to the attributes and methods
of the associated child class, providing them with the ability to further configure the run
settings for jobs.

========
Examples
========
.. _run_settings_local_ex:

Local
=====
When running SmartSim on laptops and single node workstations via the `"local"`
`launcher`, job execution is configured with the base ``RunSettings`` object.
For local launches, ``RunSettings`` accepts a `run_command` parameter to allow
the use of parallel launch binaries like `"mpirun"`, `"mpiexec"`, and others.

If no `run_command` is specified and the ``Experiment`` `launcher` is set to `"local"`,
the executable is launched locally. When utilizing the `"local"` launcher and configuring
the `run_command` parameter to `"auto"` in the ``Experiment.create_run_settings`` factory
method, SmartSim defaults to omitting any run command prefix before the executable.

Once the ``RunSettings`` object is initialized using the ``Experiment.create_run_settings`` factory
method, the :ref:`RunSettings API<rs-api>` can be used to further configure the
``RunSettings`` object prior to execution.

.. note::
      The local `launcher` is the default `launcher` for all ``Experiment`` instances.

When the user initializes the ``Experiment`` at the beginning of the Python driver script,
a `launcher` argument may be specified. SmartSim will register or detect the
`launcher` and return the supported class upon a call to ``Experiment.create_run_settings``.
Below we demonstrate creating and configuring the base ``RunSettings``
object for local launches by specifying the `"local"` launcher during ``Experiment`` creation.
We also demonstrate specifying `run_command="mpirun"` locally.

**Initialize and Configure a RunSettings Object with No Run Command Specified:**

.. code-block:: python

      from smartsim import Experiment

      # Initialize the experiment and provide launcher local
      exp = Experiment("name-of-experiment", launcher="local")


      # Initialize a RunSettings object
      run_settings = exp.create_run_settings(exe="echo", exe_args="Hello World", run_command=None)

**Initialize and Configure a RunSettings Object with the `mpirun` Run Command Specified:**

.. note::
      Please note that to run this example you need to have an MPI implementation
      (e.g. OpenMPI or MPICH) installed.

.. code-block:: python

      from smartsim import Experiment

      # Initialize the experiment and provide launcher local
      exp = Experiment("name-of-experiment", launcher="local")

      # Initialize a RunSettings object
      run_settings = exp.create_run_settings(exe="echo", exe_args="Hello World", run_command="mpirun")

Users may replace `mpirun` with `mpiexec`.

.. _run_settings_hpc_ex:

HPC System
==========
To configure an entity for launch on an HPC system, SmartSim offers ``RunSettings`` child classes.
Each WLM `launcher` supports different ``RunSettings`` child classes.
When the user initializes the ``Experiment`` at the beginning of the Python driver script,
a `launcher` argument may be specified. The specified `launcher` will be used by SmartSim to
return the correct ``RunSettings`` child class that matches with the specified (or auto-detected)
`run_command` upon a call to ``Experiment.create_run_settings``. Below we demonstrate
creating and configuring the base ``RunSettings`` object for HPC launches
by specifying the launcher during ``Experiment`` creation. We show examples
for each job scheduler.

.. tabs::

    .. group-tab:: Slurm

      The Slurm `launcher` supports the :ref:`SrunSettings API <srun_api>` as well as the :ref:`MpirunSettings API <openmpi_run_api>`,
      :ref:`MpiexecSettings API <openmpi_exec_api>` and :ref:`OrterunSettings API <openmpi_orte_api>` that each can be used to run executables
      with launch binaries like `"srun"`, `"mpirun"`, `"mpiexec"` and `"orterun"`. Below we step through initializing a ``SrunSettings`` and ``MpirunSettings``
      instance on a Slurm based machine using the associated `run_command`.

      **SrunSettings**

      Run a job with the `srun` command on a Slurm based system. Any arguments passed in
      the `run_args` dict will be converted into `srun` arguments and prefixed with `"--"`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            from smartsim import Experiment

            # Initialize the Experiment and provide launcher Slurm
            exp = Experiment("name-of-experiment", launcher="slurm")

            # Initialize a SrunSettings object
            run_settings = exp.create_run_settings(exe="echo", exe_args="Hello World", run_command="srun")
            # Set the number of nodes
            run_settings.set_nodes(4)
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **MpirunSettings**

      Run a job with the `mpirun` command (MPI-standard) on a Slurm based system. Any
      arguments passed in the `run_args` dict will be converted into `mpirun` arguments
      and prefixed with `"--"`. Values of `None` can be provided for arguments that do
      not have values.

      .. code-block:: python

            from smartsim import Experiment

            # Initialize the Experiment and provide launcher Slurm
            exp = Experiment("name-of-experiment", launcher="slurm")

            # Initialize a MpirunSettings object
            run_settings = exp.create_run_settings(exe="echo", exe_args="Hello World", run_command="mpirun")
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      Users may replace `mpirun` with `mpiexec` or `orterun`.

    .. group-tab:: PBS Pro
      The PBS Pro `launcher` supports the :ref:`AprunSettings API <aprun_api>` as well as the :ref:`MpirunSettings API <openmpi_run_api>`,
      :ref:`MpiexecSettings API <openmpi_exec_api>` and :ref:`OrterunSettings API <openmpi_orte_api>` that each can be used to run executables
      with launch binaries like `"aprun"`, `"mpirun"`, `"mpiexec"` and `"orterun"`. Below we step through initializing a ``AprunSettings`` and ``MpirunSettings``
      instance on a PBS Pro based machine using the associated `run_command`.

      **AprunSettings**

      Run a job with `aprun` command on a PBS Pro based system. Any arguments passed in
      the `run_args` dict will be converted into `aprun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher PBS Pro
            exp = Experiment("name-of-experiment", launcher="pbs")

            # Initialize a AprunSettings object
            run_settings = exp.create_run_settings(exe="echo", exe_args="Hello World", run_command="aprun")
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **MpirunSettings**

      Run a job with `mpirun` command on a PBS Pro based system. Any arguments passed
      in the `run_args` dict will be converted into `mpirun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher PBS Pro
            exp = Experiment("name-of-experiment", launcher="pbs")

            # Initialize a MpirunSettings object
            run_settings = exp.create_run_settings(exe="echo", exe_args="Hello World", run_command="mpirun")
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      Users may replace `mpirun` with `mpiexec` or `orterun`.

    .. group-tab:: PALS
      The PALS `launcher` supports the :ref:`MpiexecSettings API <openmpi_exec_api>` that can be used to run executables
      with the `mpiexec` launch binary. Below we step through initializing a ``MpiexecSettings`` instance on a PALS
      based machine using the associated `run_command`.

      **MpiexecSettings**

      Run a job with `mpiexec` command on a PALS based system. Any arguments passed in the `run_args` dict will be converted into `mpiexec` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher PALS
            exp = Experiment("name-of-experiment", launcher="pals")

            # Initialize a MpiexecSettings object
            run_settings = exp.create_run_settings(exe="echo", exe_args="Hello World", run_command="mpiexec")
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

    .. group-tab:: LSF
      The LSF `launcher` supports the :ref:`JsrunSettings API <jsrun_api>` as well as the :ref:`MpirunSettings API <openmpi_run_api>`,
      :ref:`MpiexecSettings API <openmpi_exec_api>` and :ref:`OrterunSettings API <openmpi_orte_api>` that each can be used to run executables
      with launch binaries like `"jsrun"`, `"mpirun"`, `"mpiexec"` and `"orterun"`. Below we step through initializing a ``JsrunSettings`` and ``MpirunSettings``
      instance on a LSF based machine using the associated `run_command`.

      **JsrunSettings**

      Run a job with `jsrun` command on a LSF based system. Any arguments passed in the
      `run_args` dict will be converted into `jsrun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher LSF
            exp = Experiment("name-of-experiment", launcher="lsf")

            # Initialize a JsrunSettings object
            run_settings = exp.create_run_settings(exe="echo", exe_args="Hello World", run_command="jsrun")
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      **MpirunSettings**

      Run a job with `mpirun` command on a LSF based system. Any arguments passed in the
      `run_args` dict will be converted into `mpirun` arguments and prefixed with `--`.
      Values of `None` can be provided for arguments that do not have values.

      .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher LSF
            exp = Experiment("name-of-experiment", launcher="lsf")

            # Initialize a MpirunSettings object
            run_settings = exp.create_run_settings(exe="echo", exe_args="Hello World", run_command="mpirun")
            # Set the number of cpus to use per task
            run_settings.set_cpus_per_task(2)
            # Set the number of tasks for this job
            run_settings.set_tasks(100)
            # Set the number of tasks for this job
            run_settings.set_tasks_per_node(25)

      Users may replace `mpirun` with `mpiexec` or `orterun`.

.. note::
      SmartSim will look for an allocation by accessing the associated WLM job ID environment variable. If an allocation
      is present, the entity will be launched on the reserved compute resources. A user may also specify the allocation ID
      when initializing a run settings object via the `alloc` argument. If an allocation is specified, the entity receiving
      these run parameters will launch on that allocation.