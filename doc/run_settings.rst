************
Run Settings
************

========
Overview
========
SmartSim allows you to configure the run settings of a job (``Model`` or ``Ensemble``)
based on the system and available computational resources
by initializing a ``RunSettings`` object. `Launchers` work directly with WLM schedulers to
launch, query, monitor and stop applications. For the local `launcher`, SmartSim supports
a base ``RunSettings`` object. For the WLM `launchers`, SmartSim supports a variety of
``RunSettings`` child classes.

* Navigate to :ref:`local section<Local>` to configure locally
* Navigate to :ref:`wlm section<HPC>` to configure for WLM

A run settings object is initialized with the ``Experiment.create_run_settings()`` function.
This function accepts a `run_command` argument: the command to run the executable.
If `run_command` is set to `auto`, SmartSim will attempt to match a run command on the
system with a RunSettings class in SmartSim. If found, the class corresponding to
that run_command will be created and returned.
If the `run_command` is passed a recognized run command (ex. 'Slurm') the ``RunSettings``
instance will be a child class such as ``SrunSettings``.
If not supported by smartsim, the base RunSettings class will be
created and returned with the specified run_command and run_args will be evaluated literally.

Once a ``RunSettings`` object is initialized, you have access to helper functions that
further configure the run settings of a job. The following chart shows helper functions
associated with the ``SrunSettings`` object. The ``SrunSettings`` object allows
a user running on a slurm based machine to further the job.

.. list-table:: SrunSettings Helper Functions
   :widths: 25 55 25
   :header-rows: 1

   * - ``SrunSettings`` function
     - Example
     - Description
   * - ``set_nodes()``
     - ``SrunSettings.set_nodes(nodes)``
     - Set the number of nodes
   * - ``set_tasks()``
     - ``SrunSettings.set_tasks(tasks)``
     - Set the number of tasks for this job
   * - ``set_tasks_per_node()``
     - ``SrunSettings.set_tasks_per_node(tasks_per_node)``
     - Set the number of tasks for this job
   * - ...
     - ...
     - ...

=====
Local
=====
When running SmartSim on laptops and single node workstations, the base
``RunSettings`` object is used to parameterize jobs.
RunSettings include a run_command parameter for local launches that utilize a parallel launch binary like mpirun, mpiexec, and others.
If no run_command is specified, the executable will be launched locally.

The local `launcher` supports the base :ref:`RunSettings API <rs-api>`
which can be used to run executables as well as run executables
with arbitrary launch binaries like `mpiexec`. The local launcher
is the default launcher for all ``Experiment`` instances.

Initialize
----------
There are two ways to initialize a ``RunSettings`` instance:

Case 1 : Setting the `launcher` argument during ``Experiment`` creation.
    When initializing an ``Experiment`` object, specify the `launcher` argument to `"local"`:

    .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="local")  # local launcher

    Now that the launcher is set to local, when calling ``Experiment.create_run_settings()``,
    the ``RunSettings`` object will be returned:

    .. code-block:: python

      settings = exp.create_run_settings()  # local launcher

Case 2 : Specify `local` to `run_command` during ``RunSettings`` object creation.
    Adversely, you may specify to SmartSim a run command when initializing a run settings object:

    .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="local")  # local launcher

    Pass in string `"mpiexec"` to `run_command` to tell SmartSim to run executable
    with the arbitrary launch binary mpiexec:
    .. code-block:: python

      settings = exp.create_run_settings(run_command="mpirun")  # local launcher

    The ``MpiexecSetting`` child class will be returned.

===
HPC
===
For configuring an entity to launch on a WLM system, SmartSim offers ``RunSettings`` child classes
as follows:

1. The Slurm `launcher` supports four types of ``RunSettings``:
   - ``SrunSettings``
   - ``OrterunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
2. The PBSpro `launcher` supports four types of ``RunSettings``:
   - ``AprunSettings``
   - ``OrterunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
3. The Cobalt `launcher` supports four types of ``RunSettings``:
   - ``AprunSettings``
   - ``OrterunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
4. The LSF `launcher` supports three types of ``RunSettings``:
   - ``JsrunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``

Each `launcher` supports different ``RunSettings`` child
classes as shown below:

Initialize run parameters for a slurm job with srun
SrunSettings should only be used on Slurm based systems.
If an allocation is specified, the instance receiving these run parameters will launch on that allocation.

Case 1 : To use the an HPC launcher such as `Slurm`, specify at Experiment initialization:

    More specifically, specify through the `launcher` argument:

    .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="slurm")  # slurm launcher

    ``SrunSettings`` will be returned

    .. code-block:: python

      settings = exp.create_run_settings()

Case 2 : To use the `run_command` variable, specify at RunSettings initializations

    .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="slurm")  # local launcher

    .. code-block:: python

      settings = exp.create_run_settings(run_command="mpiexec")  # local launcher

    The above will return a ``MpiexecSettings`` object.