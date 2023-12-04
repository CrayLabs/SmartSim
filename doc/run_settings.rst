************
Run Settings
************

=========
 Overview
=========
Run settings instruct SmartSim how to execute a Model or Ensemble, within an experiment,
on the system and available computational resources. SmartSim offers the top level
``RunSettings`` object to configure a SmartSim entity for a local launch.
For configuring an entity to launch on a WLM system, SmartSim offers ``RunSettings`` child classes
as follows:

* ``SrunSettings``
* ``AprunSettings``
* ``JsrunSettings``
* ``MpirunSettings``
* ``MpiexecrunSettings``
* ``OrterunSettings``

A run settings object is initialized through the ``Experiment.create_run_settings()`` function
which accepts a `run_command` argument. You may pass in your machines run command to the
`run_command` argument.
If `run_command` is set to `auto`, SmartSim will attempt to match a run command on the
system with a RunSettings class in SmartSim. If found, the class corresponding to
that run_command will be created and returned.

* Navigate to the :ref:`local section<Local RunSettings Instance>` to initialize for local settings instance
* Navigate to the :ref:`wlm section<HPC RunSetting Instance>` to initialize for local settings instance

Once initialized, the ``RunSettings`` object and child classes offer functions to configure the
run settings of the entity. The following chart displays the functions associated to
the ``SrunSettings`` object that allows a user running on a `slurm` based machine
to configure set the number of nodes, tasks and tasks per node for the job.

.. list-table:: Slurm Run Settings Functions
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

To view a 
A ``RunSettings`` instance is created through the
``Experiment.create_run_settings()`` function which accepts a `run_command` argument.
If `run_command` is set to `auto`, SmartSim will attempt to match a run command on the
system with a RunSettings class in SmartSim. If found, the class corresponding to
that run_command will be created and returned.

When initializing an ``Experiment`` object and specifying `launcher="local"`, auto detection will be turned off.
If the `run_command` is passed a recognized run command (ex. 'Slurm') the ``RunSettings``
instance will be a child class such as ``SrunSettings``.

If not supported by smartsim, the base RunSettings class will be
created and returned with the specified run_command and run_args will be evaluated literally.

==========================
Local RunSettings Instance
==========================

Local
-----
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

=======================
HPC RunSetting Instance
=======================
SmartSim offers support to run your experiment entity instances
with the following ``RunSettings`` child classes per WLM below:

1. Slurm WLM system
   - ``SrunSettings``
   - ``OrterunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
2. PBSPro WLM system
   - ``AprunSettings``
   - ``OrterunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
3. Cobalt WLM system
   - ``AprunSettings``
   - ``OrterunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
4. LSF WLM system
   - ``JsrunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``

Initialize
----------

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