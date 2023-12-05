************
Run Settings
************

========
Overview
========
SmartSim allows you to configure the run settings of a job (``Model`` or ``Ensemble``)
through a ``RunSettings`` object. Each SmartSim `launcher` interfaces with a number of
``RunSettings`` objects. For the local `launcher`, SmartSim supports
a base ``RunSettings`` object. For the WLM `launchers`, SmartSim supports a variety of
``RunSettings`` child classes.

* Navigate to :ref:`local section<Local>` to configure run settings locally
* Navigate to :ref:`wlm section<HPC>` to configure run settings for a HPC

A run settings object is initialized through the ``Experiment.create_run_settings()`` function.
This function accepts a `run_command` argument: the command to run the executable.
If `run_command` is set to `auto`, SmartSim will attempt to match a run command on the
system with a ``RunSettings`` class. If found, the class corresponding to
that `run_command` will be created and returned.
If the `run_command` is passed a recognized run command (ex. 'Slurm') the ``RunSettings``
instance will be a child class such as ``SrunSettings``.
If not supported by smartsim, the base ``RunSettings`` class will be
created and returned with the specified `run_command` and `run_args` evaluated literally.
You may also specify `mpi_run` or `mpi_exec` to the `run_command` argument. This will return
the associated child class.

Once a ``RunSettings`` object is initialized, you have access to helper functions that
further configure the run settings of a job. The following chart shows helper functions
associated with the ``SrunSettings`` object:

.. list-table:: SrunSettings Helper Functions
   :widths: 25 55 25
   :header-rows: 1

   * - ``SrunSettings`` func
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
RunSettings include a `run_command` parameter for local launches
that utilize a parallel launch binary like `mpirun`, `mpiexec`, and others.
If no `run_command` is specified, the executable will be launched locally.

The local `launcher` supports the base :ref:`RunSettings API <rs-api>`
which can be used to run executables as well as run executables
with arbitrary launch binaries like `mpiexec`. The local launcher
is the default launcher for all ``Experiment`` instances.

In the following example, we walk through creating and launching locally a
``Model`` entity with a ``RunSettings`` object.
First specify the `launcher` argument as `"local"`
when initializing the ``Experiment``:

.. code-block:: python

      exp = Experiment("name-of-experiment", launcher="local")

Next, we initialize a simple ``Model`` entity that echos `Hello World` to stdout.
The ``RunSettings`` base object will be created since we set `launcher="local"` when
initializing the experiment.

.. code-block:: python

      run_settings = exp.create_run_settings("echo", exe_args="Hello World")

Finally, create a ``Model`` using ``Experiment.create_model()``. Specify
the ``RunSettings`` object along with a `name` for the model:

.. code-block:: python

      model = exp.create_model("hello_world", run_settings)

Next, launch the Model using ``Experiment.start()``:

.. code-block:: python

      exp.start(model)

In this experiment, there is no need to use the ``Experiment.stop()`` function
since there is no launched orchestrator.

===
HPC
===
To configure an entity for launch on an HPC, SmartSim offers ``RunSettings`` child classes.
If an allocation is specified, the instance receiving these run parameters will launch on that allocation.
Each WLM `launcher` supports different ``RunSettings`` child classes as shown below:

1. The Slurm `launcher` supports:
   - ``SrunSettings``
   - ``OrterunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
2. The PBSpro `launcher` supports:
   - ``AprunSettings``
   - ``OrterunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
3. The Cobalt `launcher` supports:
   - ``AprunSettings``
   - ``OrterunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
4. The LSF `launcher` supports:
   - ``JsrunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``


In the following example, we walk through creating and launching a
``Model`` entity with a slurm ``SrunSettings`` object.
Begin by initializing an ``Experiment`` object
and specifying the Slurm WLM to the `launcher` argument:

.. code-block:: python

      exp = Experiment("name-of-experiment", launcher="local")

Next, we initialize a simple ``Model`` entity that echos `Hello World` to stdout.
The ``SrunSettings`` child class object will be created since we set `launcher="slurm"` when
initializing the experiment.

.. code-block:: python

      srun_settings = exp.create_run_settings("echo", exe_args="Hello World")

Finally, create a ``Model`` using ``Experiment.create_model()``. Specify
the ``SrunSettings`` object along with a `name` for the model:

.. code-block:: python

      model = exp.create_model("hello_world", srun_settings)

Next, launch the Model using ``Experiment.start()``:

.. code-block:: python

      exp.start(model)

In this experiment, there is no need to use the ``Experiment.stop()`` function
since there is no launched orchestrator.