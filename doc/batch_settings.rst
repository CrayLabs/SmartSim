**************
Batch Settings
**************

========
Overview
========
SmartSim offers functionality to launch entities (``Model`` or ``Ensemble``)
as batch jobs. Each SmartSim `launcher` interfaces with a ``BatchSettings`` object
specific to a systems Workload Manager (WLM). The following ``BatchSettings`` child
classes are provided per `launcher`:

1. The Slurm `launcher` supports:
   - ``SbatchSettings``
2. The PBSpro `launcher` supports:
   - ``QsubBatchSettings``
3. The Cobalt `launcher` supports:
   - ``CobaltBatchSettings``
4. The LSF `launcher` supports:
   - ``BsubBatchSettings``

The local `launcher` does not support batch launching.
Each above child class allow you to specify the run parameters for a WLM batch job.

Once a ``BatchSettings`` object is initialized via the ``Experiment.create_batch_settings()``
function, you have access to helper functions associated with the helper class that
configure the batch settings the job. The following chart illustrates helper functions
associated with the ``SrunSettings`` object. The ``SbatchSettings`` object is used for
launching batches on Slurm WLM systems.

.. list-table:: SbatchSettings Helper Functions
   :widths: 25 55 25
   :header-rows: 1

   * - ``SbatchSettings`` func
     - Example
     - Description
   * - ``set_account()``
     - ``SbatchSettings.set_account(account)``
     - Set the account for this batch job
   * - ``set_batch_command()``
     - ``SbatchSettings.set_batch_command(command)``
     - Set the command used to launch the batch e.g.
   * - ``set_nodes()``
     - ``SbatchSettings.set_nodes(num_nodes)``
     - Set the number of nodes for this batch job
   * - ...
     - ...
     - ...

In the following :ref:`HPC<HPC>` subsection, we demonstrate configuring a SmartSim entity to launch
as a batch job on a Slurm system.

===
HPC
===
In the following example, we walk through creating and launching a
``Model`` entity with ``BatchSettings`` and ``RunSettings`` objects.
The ``BatchSettings`` object allocates resources for the job while the
``RunSettings`` object feeds SmartSim the run parameters for the job.

Begin by initializing an ``Experiment`` object
and specifying the systems WLM to the `launcher` argument.
In this example, we use the slurm job scheduler:

.. code-block:: python

      exp = Experiment("name-of-experiment", launcher="slurm")

Next, we initialize a simple ``Model`` entity that echos `Hello World` to stdout.
Begin by initializing a ``RunSettings`` object to configure the run parameters of the ``Model``.
Note that since we specified the `launcher` as slurm, the slurm specific ``RunSettings`` object
(``SrunSettings``) will be returned upon initialization:

.. code-block:: python

      srun_settings = exp.create_run_settings("echo", exe_args="Hello World")

Next, initialized a ``BatchSettings`` object. Again, we specified `slurm` as the experiment `launcher`.
Therefore, the child class ``SbatchSettings`` will be returned when we use the ``Experiment.create_batch_settings()``
function. We allocate a single node for one hour and specify Slurm sbatch arguments to batch_args as a dictionary:

.. code-block:: python

      batch_settings = exp.create_batch_settings(nodes=1, time="01:00:00", account="name_of_account", batch_args={"ntasks": 1})

Note that initialization values provided (nodes, time, account) will overwrite the same arguments in batch_args if present.
Next, we further configure the batch settings with the helper functions provided through the ``SbatchSettings`` object,
``batch_settings``:

.. code-block:: python

      batch_settings.set_batch_command("sbatch")
      batch_settings.set_cpus_per_task(5)

Finally, create a ``Model`` using ``Experiment.create_model()``. Specify
the ``SbatchSettings`` and ``SrunSettings`` objects along with a `name` for the model:

.. code-block:: python

      model = exp.create_model("hello_world", batch_settings, run_settings)

Next, launch the Model using ``Experiment.start()``:

.. code-block:: python

      exp.start(model)

In this experiment, there is no need to use the ``Experiment.stop()`` function
since there is no launched orchestrator.