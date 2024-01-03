**************
Batch Settings
**************
========
Overview
========
SmartSim provides functionality to launch workloads (``Model`` or ``Ensemble``)
as batch jobs via the ``BatchSettings`` base class. The ``BatchSettings`` base class extends
to specialized child classes designed for HPC systems and Workload Managers (WLM) to support
environment specific batch job launches. Each SmartSim `launcher` interfaces with a
``BatchSettings`` subclass specific to a systems WLM.

- The Slurm `launcher` supports:
   - :ref:`SbatchSettings<sbatch_api>`
- The PBSpro `launcher` supports:
   - :ref:`QsubBatchSettings<qsub_api>`
- The LSF `launcher` supports:
   - :ref:`BsubBatchSettings<bsub_api>`

.. note::
      The local `launcher` does not support batch jobs.

After a ``BatchSettings`` instance is created, a user has access to the associated child class feature set
that allows a user to configure the job batch settings.

In the following :ref:`HPC<HPC>` subsection, we demonstrate initializing and configuring a batch settings object
per supported SmartSim `launcher`.

===
HPC
===
A ``BatchSettings`` subclass is created through the ``Experiment.create_batch_settings()``
helper function. When the user initializes the ``Experiment`` at the beginning of the Python driver script,
a `launcher` argument may be specified. SmartSim will register or detect the `launcher` and return the supported child class
upon a call to ``Experiment.create_batch_settings()``. Below are examples of how to initialize a ``BatchSettings`` object per `launcher`.

.. tabs::

    .. group-tab:: Slurm
      To instantiate the ``SbatchSettings`` object that interfaces with the slurm job scheduler, specify
      `launcher="slurm"` when initializing the ``Experiment``. Upon the call to ``create_batch_settings()``
      SmartSim will detect the job scheduler and return the appropriate batch settings object.

        .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher slurm
            exp = Experiment("name-of-experiment", launcher="slurm")

            # Initialize a SbatchSettings object
            sbatch_settings = exp.create_batch_settings(nodes=1, time="10:00:00", batch_args={"ntasks": 1})
            # Set the account for the slurm batch job
            sbatch_settings.set_account("12345-Cray")
            # Set the partition for the slurm batch job
            sbatch_settings.set_queue("default")

      The initialized ``SbatchSettings`` instance can now be pass to a SmartSim entity via the `batch_args` argument.

    .. group-tab:: PBSpro
      To instantiate the ``QsubBatchSettings`` object that interfaces with the slurm job scheduler, specify
      `launcher="pbs"` when initializing the ``Experiment``. Upon the call to ``create_batch_settings()``
      SmartSim will detect the job scheduler and return the appropriate batch settings object.

        .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher pbs
            exp = Experiment("name-of-experiment", launcher="pbs")

            # Initialize a QsubBatchSettings object
            qsub_batch_settings = exp.create_batch_settings(nodes=1, time="10:00:00", batch_args={"ntasks": 1})
            # Set the account for the pbs batch job
            qsub_batch_settings.set_account("12345-Cray")
            # Set the partition for the pbs batch job
            qsub_batch_settings.set_queue("default")

      The initialized ``QsubBatchSettings`` instance can now be pass to a SmartSim entity via the `batch_args` argument.

    .. group-tab:: LSF
      To instantiate the ``BsubBatchSettings`` object that interfaces with the slurm job scheduler, specify
      `launcher="lsf"` when initializing the ``Experiment``. Upon the call to ``create_batch_settings()``
      SmartSim will detect the job scheduler and return the appropriate batch settings object.

        .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher lsf
            exp = Experiment("name-of-experiment", launcher="lsf")

            # Initialize a BsubBatchSettings object
            bsub_batch_settings = exp.create_batch_settings(nodes=1, time="10:00:00", batch_args={"ntasks": 1})
            # Set the account for the lsf batch job
            bsub_batch_settings.set_account("12345-Cray")
            # Set the partition for the lsf batch job
            bsub_batch_settings.set_queue("default")

      The initialized ``BsubBatchSettings`` instance can now be pass to a SmartSim entity via the `batch_args` argument.

.. warning::
      Note that initialization values provided (nodes, time, account) will overwrite the same arguments in `batch_args` if present.