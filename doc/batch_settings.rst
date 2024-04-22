.. _batch_settings_doc:

**************
Batch Settings
**************
========
Overview
========
SmartSim provides functionality to launch entities (``Model`` or ``Ensemble``)
as batch jobs supported by the ``BatchSettings`` base class. While the ``BatchSettings`` base
class is not intended for direct use by users, its derived child classes offer batch
launching capabilities tailored for specific workload managers (WLMs). Each SmartSim
`launcher` interfaces with a ``BatchSettings`` subclass specific to a system's WLM:

- The Slurm `launcher` supports:
   - :ref:`SbatchSettings<sbatch_api>`
- The PBS Pro `launcher` supports:
   - :ref:`QsubBatchSettings<qsub_api>`
- The LSF `launcher` supports:
   - :ref:`BsubBatchSettings<bsub_api>`

.. note::
      The local `launcher` does not support batch jobs.

After creating a ``BatchSettings`` instance, users gain access to the methods
of the associated child class, providing them with the ability to further configure the batch
settings for jobs.

In the following :ref:`Examples<batch_settings_ex>` subsection, we demonstrate the initialization
and configuration of a batch settings object.

.. _batch_settings_ex:

========
Examples
========
A ``BatchSettings`` child class is created using the ``Experiment.create_batch_settings``
factory method. When the user initializes the ``Experiment`` at the beginning of the Python driver script,
they may specify a `launcher` argument. SmartSim will then register or detect the `launcher` and return the
corresponding supported child class when ``Experiment.create_batch_settings`` is called. This
design allows SmartSim driver scripts utilizing ``BatchSettings`` to be portable between systems,
requiring only a change in the specified `launcher` during ``Experiment`` initialization.

Below are examples of how to initialize a ``BatchSettings`` object per `launcher`.

.. tabs::

    .. group-tab:: Slurm
      To instantiate the ``SbatchSettings`` object, which interfaces with the Slurm job scheduler, specify
      `launcher="slurm"` when initializing the ``Experiment``. Upon calling ``create_batch_settings``,
      SmartSim will detect the job scheduler and return the appropriate batch settings object.

      .. code-block:: python

          from smartsim import Experiment

          # Initialize the experiment and provide launcher Slurm
          exp = Experiment("name-of-experiment", launcher="slurm")

          # Initialize a SbatchSettings object
          sbatch_settings = exp.create_batch_settings(nodes=1, time="10:00:00")
          # Set the account for the slurm batch job
          sbatch_settings.set_account("12345-Cray")
          # Set the partition for the slurm batch job
          sbatch_settings.set_queue("default")

      The initialized ``SbatchSettings`` instance can now be passed to a SmartSim entity
      (``Model`` or ``Ensemble``) via the `batch_settings` argument in ``create_batch_settings``.

      .. note::
        If `launcher="auto"`, SmartSim will detect that the ``Experiment`` is running on a Slurm based
        machine and set the launcher to `"slurm"`.

    .. group-tab:: PBS Pro
      To instantiate the ``QsubBatchSettings`` object, which interfaces with the PBS Pro job scheduler, specify
      `launcher="pbs"` when initializing the ``Experiment``. Upon calling ``create_batch_settings``,
      SmartSim will detect the job scheduler and return the appropriate batch settings object.

        .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher PBS Pro
            exp = Experiment("name-of-experiment", launcher="pbs")

            # Initialize a QsubBatchSettings object
            qsub_batch_settings = exp.create_batch_settings(nodes=1, time="10:00:00")
            # Set the account for the PBS Pro batch job
            qsub_batch_settings.set_account("12345-Cray")
            # Set the partition for the PBS Pro batch job
            qsub_batch_settings.set_queue("default")

      The initialized ``QsubBatchSettings`` instance can now be passed to a SmartSim entity
      (``Model`` or ``Ensemble``) via the `batch_settings` argument in ``create_batch_settings``.

      .. note::
        If `launcher="auto"`, SmartSim will detect that the ``Experiment`` is running on a PBS Pro based
        machine and set the launcher to `"pbs"`.

    .. group-tab:: LSF
      To instantiate the ``BsubBatchSettings`` object, which interfaces with the LSF job scheduler, specify
      `launcher="lsf"` when initializing the ``Experiment``. Upon calling ``create_batch_settings``,
      SmartSim will detect the job scheduler and return the appropriate batch settings object.

        .. code-block:: python

            from smartsim import Experiment

            # Initialize the experiment and provide launcher LSF
            exp = Experiment("name-of-experiment", launcher="lsf")

            # Initialize a BsubBatchSettings object
            bsub_batch_settings = exp.create_batch_settings(nodes=1, time="10:00:00", batch_args={"ntasks": 1})
            # Set the account for the lsf batch job
            bsub_batch_settings.set_account("12345-Cray")
            # Set the partition for the lsf batch job
            bsub_batch_settings.set_queue("default")

      The initialized ``BsubBatchSettings`` instance can now be passed to a SmartSim entity
      (``Model`` or ``Ensemble``) via the `batch_settings` argument in ``create_batch_settings``.

      .. note::
        If `launcher="auto"`, SmartSim will detect that the ``Experiment`` is running on a LSF based
        machine and set the launcher to `"lsf"`.

.. warning::
      Note that initialization values provided (e.g., `nodes`, `time`, etc) will overwrite the same arguments in `batch_args` if present.