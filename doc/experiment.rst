***********
Experiments
***********

=========
 Overview
=========

1. resource allocation: exp.create_run_settings, exp.create_batch_settings -> created an object -> bc pass into Experiment entities and then pass entities into exp.start, exp.stop
2. job submission: exp.start -> use to launch smartsim entities with the WLM
3. job control: Exp.stop -> cleanup entities launched by exp.start
4. monitoring: exp.get_status -> retrieve status of launched entities, exp.poll -> used for polling the experiment not entities of the exp, exp.summary -> summary of exp, exp.finished -> query if a experiment entitiy has finished

The Experiment API is responsible for job submission, job control, error handling
and cleanup. Experiments are the SmartSim Python user interface to interact with the WLM
for automating the deployment of HPC workflows and distributed, in-memory storage. 

Experiments are the SmartSim Python user interface for automating the deployment of HPC workflows
and distributed, in-memory storage. The Experiment is a factory class 
that is responsible for creating, managing and monitoring the stages of an experiment: 
``Model``, ``Ensemble`` and ``Orchestrator``.
It is also responsible for creating instances ``RunSettings``
and ``BatchSettings`` that are passed to the creation of the stages of an experiment
for the configuring the entities ``Model``, ``Ensemble`` and ``Orchestrator``.

It is also used to interact with the entities associated with the Experiment 
object in the workflow. This interaction includes retrieving the status of these 
entities, polling to monitor the progress of jobs, generating summaries of their 
performance, as well as initiating the start and stop actions when necessary.


The Experiment acts as both a factory class for constructing the stages of an
experiment (``Model``, ``Ensemble``, ``Orchestrator``, etc.) as well as an
interface to interact with the entities created by the experiment.

Users can initialize an :ref:`Experiment <experiment_api>` at the beginning of a
Jupyter notebook, interactive python session, or Python file and use the
``Experiment`` to iteratively create, configure and launch computational kernels
on the system through the specified launcher.

.. |SmartSim Architecture| image:: images/ss-arch-overview.png
  :width: 700
  :alt: Alternative text

|SmartSim Architecture|


The interface was designed to be simple, with as little complexity as possible,
and agnostic to the backend launching mechanism (local, Slurm, PBSPro, etc.).

Defining workflow stages requires the utilization of functions associated
with the ``Experiment`` object. The Experiment object is intended to be instantiated
once and utilized throughout the workflow runtime.

==========
 Launchers
==========

The Experiment API *interfaces* both locally and with four
different job schedulers that SmartSim refers to as `launchers`. `Launchers`
provide SmartSim the ability to construct and execute complex workloads
on HPC systems with schedulers (workload managers) like Slurm, or PBS.

SmartSim currently supports 5 `launchers`:
  1. ``local``: for single-node, workstation, or laptop
  2. ``slurm``: for systems using the Slurm scheduler
  3. ``pbs``: for systems using the PBSpro scheduler
  4. ``cobalt``: for systems using the Cobalt scheduler
  5. ``lsf``: for systems using the LSF scheduler
  6. ``auto``: have SmartSim auto-detect the launcher to use.

If `launcher="auto"` is used, the experiment will attempt to find a launcher
on the system, and use the first one it encounters. If a launcher cannot
be found or no launcher parameter is provided, the default value of
`launcher="local"` will be used.

A `launcher` enables SmartSim to interact with the compute system
programmatically through the Experiment python user interface.
SmartSim users do not have to leave the Jupyter Notebook,
Python REPL, or Python script to launch, query, and interact with their jobs.

To specify a specific launcher, one argument needs to be provided
to the ``Experiment`` initialization.

.. code-block:: python

    from smartsim import Experiment

    exp = Experiment("name-of-experiment", launcher="local")  # local launcher
    exp = Experiment("name-of-experiment", launcher="slurm")  # Slurm launcher
    exp = Experiment("name-of-experiment", launcher="pbs")    # PBSpro launcher
    exp = Experiment("name-of-experiment", launcher="cobalt") # Cobalt launcher
    exp = Experiment("name-of-experiment", launcher="lsf")    # LSF launcher
    exp = Experiment("name-of-experiment", launcher="auto")   # auto-detect launcher


=========
 Entities
=========

The ``Experiment`` object offers three entities that represent stages in the
workflow: ``Orchestrator``, ``Model`` and ``Ensemble``. To configure
the job (entity) execution parameters of the ``Model`` and ``Ensemble``, the ``Experiment``
object offers two additional entities: ``RunSettings`` and ``BatchSettings``.

Workflow Stages:

Orchestrator
^^^^^^^^^^^^
The ``Orchestrator`` is the SmartSim object for the in-memory database.
The ``Orchestrator`` can be used to store and retrieve
data during the course of an experiment and across multiple entities.

Model
^^^^^
``Model(s)`` are subclasses of ``SmartSimEntity(s)`` and are created through the
Experiment API. Models represent any computational kernel. Models are flexible
enough to support many different applications, however, to be used with our
clients (SmartRedis) the application will have to be written in Python, C, C++,
or Fortran.

1. The Experiment is used to create Model instances which represent applications, scripts, or generally a program.
2. An experiment can start and stop a Model and monitor execution.
Ensemble
^^^^^^^^
In addition to a single model, SmartSim has the ability to launch an
``Ensemble`` of ``Model`` applications simultaneously.
Ensembles can be given parameters and permutation strategies that define how the
``Ensemble`` will create the underlying model objects.

Job Parameterization:

RunSettings
^^^^^^^^^^^
When running SmartSim on laptops and single node workstations,
the base ``RunSettings`` object is used to parameterize jobs.
``RunSettings`` include a ``run_command`` parameter for local
launches that utilize a parallel launch binary like
``mpirun``, ``mpiexec``, and others.

BatchSettings
^^^^^^^^^^^^^
``BatchSettings`` is used to configure jobs that should be launched
as a batch on a WLM system.

===========
 Initialize
===========

To *initialize* a ``Experiment`` object, you must specify a `string` name and the systems
`launcher`. For simplicity, we will start on a single host and only
launch single-host jobs, and as such will set the `launcher` argument to `local`.

.. code-block:: python

    from smartsim import Experiment
    from smartsim.log import get_logger

    # Init Experiment and specify to launch locally
    exp = Experiment("name-of-experiment", launcher="local")
    # Init a SmartSim logger
    smartsim_logger = get_logger("tutorial-experiment")

=========
 Starting
=========

Defining workflow stages requires the utilization of functions associated
with the ``Experiment`` object. Here we will demonstrate how to create an Orchestrator
stage using ``Experiment.create_database()``, then launch the database with ``Experiment.start()``.

.. code-block:: python

  # create and start an instance of the Orchestrator database
  db = exp.create_database(db_nodes=1, port=6899, interface="lo")
  # create an output directory for the database log files
  exp.generate(db)
  # start the database
  exp.start(db)
  # log the status of the db
  smartsim_logger(f"Database status: {exp.get_status(db)}")

=========
 Stopping
=========

To clean up, we need to tear down the DB. We do this by stopping the Orchestrator.

.. code-block:: python

  exp.stop(db)
  # log the summary of the experiment
  smartsim_logger(f"{exp.summary()}")