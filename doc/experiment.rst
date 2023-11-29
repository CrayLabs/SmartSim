***********
Experiments
***********

=========
 Overview
=========
SmartSim offers functionality to automate the deployment of HPC workloads and distributed,
in-memory storage via the :ref:`Experiment API<experiment_api>`.
The ``Experiment`` is SmartSims Python user interface that enables users to create and construct
the stages of a workflow. The Experiment API offers three different workflow stages:
:ref:`Orchestrator<Orchestrator>`, :ref:`Model<Model>`, and :ref:`Ensemble<Ensemble>`.

Once an experiment entity is initialized, a user has access
to the associated entity interface that enables a user to configure the entity and
retrieve entity information: :ref:`Model API<mode_api>`, :ref:`Orchestrator API<orc_api>` and
:ref:`Ensemble API<ensem_api>`. There is no limit to the number of stages a user can
initialize within an experiment.

The Experiment controls launching, monitoring
and stopping all entities.
Users can initialize an :ref:`Experiment <experiment_api>` at the beginning of a
Jupyter notebook, interactive python session, or Python file and use the
``Experiment`` to iteratively create, configure and launch computational kernels
via entities on the system through the specified launcher.
The interface was designed to be simple, with as little complexity
as possible, and agnostic to the backend launching mechanism
(local, Slurm, PBSPro, etc.).

.. figure:: images/Experiment.png

  Sample experiment showing a user application leveraging
  machine learning infrastructure launched by SmartSim and connected
  to online analysis and visualization via the in-memory database.

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
Entities are SmartSim API objects that can be launched and
managed on the compute system via the Experiment API. While the
``Experiment`` object is intended to be instantiated once in a
SmartSim driver script, there is no limit to the number of SmartSim entities
within an Experiment. In the following subsections, we define the
general purpose of the three entities that can be created via
Experiment API factory methods:

* ``Orchestrator``
* ``Model``
* ``Ensemble``

To create a reference to each entity object, use the associated
``Experiment.create_...()`` function.

.. list-table:: Experiment API Entity Creation
   :widths: 20 65 25
   :header-rows: 1

   * - Factory Method
     - Example
     - Return Type
   * - ``create_database()``
     - ``orch = exp.create_database([port, db_nodes, ...])``
     - :ref:`Orchestrator <orc_api>`
   * - ``create_model()``
     - ``model = exp.create_model(name, run_settings)``
     - :ref:`Model <mode_api>`
   * - ``create_ensemble()``
     - ``ensemble = exp.create_ensemble(name[, params, ...])``
     - :ref:`Ensemble <ensem_api>`

Each entity instance can be used to start,
monitor, and stop simulations from the notebook
using the :ref:`Experiment API<experiment_api>`.

.. list-table:: Interact with Entities during the Experiment
   :widths: 25 55 25
   :header-rows: 1

   * - Experiment Function
     - Example
     - Desc
   * - ``start()``
     - ``exp.start(*args[, block, summary, ...])``
     - Launch an Entity
   * - ``stop()``
     - ``exp.stop(*args)``
     - Clobber an Entity
   * - ``get_status()``
     - ``exp.get_status(*args)``
     - Retrieve Entity Status

Orchestrator
------------
The ``Orchestrator`` is an in-memory database with features designed
to enable a wide variety of AI-enabled workflows, including features
for online training, low-latency inference, cross-application data
exchange, online interactive visualization, online data analysis, computational
steering, and more. The ``Orchestrator`` can be thought of as a general
feature store capable of storing numerical data, ML models, and scripts
and capable of performing inference and script evaluation on feature
store data. Any SmartSim ``Model`` or ``Ensemble`` model can connect to the
``Orchestrator`` via the :ref:`SmartRedis<SmartRedis Client Library Hook>`
client library to transmit data, execute ML models, and execute scripts.

**SmartSim offers two types Orchestrator deployments:**

* :ref:`Clustered Orchestrator <Clustered Orchestrator>`
* :ref:`Colocated Orchestrator <Colocated Orchestrator>`

Clustered Orchestrator
^^^^^^^^^^^^^^^^^^^^^^
The ``Orchestrator`` can be composed of one or more in-memory database shards that are spread
across one or more compute nodes.
The multiple compute hosts memory can be used together to store data.
Users do not need to know how the data is stored in a clustered
configuration and can address the cluster with a SmartRedis client
like a single block of memory using simple put/get semantics in SmartRedis.
The database shards communicate with each other via TCP/IP in the driver script and application.
SmartRedis will ensure that data is evenly distributed among all nodes in the cluster.

Clustered Deployment Diagram
""""""""""""""""""""""""""""
During clustered deployment, a SmartSim ``Model`` (the application) runs on separate
compute node(s) from the database node(s).
A clustered database is optimal for high data throughput scenarios
such as online analysis, training and processing.

Below is an image illustrating communication
between a clustered ``Orchestrator`` and a
multi-node ``Model``. In the Diagram, an instance of the application is
running on each application compute node. A single SmartRedis client is initialized with
the clustered database address and used to communicate with the application's compute nodes.
Data is streamed from the application compute nodes to the sharded database via the client.

.. figure::  images/clustered-orc-diagram.png

Initialize a Clustered Orchestrator
"""""""""""""""""""""""""""""""""""
To create an orchestrator that does not share compute resources with other
SmartSim entities, use the ``Experiment.create_database()`` factory method.
Specifying the parameter `db_nodes` as greater than or equal to 1 will determine
whether your database is multi-sharded or single-sharded.
This factory method returns an initialized ``Orchestrator`` object that
gives you access to functions associated with the :ref:`Orchestrator API<orc_api>`.

Colocated Orchestrator
^^^^^^^^^^^^^^^^^^^^^^
An ``Orchestrator`` can be created to share the compute node(s)
and resources with a SmartSim ``Model``. In this case, the Orchestrator
is deployed on the same compute hosts as a Model instance
defined by the user. In this deployment, the database is not connected
together in a cluster and each shard of the database is addressed
individually by the processes running on that compute host.
If the SmartSim ``Model`` spans more than one
compute node, the colocated database will also span all of the
compute nodes. The colocated deployment strategy for the Orchestrator
is ideal for use cases where a SmartSim ``Model`` is run on a compute node
that has hardware accelerators (e.g. GPUs) and low-latency inference is
a critical component of the workflow.

Colocated Deployment Diagram
""""""""""""""""""""""""""""
During colocated deployment, a SmartSim ``Orchestrator`` (the database) runs on the same
compute node(s) as a Smartsim ``Model`` (the application).
This type of deployment is optimal for high data inference scenarios.

Below is an image illustrating communication
between a colocated ``Model`` spanning multiple compute nodes, and the ``Orchestrator``
running on each application compute node. A single SmartRedis client is initialized
for the colocated Orchestrator and is used to communicate with the application.
Data is streamed from the application to the database via the client on the same node.

.. figure:: images/co-located-orc-diagram.png

Initialize a Colocated Orchestrator
"""""""""""""""""""""""""""""""""""
To create an orchestrator that shares compute resources with a ``Model``
SmartSim entity, use the ``model.colocate_db()`` factory method.
In this case, the Orchestrator
is created via the SmartSim Model API function ``model.colocate_db``.
The :ref:`Model API<model_api>` is accessed once a ``Model`` object has been initialized.


Multi-db support
^^^^^^^^^^^^^^^^
SmartSim supports multi-database functionality, enabling an experiment
to have several concurrently launched ``Orchestrator(s)``. If there is
a need to launch more than one ``Orchestrator``, the ``Experiment.create_database()``
function mandates the specification of a unique database identifier,
denoted by the `db_identifier` argument, per created orchestrator.

The `db-identifier` is used to reference SmartSim
``Orchestrator(s)`` from application client code. This is particularly
useful in instances where an ``Orchestrator`` is colocated with a SmartSim
model for low-latency inference and another Orchestrator is launched to
handle other aspects of the workflow such as visualization and ML model
training. More detailed information on the ideal use cases for ``Orchestrator(s)``
and co-located ``Orchestrator(s)`` are available in sections... (link)

Model
-----
``Model(s)`` represent any computational kernel, including applications,
scripts, or generally, a program.
They can interact with other
SmartSim entities via data transmitted to/from SmartSim Orchestrators
using a SmartRedis client.
Models in PT, TF, and ONNX (scikit-learn, spark, and others) can be
written in Python and called from Fortran or any other client languages.
The Python code executes in a C runtime without the python interpreter.

Create a Model
^^^^^^^^^^^^^^
A ``Model`` is created through the function: ``Experiment.create_model()``.
For initialization, models require ``RunSettings`` objects that specify
how a kernel should be executed with regard to the workload manager
(e.g., Slurm) and the available compute resources on the system.
Optionally, the user may also specify a ``BatchSettings`` object if
the model should be launched as a batch on the WLM system.
The ``create_model()`` factory method returns an initialized ``Model`` object that
gives you access to functions associated with the :ref:`Model API<mode_api>`.

Ensemble
--------
In addition to a single model, SmartSim offers the ability to run an
``Ensemble`` of ``Model`` applications, i.e. multiple replicas of the simulation.
More specifically, you can create, configure and launch groups of workloads (Ensembles)
within the Experiment.
Ensembles can be given parameters and permutation strategies that define how the
``Ensemble`` will create the underlying model objects.

Create a Ensemble
^^^^^^^^^^^^^^^^^
An ensemble is created through the function: ``Experiment.create_ensemble()``. The function requires
one of the subsequent sets of arguments upon initialization:

Case 1 : ``RunSettings`` and `params` or `replicas`
    If it only passed RunSettings, Ensemble, objects will
    require either a replicas argument or a params argument to
    expand parameters into Model instances.
    At launch, the Ensemble will look for interactive allocations to launch models in.

Case 2 : ``BatchSettings``
    If it passed BatchSettings without other arguments,
    an empty Ensemble will be created that Model objects
    can be added to manually. All Model objects added to
    the Ensemble will be launched in a single batch.

Case 3 : ``BatchSettings``, `run_settings`, and `params`
    If it passed BatchSettings and RunSettings, the BatchSettings
    will determine the allocation settings for the entire batch,
    and the RunSettings will determine how each individual Model
    instance is executed within that batch.

Case 4 : ``BatchSettings``, ``RunSettings``, and `replicas`
    If each of multiple ensemble members attempt to use the
    same code to access their respective models in the Orchestrator,
    the keys by which they do this will overlap and they can end up
    accessing each others’ data inadvertently. To prevent
    this situation, the SmartSim Entity object supports
    key prefixing, which automatically prepends the name
    of the model to the keys by which it is accessed. With
    this enabled, key overlapping is no longer an issue and
    ensemble members can use the same code.

The ``create_ensemble()`` factory method returns an initialized ``Ensemble`` object that
gives you access to functions associated with the :ref:`Ensemble API<ensem_api>`.

===================
 Experiment Example
===================
.. compound::
  In the following subsections, we provide an example of using SmartSim to automate the
  deployment of an HPC workload and distributed, in-memory storage, within
  the workflow.

  Continue to the example to:

  .. list-table:: Experiment example contents
   :widths: auto
   :header-rows: 1

   * - Initialize
     - Start
     - Stop
   * - a workflow (``Experiment``)
     - the in-memory database (``Orchestrator``)
     - the in-memory database (``Orchestrator``)
   * - a in-memory database (``Orchestrator``)
     - the workload (``Model``)
     - 
   * - a workload (``Model``)
     - 
     - 

Initialize
----------
.. compound::
  To create a workflow, we *initialize* an ``Experiment`` object
  once at the beginning of the Python driver script.
  To create an Experiment, we specify a name
  and the system launcher of which we will execute the driver script on.
  We are running the example on a Slurm machine and as such will
  set the `launcher` argument to `slurm`.

  .. code-block:: python

      from smartsim import Experiment
      from smartsim.log import get_logger

      # Initialize an Experiment
      exp = Experiment("name-of-experiment", launcher="slurm")
      # Initialize a SmartSim logger
      smartsim_logger = get_logger("tutorial-experiment")

  We also initialize a SmartSim logger. We will use the logger throughout the experiment
  to monitor the entities.

.. compound::
  Next, we will launch a SmartSim in-memory database called an ``Orchestrator``.
  To *initialize* an ``Orchestrator`` object, use the ``Experiment.create_database()``
  function. We will create a single-sharded database and therefore will set
  the argument `db_nodes` to 1. SmartSim will assign a `port` to the database
  and detect your machines `interface`.

  .. code-block:: python

      # Initialize an Orchestrator
      database = exp.create_database(db_nodes=1)
      # Create an output directory
      exp.generate(database)

  We use the ``Experiment.generate()`` function to create an
  output directory for the database log files.

.. compound::
  Next, we create a workload within the experiment.
  We begin by *initializing* a ``Model`` object.
  To create a ``Model``, we must instruct SmartSim how we would
  like to execute the workload by passing in a ``RunSettings``` object.
  We create a RunSettings object using the
  ``Experiment.create_run_settings()`` function.
  We specify the executable to run and the arguments to pass to
  the executable. The example workload is a simple `Hello World` program
  that `echos` `Hello World` to stdout.

  .. code-block:: python

      settings = exp.create_run_settings("echo", exe_args="Hello World")
      model = exp.create_model("hello_world", settings)

  Notice above we creating the ``Model`` through the ``Experiment.create_model()``
  function. We specify a `name` and the ``RunSettings`` object we created.


Starting
--------
.. compound::
  Next we will launch the stages of our experiment (``Orchestrator`` and ``Model``) using functions
  provided by the ``Experiment`` API. To do so, we will use
  the ``Experiment.start()`` function and pass in the ``Orchestrator``
  and ``Model`` instance previously created.

  .. code-block:: python

    # Launch the Orchestrator and Model instance
    exp.start(database, model)
    # log the status of the db
    exp.get_status(database)
    exp.get_status(model)

  Notice above we use the ``Experiment.get_status()`` function to query the
  status of launched instances.


Stopping
--------
.. compound::
  Lastly, to clean up the experiment, we need to tear down the launched database.
  We do this by stopping the Orchestrator using the ``Experiment.stop()`` function.

  .. code-block:: python

    exp.stop(db)
    # log the summary of the experiment
    exp.summary()

  Notice that we use the ``Experiment.summary()`` function to print
  the summary of our workflow.