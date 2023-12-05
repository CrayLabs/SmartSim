***********
Experiments
***********

========
Overview
========
SmartSim offers functionality to automate the deployment of loosely-coupled HPC and
AI workflows using an in-memory, AI-enabled, distributed memory store via the
:ref:`Experiment API<experiment_api>`.
The ``Experiment`` is SmartSim's top level object that enables users to create and construct
the components of a workflow. The Experiment API offers three different workflow components:
:ref:`Orchestrator<Orchestrator>`, :ref:`Model<Model>`, and :ref:`Ensemble<Ensemble>`.

Once an experiment entity is initialized, a user has access
to the associated entity interface that enables a user to configure the entity and
retrieve entity information: :ref:`Model API<model_api>`, :ref:`Orchestrator API<orchestrator_api>` and
:ref:`Ensemble API<ensemble_api>`. There is no limit to the number of entities a user can
initialize within an experiment.

.. figure:: images/Experiment.png

  Sample experiment showing a user application leveraging
  machine learning infrastructure launched by SmartSim and connected
  to online analysis and visualization via the in-memory database.

=========
Launchers
=========
The components of the workflow can be executed locally or via different job schedulers.
`Launchers` provide SmartSim the ability to construct and execute complex workloads
on HPC systems with job schedulers (workload managers) like Slurm, or PBS.

SmartSim currently supports 6 `launchers`:
  1. ``local``: for single-node, workstation, or laptop
  2. ``slurm``: for systems using the Slurm scheduler
  3. ``pbs``: for systems using the PBSpro scheduler
  4. ``cobalt``: for systems using the Cobalt scheduler
  5. ``lsf``: for systems using the LSF scheduler
  6. ``auto``: have SmartSim auto-detect the launcher to use

If a launcher is not specified, it will default to `"local"`.

.. compound::
  By default, SmartSim will attempt to choose the correct
  launcher for the system, however the user can also choose
  a specific launcher by passing in a value for launcher optional argument.
  For example, to set up a local launcher, the workflow should be initialized as follows:

  .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="local")  # local launcher

  To have SmartSim attempt to find a launcher on your machine, set the `launcher`
  argument to `"auto"` during ``Experiment`` initialization like below:

  .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="auto")  # auto-detect launcher

  If a launcher cannot be found or no `launcher` parameter is provided, the default value of
  `launcher="local"` will be used.

========
Entities
========
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
     - :ref:`Orchestrator <orchestrator_api>`
   * - ``create_model()``
     - ``model = exp.create_model(name, run_settings)``
     - :ref:`Model <model_api>`
   * - ``create_ensemble()``
     - ``ensemble = exp.create_ensemble(name[, params, ...])``
     - :ref:`Ensemble <ensemble_api>`

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
The orchestrator is an in-memory database with features designed
to enable a wide variety of AI-enabled workflows, including features
for online training, low-latency inference, cross-application data
exchange, online interactive visualization, online data analysis, computational
steering, and more. The ``Orchestrator`` can be thought of as a general
feature store capable of storing numerical data, ML models, and scripts.
The orchestrator is capable of performing inference and script evaluation on feature
store data. Any SmartSim ``Model`` or ``Ensemble`` model can connect to the
``Orchestrator`` via the :ref:`SmartRedis<SmartRedis Client Library Hook>`
client library to transmit data, execute ML models, and execute scripts.

**SmartSim offers two types Orchestrator deployments:**

* :ref:`Standard Orchestrator <Standard Orchestrator>`
* :ref:`Colocated Orchestrator <Colocated Orchestrator>`

Standard Orchestrator
^^^^^^^^^^^^^^^^^^^^^^
The standard orchestrator can be deployed on a single compute
node or can be sharded (distributed) over multiple nodes.
The multiple compute hosts memory can be used together to store data.
Users do not need to know how the data is stored in a clustered
configuration and can address the cluster with a SmartRedis client
like a single block of memory using simple put/get semantics in SmartRedis.
The database shards communicate with each other via TCP/IP in the driver script and application.

Clustered Deployment Diagram
""""""""""""""""""""""""""""
During clustered deployment, a SmartSim ``Model`` (the application) runs on separate
compute node(s) from the database node(s).
A clustered database is optimal for high data throughput scenarios
such as online analysis, training and processing.

The following image illustrates communication
between a clustered orchestrator and a
multi-node model. In the Diagram, an instance of the application is
running on each application compute node. A single SmartRedis Client object is initialized with
the clustered database address and used to communicate with the application's compute nodes.
Data is streamed from the application compute nodes to the sharded database via the client.

.. figure::  images/clustered-orc-diagram.png

Initialize a Standard Orchestrator
"""""""""""""""""""""""""""""""""""
To create an ``Orchestrator`` that does not share compute resources with other
SmartSim entities, use the ``Experiment.create_database()`` factory method.
Specifying 1 for the `db_nodes` parameter causes the database to
be single-sharded; otherwise it is multi-shared.
This factory method returns an initialized ``Orchestrator`` object that
gives you access to functions associated with the :ref:`Orchestrator API<orchestrator_api>`.

Colocated Orchestrator
^^^^^^^^^^^^^^^^^^^^^^
An ``Orchestrator`` can be created to share the compute node(s)
and resources with a SmartSim ``Model``. In this case, the database
is deployed on the same compute hosts as a Model instance
defined by the user. In this deployment, the database is not connected
together in a cluster and each shard of the database is addressed
individually by the processes running on that compute host.
Essentially, this means that you have N independent databases,
where N is the number of compute nodes assigned to the application.
The colocated deployment strategy for the Orchestrator
is ideal for use cases where a SmartSim ``Model`` is run on a compute node
that has hardware accelerators (e.g. GPUs) and low-latency inference is
a critical component of the workflow.

Colocated Deployment Diagram
""""""""""""""""""""""""""""
During colocated deployment, a SmartSim orchestrator (the database) runs on the same
compute node as a Smartsim model (the application).
This type of deployment is optimal for high data inference scenarios.

Below is an image illustrating communication
between a colocated model spanning multiple compute nodes, and the database
running on each application compute node. A single SmartRedis client is initialized
for the colocated Orchestrator and is used to communicate with the application.
Data is streamed from the application to the database via the client on the same node.

.. figure:: images/co-located-orc-diagram.png

Initialize a Colocated Orchestrator
"""""""""""""""""""""""""""""""""""
To create an ``Orchestrator`` that shares compute resources with a ``Model``
SmartSim entity, use the ``model.colocate_db()`` factory method.
In this case, the database
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
orchestrator(s) from application client code. This is particularly
useful in instances where an orchestrator is colocated with a SmartSim
model for low-latency inference and another Orchestrator is launched to
handle other aspects of the workflow such as visualization and ML model
training. More detailed information on the ideal use cases for orchestrator(s)
and co-located ``Orchestrator(s)`` are available in sections... (update this when use cases added)

Model
-----
``Model(s)`` represent any computational kernel, including applications,
scripts, or generally, a program.
They can interact with other
SmartSim entities via data transmitted to/from SmartSim Orchestrators
using a SmartRedis client.
Models in PT, TF, and ONNX (scikit-learn, spark, and others) can be
written in Python and called from Fortran or any other client languages.
The Python code executes in a C runtime without the Python interpreter.

Create a Model
^^^^^^^^^^^^^^
A ``Model`` is created through the factory method: ``Experiment.create_model()``.
Models are initialized via ``RunSettings`` objects that specify
how a kernel should be executed with regard to the workload manager
(e.g., Slurm) and the available compute resources on the system.
Optionally, the user may also specify a ``BatchSettings`` object if
the model should be launched as a batch job on the WLM system.
The ``create_model()`` factory method returns an initialized ``Model`` object that
gives you access to functions associated with the :ref:`Model API<model_api>`.

Ensemble
--------
In addition to a single model, SmartSim allows users to create,
configure, and launch an ``Ensemble`` of ``Model`` objects.
Ensembles can be given parameters and permutation strategies that define how the
``Ensemble`` will create the underlying model objects.

If each of multiple ensemble members attempt to use the
same code to access their respective models in the Orchestrator,
the keys by which they do this will overlap and they can end up
accessing each others’ data inadvertently. To prevent
this situation, the SmartSim Entity object supports
key prefixing, which automatically prepends the name
of the model to the keys by which it is accessed. With
this enabled, key overlapping is no longer an issue and
ensemble members can use the same code.

Create a Ensemble
^^^^^^^^^^^^^^^^^
An ``Ensemble`` is created through the factory method: ``Experiment.create_ensemble()``.
To create an ensemble, follow one of the cases below:

Case 1 : Launch in previously obtained interactive allocation.
    A ``RunSettings`` object and `params` or `replicas` are required.
    At launch, the Ensemble will look for interactive
    allocations to launch models in.
    A `replicas` argument or a `params` argument
    is required to expand parameters into ``Model`` instances.

Case 2 : Launch as a batch job.
    A ``BatchSettings`` object is required.
    If passed BatchSettings without other arguments,
    an empty Ensemble will be created that ``Model`` objects
    can be added to manually. All ``Model`` objects added to
    the Ensemble will be launched in a single batch.

Case 3 : Launch as batch and configure individual ``Model`` instances.
    A ``BatchSettings``, ``RunSettings``, and `params` or `replicas`
    are required.
    If it passed ``BatchSettings`` and ``RunSettings``, the ``BatchSettings`` will
    determine the allocation settings for the entire batch, and the ``RunSettings``
    will determine how each individual Model instance is executed within that batch.
    A `replicas` argument or a `params` argument
    is required to expand parameters into ``Model`` instances.

The ``create_ensemble()`` factory method returns an initialized ``Ensemble`` object that
gives you access to functions associated with the :ref:`Ensemble API<ensemble_api>`.

==================
Experiment Example
==================
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
  and the system launcher with which we will execute the driver script.
  Here, we are running the example on a Slurm machine and as such will
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
  and detect your machine's `interface`.

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
  that echos `Hello World` to stdout.

  .. code-block:: python

      settings = exp.create_run_settings("echo", exe_args="Hello World")
      model = exp.create_model("hello_world", settings)

  Notice above we creating the ``Model`` through the ``Experiment.create_model()``
  function. We specify a `name` and the ``RunSettings`` object we created.


Starting
--------
.. compound::
  Next we will launch the components of the experiment (``Orchestrator`` and ``Model``) using functions
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
  the summary of the workflow.

.. note::
  Failure to tear down the Orchestrator at the end of an experiment
  may lead to Orchestrator launch failures if another experiment is
  started on the same node.