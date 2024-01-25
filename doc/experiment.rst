***********
Experiments
***********
========
Overview
========
SmartSim helps automate the deployment of AI-enabled workflows on HPC systems. With SmartSim, users
can describe and launch combinations of applications and AI/ML infrastructure to produce novel and
scalable workflows. SmartSim supports launching these workflows on a diverse set of systems, including
local environments such as Mac or Linux, as well as HPC job schedulers (e.g. Slurm, PBS, and LSF).

The Experiment API is SmartSim's top level API that provides users with methods for creating, combining,
configuring, launching and monitoring :ref:`entities<entities_exp_docs>` in an AI-enabled workflow. More specifically, the
Experiment API offers three customizable workflow components that are created and initialized via factory
methods:

1. :ref:`Orchestrator<orchestrator_exp_docs>`
2. :ref:`Model<model_exp_docs>`
3. :ref:`Ensemble<ensemble_exp_docs>`

Settings are given to ``Model`` and ``Ensemble`` objects to provide parameters for how a job should be executed. The
:ref:`Experiment API<experiment_api>` offers customizable settings objects that are created and initialized via factory
methods:

1. :ref:`RunSettings<run_settings_doc>`
2. :ref:`BatchSettings<batch_settings_doc>`

Once a workflow component is initialized, a user has access
to the associated entity API that supports configuring and
retrieving entity information:

* :ref:`Orchestrator API<orchestrator_api>`
* :ref:`Model API<model_api>`
* :ref:`Ensemble API<ensemble_api>`

There is no limit to the number of entities a user can
initialize within an experiment.

.. figure:: images/Experiment.png

  Sample experiment showing a user application leveraging
  machine learning infrastructure launched by SmartSim and connected
  to online analysis and visualization via the in-memory ``Orchestrator``.

.. _launcher_exp_docs:
=========
Launchers
=========
SmartSim supports launching AI-enabled workflows on a wide variety of systems, including locally on a Mac or
Linux machine or on HPC machines with a job scheduler (e.g. Slurm, PBS, and LSF). When creating a SmartSim
``Experiment``, the user has the opportunity to specify the `launcher` type or defer to automatic `launcher` selection.
`Launcher` selection determines how SmartSim translates SmartSim entity configurations into system calls to launch,
manage, and monitor entities. Currently, SmartSim supports 6 `launchers`:

1. ``local``: for single-node, workstation, or laptop
2. ``slurm``: for systems using the Slurm scheduler
3. ``pbs``: for systems using the PBSpro scheduler
4. ``pals``: for systems using the PALS scheduler
5. ``cobalt``: for systems using the Cobalt scheduler
6. ``lsf``: for systems using the LSF scheduler
7. ``auto``: have SmartSim auto-detect the launcher to use

If a `launcher` is not specified, SmartSim will default to `"local"` which will start all ``Experiment`` launched
entities on the localhost.

.. compound::
  For example, to set up a slurm `launcher`, the workflow should be initialized as follows:

  .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="slurm")  # slurm launcher

  To instruct SmartSim to search for a `launcher` on your machine, set the `launcher`
  argument to `"auto"` during ``Experiment`` initialization as shown below:

  .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="auto")  # auto-detect launcher

  If the systems `launcher` cannot be found or no `launcher` argument is provided, the default value of
  `launcher="local"` will be used.

.. _entities_exp_docs:
========
Entities
========
Entities are SmartSim API objects that can be launched and
managed on the compute system via the Experiment API. While the
``Experiment`` object is intended to be instantiated once in a
SmartSim driver script, there is no limit to the number of SmartSim entities
within an ``Experiment``. In the following subsections, we define the
general purpose of the three entities that can be created via
Experiment API factory methods:

* ``Orchestrator``
* ``Model``
* ``Ensemble``

To create a reference to a newly instantiated entity object, use the associated
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

After initialization via the ``Experiment`` factory methods, each entity can be started, monitored, and stopped
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
     - Stop an Entity
   * - ``get_status()``
     - ``exp.get_status(*args)``
     - Retrieve Entity Status

.. _orchestrator_exp_docs:
Orchestrator
============
The ``Orchestrator`` is an in-memory database with features built for
a wide variety of AI-enabled workflows, including features
for online training, low-latency inference, cross-application data
exchange, online interactive visualization, online data analysis, computational
steering, and more. The ``Orchestrator`` can be thought of as a general
feature store capable of storing numerical data, ML models, and scripts.
The ``Orchestrator`` is capable of performing inference and script evaluation using data in the feature store.
Any SmartSim ``Model`` or ``Ensemble`` model can connect to the
``Orchestrator`` via the :ref:`SmartRedis<smartredis-api>`
client library to transmit data, execute ML models, and execute scripts.

**SmartSim offers two types Orchestrator deployments:**

* :ref:`Standalone Orchestrator Deployment<standalone_deployment_exp_docs>`
* :ref:`Colocated Orchestrator Deployment<colocated_deployment_exp_docs>`

.. _standalone_deployment_exp_docs:
Standalone Deployment
---------------------
The standard ``Orchestrator`` can be deployed on a single compute
node or can be sharded (distributed) over multiple compute nodes.
With multiple nodes, available hardware for inference and script
evaluation increases and overall memory for data storage increases.
Users do not need to know the number of compute nodes (shards) used for the
``Orchestrator`` when interacting via the SmartRedis Client API;
SmartRedis Client API functions are designed to work with both single and multi-node
configurations.

During standalone deployment, a SmartSim ``Model`` (the application) runs on separate
compute node(s) from the ``Orchestrator`` node(s).
A standalone ``Orchestrator`` is optimal for high data throughput scenarios
such as online analysis, training and processing.

The following image illustrates communication between a standalone ``Orchestrator`` and a
model. In the diagram, the application is running on multiple compute nodes,
separate from the ``Orchestrator`` compute nodes. Connections are established between the
Model application and the standalone deployment using the SmartRedis ``Client``.

.. figure::  images/clustered_orchestrator-1.png

  Sample Standalone Orchestrator Deployment

To create an ``Orchestrator`` that does not share compute resources with other
SmartSim entities, use the ``Experiment.create_database()`` factory method.
Specifying 1 for the `db_nodes` parameter causes the ``Orchestrator`` to
be single-sharded; otherwise it is multi-shard.
This factory method returns an initialized ``Orchestrator`` object that
gives you access to functions associated with the :ref:`Orchestrator API<orchestrator_api>`.

.. _colocated_deployment_exp_docs:
Colocated Deployment
--------------------
A colocated ``Orchestrator`` shares compute resources with a ``Model`` instance defined by the user.
In this deployment, the ``Orchestrator`` is not connected
together as a single cluster, and the ``Orchestrator`` on each
application node is utilized by SmartRedis
clients on the same node.
Essentially, this means that you have N independent ``Orchestrators``,
where N is the number of compute nodes assigned to the application.
The colocated deployment strategy for the ``Orchestrator``
is ideal for use cases where a SmartSim ``Model`` is run on a compute node
that has hardware accelerators (e.g. GPUs) and low-latency inference is
a critical component of the workflow.

Below is an image illustrating communication within a colocated ``Model`` spanning multiple compute nodes.
As demonstrated in the diagram, each process of the application creates its own SmartRedis client
connection to the orchestrator running on the same host.

.. figure:: images/colocated_orchestrator-1.png

  Sample Colocated Orchestrator Deployment

To create an ``Orchestrator`` that shares compute resources with a ``Model``
SmartSim entity, use the ``model.colocate_db()`` helper method accessible after a
``Model`` object has been initialized. This function instructs
SmartSim to launch a ``Orchestrator`` on the application compute nodes. A ``Orchestrator`` object is not
returned from a ``model.colocate_db()`` instruction, and subsequent interactions with the
colocated ``Orchestrator`` are handled through the :ref:`Model API<model_api>`.

Multiple Orchestrator support
-----------------------------
SmartSim supports multi-database functionality, enabling an experiment
to have several concurrently launched ``Orchestrator(s)``. If there is
a need to launch more than one ``Orchestrator``, the ``Experiment.create_database()``
function mandates the specification of a unique ``Orchestrator`` identifier,
denoted by the `db_identifier` argument, per created ``Orchestrator``.

The `db-identifier` is used to reference SmartSim
``Orchestrators`` from application client code. This is particularly
useful in instances where an ``Orchestrator`` is colocated with a SmartSim
model for low-latency inference and another ``Orchestrator`` is launched to
handle other aspects of the workflow such as visualization and ML model
training. More detailed information on the ideal use cases for standalone ``Orchestrator(s)``
and co-located ``Orchestrator(s)`` is available in the :ref:`Orchestrator documentation
page<orch_docs>`.

.. _model_exp_docs:
Model
=====
``Model(s)`` represent a simulation model or any
computational kernel, including applications,
scripts, or generally, a program.
They can interact with other
SmartSim entities via data transmitted to/from SmartSim ``Orchestrators``
using a SmartRedis client.

A ``Model`` is created through the factory method: ``Experiment.create_model()``.
Models are initialized with ``RunSettings`` objects that specify
how a ``Model`` should be launched via a workload manager
(e.g., Slurm) and the compute resources required.
Optionally, the user may also specify a ``BatchSettings`` object if
the model should be launched as a batch job on the WLM system.
The ``create_model()`` factory method returns an initialized Model object that
gives you access to functions associated with the :ref:`Model API<model_api>`.

.. _ensemble_exp_docs:
Ensemble
========
In addition to a single ``Model``, SmartSim allows users to create,
configure, and launch an ``Ensemble`` of ``Model`` objects.
``Ensembles`` can be given parameters and permutation strategies that define how the
``Ensemble`` will create the underlying ``Model`` objects. Users may also
manually create and append ``Model(s)`` to an ``Ensemble``.
Lastly, the :ref:`Ensemble API<ensemble_api>` supports launching Machine Learning models, TensorFlow
scripts and functions at runtime to enable AI and ML within an ``Ensemble``
Workload.

Ensemble Prefixing
------------------
If each of the ``Ensemble`` members attempt to use the
same code to access their respective data in the Orchestrator,
the names used to reference data, models, and scripts will be identical,
and without the use of SmartSim and SmartRedis helper methods, ``Ensemble`` members
will end up inadvertently accessing or overwriting each other’s data. To prevent
this situation, the SmartSim ``Ensemble`` object supports
key prefixing, which automatically prepends the name
of the model to the keys by which it is accessed. With
this enabled, collision is resolved and
``Ensemble`` members can use the same code.

For example, assume you have two models in the ``Ensemble`` object,
named `model_0` and `model_1`. In the application code you
use the function ``Client.put_tensor("tensor")``. With
``Ensemble`` key prefixing turned on, the `model_0` and `model_1` ``Model`` applications
can access the tensor `"tensor"` by name without overwriting or accessing the other
``Ensemble`` member's `"tensor"` tensor.

Ensemble Creation
-----------------
An ``Ensemble`` is created through the factory method: ``Experiment.create_ensemble()``.
The ``create_ensemble()`` factory method returns an initialized ``Ensemble`` object that
gives you access to functions associated with the :ref:`Ensemble API<ensemble_api>`.
To initialize an ``Ensemble``, a user must follow one of the three methods of ensemble
creation:

1. Manual ``Model`` Appending
     A technique that allows users to create and add model instances to an ensemble, offering a level
     of customization in `Ensemble`` design.
2. Parameter Expansion
     A technique that allows users to set parameter values and control how the parameter values
     spread across the `Ensemble`` members by specifying a permutation strategy.
3. The Utilization of Replicas
     A technique that allows users to create identical or closely related models within an ensemble. Users can assess
     how a system responds to the same set of parameters under multiple instances.

.. note::
  For more information and instruction on `Ensemble`` creation methods, navigate to the :ref:`Ensemble documentation page<ensemble_doc>`.

=======
Example
=======
.. compound::
  In the following subsections, we provide an example of using SmartSim to automate the
  deployment of an HPC workflow consisting of a ``Model`` and standard ``Orchestrator``.
  The example demonstrates:

  *Initializing*
   - a workflow (``Experiment``)
   - an in-memory database (standalone ``Orchestrator``)
   - an application (``Model``)
  *Generating*
   - an in-memory database (standalone ``Orchestrator``) folder
   - an application (``Model``) folder
  *Starting*
   - an in-memory database (standalone ``Orchestrator``)
   - an application (``Model``)
  *Stopping*
   - an in-memory database (standalone ``Orchestrator``)

Initializing
============
.. compound::
  To create a workflow, we *initialize* an ``Experiment`` object
  once at the beginning of the Python driver script.
  To create an ``Experiment``, we specify a name
  and the system launcher with which all entities will be executed.
  Here, we are running the example on a Slurm machine and as such will
  set the `launcher` argument to `slurm`.

  .. code-block:: python

      from smartsim import Experiment
      from smartsim.log import get_logger

      # Initialize an Experiment
      exp = Experiment("name-of-experiment", launcher="slurm")
      # Initialize a SmartSim logger
      smartsim_logger = get_logger("tutorial-experiment")

  We also initialize a SmartSim logger. We will use the logger throughout the ``Experiment``
  to monitor the entities.

.. compound::
  Next, we will launch a SmartSim in-memory database called an ``Orchestrator``.
  To *initialize* an ``Orchestrator`` object, use the ``Experiment.create_database()``
  function. We will create a single-sharded ``Orchestrator`` and therefore will set
  the argument `db_nodes` to 1. SmartSim will assign a `port` to the ``Orchestrator``
  and attempt to detect your machine's interface if values are not provided to the ``Experiment.create_database()`` factory method.

  .. code-block:: python

      # Initialize an Orchestrator
      database = exp.create_database(db_nodes=1)

.. compound::
  Before invoking the factory method to create a ``Model``, we must
  first create a ``RunSettings`` object which holds the information needed to execute the ``Model``
  on the system. The ``RunSettings`` object is initialized using the
  ``Experiment.create_run_settings()`` factory method. In this factory method,
  we specify the executable to run and the arguments to pass to
  the executable.

  The example ``Model`` is a simple `Hello World` program
  that echos `Hello World` to stdout.

  .. code-block:: python

      settings = exp.create_run_settings("echo", exe_args="Hello World")
      model = exp.create_model("hello_world", settings)

  After creating the ``RunSettings`` object, the ``Model`` object can be created and initialized using
  the ``RunSettings`` object via the ``Experiment.create_model()`` function. In the ``Model`` factory method,
  the ``Model`` `name` and the ``RunSettings`` object are provided as input parameters.

Generating
==========
.. compound::
  Next we generate the file structure for the ``Experiment``. A call to ``Experiment.generate()``
  instructs SmartSim to create directories within the ``Experiment`` folder for each instance passed in.
  We plan to organize the ``Orchestrator`` and ``Model`` output files within the ``Experiment`` folder and
  therefore pass the ``Orchestrator`` and ``Model`` instances to ``exp.generate()``:

  .. code-block:: python

    # Create an output directory
    exp.generate(database, model)

  .. note::
    If files or folders are attached to a ``Model`` or ``Ensemble`` members through ``Model.attach_generator_files()``
    or ``Ensemble.attach_generator_files()``, the attached files or directories will be symlinked, copied, or configured and
    written into the created directory for that instance.

Starting
========
.. compound::
  Next we will launch the components of the experiment (``Orchestrator`` and ``Model``) using functions
  provided by the ``Experiment`` API. To do so, we will use
  the ``Experiment.start()`` function and pass in the ``Orchestrator``
  and ``Model`` instances previously created.

  .. code-block:: python

    # Launch the Orchestrator and Model instance
    exp.start(database, model)

  We use the ``Experiment.generate()`` function to create an
  output directory for the ``Orchestrator`` log files.

Stopping
========
.. compound::
  Lastly, to clean up the experiment, we need to tear down the launched ``Orchestrators``.
  We do this by stopping the ``Orchestrator`` using the ``Experiment.stop()`` function.

  .. code-block:: python

    exp.stop(db)
    # log the summary of the Experiment
    exp.summary()

  Notice that we use the ``Experiment.summary()`` function to print
  the summary of the workflow.

.. note::
  Failure to tear down the ``Orchestrator`` at the end of an ``Experiment``
  may lead to ``Orchestrator`` launch failures if another ``Experiment`` is
  started on the same node.