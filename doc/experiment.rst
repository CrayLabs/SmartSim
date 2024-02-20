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

A ``Experiment`` example is offered in the :ref:`Example<exp_example>` section of this page.

.. _launcher_exp_docs:
=========
Launchers
=========
SmartSim supports launching AI-enabled workflows on a wide variety of systems, including locally on a Mac or
Linux machine or on HPC machines with a job scheduler (e.g. Slurm, PBS, and LSF). When creating a SmartSim
``Experiment``, the user has the opportunity to specify the `launcher` type or defer to automatic `launcher` selection.
`Launcher` selection determines how SmartSim translates SmartSim entity configurations into system calls to launch,
manage, and monitor entities. Currently, SmartSim supports 6 `launchers`:

1. ``local`` **[default]**: for single-node, workstation, or laptop
2. ``slurm``: for systems using the Slurm scheduler
3. ``pbs``: for systems using the PBSpro scheduler
4. ``pals``: for systems using the PALS scheduler
5. ``cobalt``: for systems using the Cobalt scheduler
6. ``lsf``: for systems using the LSF scheduler
7. ``auto``: have SmartSim auto-detect the launcher to use

If the systems `launcher` cannot be found or no `launcher` argument is provided, the default value of
`launcher="local"` will be used which will start all ``Experiment`` launched entities on the
localhost.

For examples specifying a `launcher` during ``Experiment`` initialization, navigate to the
:ref:`Experiment __init__ function<exp_init>` in the ``Experiment`` API docstring.

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
The :ref:`Orchestrator<orch_docs>` is an in-memory database built for
a wide variety of AI-enabled workflows. The ``Orchestrator`` can be thought of as a general
feature store capable of storing numerical data, ML models, and scripts.
The ``Orchestrator`` is capable of performing inference and script evaluation using data in the feature store.
Any SmartSim ``Model`` or ``Ensemble`` member can connect to the
``Orchestrator`` via the :ref:`SmartRedis<smartredis-api>`
client library to transmit data, execute ML models, and execute scripts.

**SmartSim offers two types Orchestrator deployments:**

* :ref:`Standalone Orchestrator Deployment<standalone_orch_doc>`
* :ref:`Colocated Orchestrator Deployment<colocated_orch_doc>`

To create a standalone ``Orchestrator`` that does not share compute resources with other
SmartSim entities, use the ``Experiment.create_database()`` factory method which
returns an ``Orchestrator`` object. To create a colocated ``Orchestrator`` that
shares compute resources with a ``Model`` entity, use the ``Model.colocate_db_tcp()``
or ``Model.colocate_db_uds()`` helper methods accessible after a
``Model`` object has been initialized. This function instructs
SmartSim to launch a ``Orchestrator`` on the application compute nodes. A ``Orchestrator`` object is not
returned from a ``Model.colocate_db()`` instruction, and subsequent interactions with the
colocated ``Orchestrator`` are handled through the :ref:`Model API<model_api>`.

SmartSim supports :ref:`multi-database<mutli_orch_doc>` functionality, enabling an ``Experiment`` to have
several concurrently launched ``Orchestrator(s)``. If there is a need to launch more than
one ``Orchestrator``, the ``Experiment.create_database()`` and ``Model.colocate..()``
functions mandate the specification of a unique ``Orchestrator`` identifier, denoted
by the `db_identifier` argument, per created ``Orchestrator``. The `db_identifier` is used
in an application script by a SmartRedis ``Client`` to connect to a specific ``Orchestrator``.

.. _model_exp_docs:
Model
=====
:ref:`Model(s)<model_object_doc>` represent a simulation model or any computational kernel,
including applications, scripts, or generally, a program. They can
interact with other SmartSim entities via data transmitted to/from
SmartSim ``Orchestrators`` using a SmartRedis ``Client``.

A ``Model`` is created through the factory method: ``Experiment.create_model()``.
``Models`` are initialized with ``RunSettings`` objects that specify
how a ``Model`` should be launched via a workload manager
(e.g., Slurm) and the compute resources required.
Optionally, the user may also specify a ``BatchSettings`` object if
the model should be launched as a batch job on the WLM system.
The ``create_model()`` factory method returns an initialized ``Model`` object that
gives you access to functions associated with the :ref:`Model API<model_api>`.

A ``Model`` supports key features, including methods to:

- :ref:`Attach configuration files<files_doc>` for use at ``Model`` runtime.
- :ref:`Colocate an Orchestrator<colo_model_doc>` to a SmartSim ``Model``.
- :ref:`Attach a ML model<ai_model_doc>`  into the ``Orchestrator`` at ``Model`` runtime.
- :ref:`Attach a TorchScript function<TS_doc>`  into the ``Orchestrator`` at ``Model`` runtime.
- :ref:`Enable SmartSim Model data collision prevention<model_key_collision>`, which allows for reuse of key names.

Visit the respective links for more information on each topic.

.. _ensemble_exp_docs:
Ensemble
========
In addition to a single ``Model``, SmartSim allows users to create,
configure, and launch an :ref:`Ensemble<ensemble_doc>` of ``Model`` objects.
``Ensembles`` can be given parameters and permutation strategies that define how the
``Ensemble`` will create the underlying ``Model`` objects. Users may also
manually create and append ``Model(s)`` to an ``Ensemble``. For information
and examples on ``Ensemble`` creation strategies, visit the :ref:`Initialization<init_ensemble_strategies>`
section within the :ref:`Ensemble<ensemble_doc>` documentation.

An ``Ensemble`` supports key features, including methods to:

- :ref:`Attach configuration files<attach_files_ensemble>` for use at ``Ensemble`` runtime.
- :ref:`Load AI models<ai_model_ensemble_doc>` (TF, TF-lite, PT, or ONNX) into the ``Orchestrator`` at ``Ensemble`` runtime.
- :ref:`Load TorchScripts<TS_ensemble_doc>` into the ``Orchestrator`` at ``Ensemble`` runtime.
- :ref:`Prevent data collisions<prefix_ensemble>` within the ``Ensemble``, which allows for reuse of application code.

Visit the respective links for more information on each topic.

.. _exp_example:
=======
Example
=======
.. compound::
  In the following subsections, we provide an example of using SmartSim to automate the
  deployment of an HPC workflow consisting of a ``Model`` and standalone ``Orchestrator``.
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

  The source code example is available in the dropdown below for convenient execution
  and customization.

  .. dropdown:: Example Driver Script source code

      .. literalinclude:: ../tutorials/doc_examples/experiment_doc_examples/exp.py

Initializing
============
.. compound::
  To create a workflow, we *initialize* an ``Experiment`` object
  once at the beginning of the Python driver script.
  To create an ``Experiment``, we specify a name
  and the system launcher with which all entities will be executed.
  Here, we are running the example on a Slurm machine and will
  set the `launcher` argument to `auto` to instruct SmartSim to
  attempt to find the WLM.

  .. literalinclude:: ../tutorials/doc_examples/experiment_doc_examples/exp.py
    :language: python
    :linenos:
    :lines: 1-7

  We also initialize a SmartSim logger. We will use the logger to log the ``Experiment``
  summary.

.. compound::
  Next, we will launch a SmartSim in-memory database called an ``Orchestrator``.
  To *initialize* an ``Orchestrator`` object, use the ``Experiment.create_database()``
  function. We will create a multi-sharded ``Orchestrator`` and therefore will set
  the argument `db_nodes` to 3. SmartSim will assign a `port` to the ``Orchestrator``
  and attempt to detect your machine's interface if values are not provided to the
  ``Experiment.create_database()`` factory method.

  .. literalinclude:: ../tutorials/doc_examples/experiment_doc_examples/exp.py
    :language: python
    :linenos:
    :lines: 9-10

.. compound::
  Before invoking the factory method to create a ``Model``, we must
  first create a ``RunSettings`` object which holds the information needed to execute the ``Model``
  on the system. The ``RunSettings`` object is initialized using the
  ``Experiment.create_run_settings()`` factory method. In this factory method,
  we specify the executable to run and the arguments to pass to
  the executable.

  The example ``Model`` is a simple `Hello World` program
  that echos `Hello World` to stdout.

  .. literalinclude:: ../tutorials/doc_examples/experiment_doc_examples/exp.py
    :language: python
    :linenos:
    :lines: 12-15

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

  .. literalinclude:: ../tutorials/doc_examples/experiment_doc_examples/exp.py
    :language: python
    :linenos:
    :lines: 17-18

  `Overwrite=True` instructs SmartSim to overwrite each file contents if the ``Experiment`` is
  executed again.

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

  .. literalinclude:: ../tutorials/doc_examples/experiment_doc_examples/exp.py
    :language: python
    :linenos:
    :lines: 20-21

  We use the ``Experiment.generate()`` function to create an
  output directory for the ``Orchestrator`` `.err` and `.out` log files.

Stopping
========
.. compound::
  Lastly, to clean up the ``Experiment``, we need to tear down the launched ``Orchestrators``.
  We do this by stopping the ``Orchestrator`` using the ``Experiment.stop()`` function.

  .. literalinclude:: ../tutorials/doc_examples/experiment_doc_examples/exp.py
    :language: python
    :linenos:
    :lines: 23-26

  Notice that we use the ``Experiment.summary()`` function to print
  the summary of the workflow.

When you run the experiment, the following output will appear::

  |    | Name           | Entity-Type   | JobID       | RunID   | Time    | Status    | Returncode   |
  |----|----------------|---------------|-------------|---------|---------|-----------|--------------|
  | 0  | hello_world    | Model         | 1778304.4   | 0       | 10.0657 | Completed | 0            |
  | 1  | orchestrator_0 | DBNode        | 1778304.3+2 | 0       | 43.4797 | Cancelled | 0            |

.. note::
  Failure to tear down the ``Orchestrator`` at the end of an ``Experiment``
  may lead to ``Orchestrator`` launch failures if another ``Experiment`` is
  started on the same node.