************
Orchestrator
************
========
Overview
========
The ``Orchestrator`` is an in-memory database with features built for
AI-enabled workflows including online training, low-latency inference, cross-application data
exchange, online interactive visualization, online data analysis, computational steering, and more.

An ``Orchestrator`` can be thought of as a general feature store
capable of storing numerical data (Tensors and Datasets), AI Models, and scripts (TorchScripts).
In addition to storing data, the ``Orchestrator`` is capable of executing ML models and TorchScripts
on the stored data using CPUs or GPUs.

.. |orchestrator| image:: images/Orchestrator.png
  :width: 700
  :alt: Alternative text

|orchestrator|

Users can establish a connection to the ``Orchestrator`` from within SmartSim ``Model`` executable code, ``Ensemble``
model executable code, or driver scripts using the :ref:`SmartRedis<smartredis-api>` client library.

SmartSim offers two types of ``Orchestrator`` deployments:

- :ref:`clustered deployment<clustered_orch_doc>`
  A clustered ``Orchestrator`` is ideal for systems that have heterogeneous node types
  (i.e. a mix of CPU-only and GPU-enabled compute nodes) where
  ML model and TorchScript evaluation is more efficiently performed off-node for a ``Model``. This
  deployment is also ideal for workflows relying on data exchange between multiple
  applications (e.g. online analysis, visualization, computational steering, or
  producer/consumer application couplings). Clustered deployment is also optimal for
  high data throughput scenarios such as online analysis, training and processing and
  databases that require a large amount of hardware.

- :ref:`colocated deployment<colocated_orch_doc>`.
   A colocated ``Orchestrator`` is ideal when the data and hardware accelerator are located on the same compute node.
   This setup helps reduce latency in ML inference and TorchScript evaluation by eliminating off-node communication.

During clustered deployment, the ``Orchestrator`` is launched
on separate compute nodes than a ``Model``. Clustered deployment is well-suited for throughput
scenarios. In colocated deployment, an ``Orchestrator`` shares compute nodes with a ``Model``. Colocated
deployment is well-suited for inference scenarios.

SmartSim allows users to launch multiple orchestrators during the course of an experiment of
either deployment type. If a workflow requires a multiple database environment, a
`db_identifier` argument must be specified during database initialization. Users can connect to
orchestrators in a parallel database workflow by specifying the respective `db_identifier` argument
when initializing a SmartRedis client object. The client can then be used to transmit data,
execute ML models, and execute scripts on the linked database.

.. _clustered_orch_doc:
======================
Clustered Deployment
======================
--------
Overview
--------
During clustered deployment, a SmartSim ``Orchestrator`` (the database) runs on separate
compute node(s) from the ``Model`` node(s). A clustered ``Orchestrator`` can be deployed on a single
node or sharded (distributed) over multiple nodes. With a sharded ``Orchestrator``, available hardware
for inference and script evaluation increases and overall memory for data storage increases.

Communication between a clustered Orchestrator and Model
is initialized in the ``Model`` application script via a SmartRedis client.
Users do not need to know how the data is stored in a clustered configuration and
can address the cluster with the SmartRedis clients like a single block of memory
using simple put/get semantics in SmartRedis. The client can establish a connection
with an ``Orchestrator`` through **three** processes:

- SmartSim establishes a connection using the database address provided by SmartSim through ``Model`` environment configuration
  at runtime.
- A user provides the database address in the Client constructor.
- In multiple database experiments, a user provides the `db_identifier` used to create the clustered
  database when creating a client.

.. |cluster-orc| image:: images/clustered-orc-diagram.png
  :width: 700
  :alt: Alternative text

|cluster-orc|

A clustered database is optimal for high data throughput scenarios
such as online analysis, training and processing.
Clustered Orchestrators support data communication across multiple simulations.
With clustered database deployment, SmartSim can run AI models, and Torchscript
code on the CPU(s) or GPU(s) with existing data in the ``Orchestrator``.
Data produced by these processes and stored in the clustered database is available for
consumption by other applications.

-------
Example
-------
In the following example, we provide a demonstration on automating the deployment of
a clustered Orchestrator using SmartSim from within a Python driver script. Once the standard database is launched,
we demonstrate connecting a client to the database from within the application script to transmit and poll data.

The example is comprised of two script files:

- The Application Script
   The application script is a Python file that contains instructions to create SmartRedis
   client connection to the standard Orchestrator launched in the driver script. From within the
   application script, the client sends and retrieve data.
- The Experiment Driver Script
   The experiment driver script launches and manages SmartSim entities. In the driver script, we use the Experiment
   API to create and launch a standard ``orchestrator``. We create a client connection and store a tensor for use within
   the application. We then initialize a ``Model`` object with the
   application script as an executable argument. Once the database has launched, we launch the ``Model``.
   We then retrieve the tensors stored by the ``Model`` from within the driver script. Lastly, we tear down the database.

The Application Script
======================
To begin writing the application script, import the necessary SmartRedis packages:

.. code-block:: python

  from smartredis import Client, log_data
  from smartredis import *
  import numpy as np

Initialize the Client
---------------------
To establish a connection with the standard database, we need to initialize a new SmartRedis client.
Since the standard database we launch in the driver script is sharded, we specify the `cluster` as `True`:

.. code-block:: python

  # Initialize a Client
  standard_db_client = Client(cluster=True)

Retrieve Data
-------------
To confirm a successful connection to the database, we retrieve the tensor we store in the Python driver script.
Use the ``Client.get_tensor()`` method to retrieve the tensor by specifying the name `tensor_1` we
used during ``Client.put_tensor()`` in the driver script:

.. code-block:: python

    # Retrieve tensor from Orchestrator
    value_1 = standard_db_client.get_tensor("tensor_1")
    # Log tensor
    standard_db_client.log_data(LLInfo, f"The single sharded db tensor is: {value_1}")

Later, when you run the experiment driver script the following output will appear in ``model.out``
located in ``getting-started/tutorial_model/``::

  Default@17-11-48:The single sharded db tensor is: [1 2 3 4]

Store Data
----------
Next, create a NumPy tensor to send to the standard database using
``Client.put_tensor(name, data)``:

.. code-block:: python

  # Create a NumPy array
  array_2 = np.array([5, 6, 7, 8])
  # Use SmartRedis client to place tensor in multi-sharded db
  standard_db_client.put_tensor("tensor_2", array_2)

We will retrieve `"tensor_2"` in the Python driver script.

The Experiment Driver Script
============================
To run the previous application script, we define a ``Model`` and ``Orchestrator`` within an
Python driver script. Defining workflow stages (``Model`` and ``Orchestrator``) requires the utilization of functions associated
with the ``Experiment`` object. The ``Experiment`` object is intended to be instantiated
once and utilized throughout the workflow runtime.
In this example, we instantiate an ``Experiment`` object with the name ``getting-started``.
We setup the SmartSim ``logger`` to output information from the Experiment:

.. code-block:: python

  import numpy as np
  from smartredis import Client
  from smartsim import Experiment
  from smartsim.log import get_logger
  import sys

  exe_ex = sys.executable
  logger = get_logger("Example Experiment Log")
  # Initialize the Experiment
  exp = Experiment("getting-started", launcher="auto")

Launch Standard Orchestrator
----------------------------
In the context of this ``Experiment``, it's essential to create and launch
the databases as a preliminary step before any other workflow components. This is because
the application script requests and sends tensors to and from a launched database.

We aim to demonstrate the standard orchestrator automation capabilities of SmartSim, so we
create a clustered database in the workflow.

Step 1: Initialize Orchestrator
'''''''''''''''''''''''''''''''
To create a standard database, utilize the ``Experiment.create_database()`` function.
.. code-block:: python

  # Initialize a multi sharded database
  standard_db = exp.create_database(db_nodes=3)
  exp.generate(standard_db)

Step 2: Start Databases
'''''''''''''''''''''''
Next, to launch the database, pass the database instance to ``Experiment.start()``.
.. code-block:: python

  # Launch the multi sharded database
  exp.start(standard_db)

The ``Experiment.start()`` function launches the ``Orchestrator`` for use within the workflow.
In other words, the function deploys the database on the allocated compute resources.

Create a Client Connection to the Orchestrator
----------------------------------------------
The SmartRedis ``Client`` object contains functions that manipulate, send, and retrieve
data on the database. Each database can have a single, dedicated SmartRedis ``Client`` connection.
Begin by initializing a SmartRedis ``Client`` object for the standard database.

When creating a client connection from within a driver script,
specify the address of the database you would like to connect to.
You can easily retrieve the database address using the ``Orchestrator.get_address()`` function:

.. code-block:: python

  # Initialize a SmartRedis client for multi sharded database
  driver_client_standard_db = Client(cluster=True, address=standard_db.get_address()[0])

Store Data Using Clients
------------------------
In the application script, we retrieved a NumPy tensor stored from within the driver script.
To support the application functionality, we create a
NumPy array in the experiment workflow to send to the database. To
send a tensor to the database, use the function ``Client.put_tensor()``:
.. code-block:: python

  # Create NumPy array
  array_1 = np.array([1, 2, 3, 4])
  # Use the SmartRedis client to place tensor in the standard database
  driver_client_standard_db.put_tensor("tensor_1", array_1)

Initialize a Model
------------------
In the next stage of the experiment, we execute the application script by configuring and creating
a SmartSim ``Model`` and specifying the application script name during ``Model`` creation.

Step 1: Configure
'''''''''''''''''
In the example experiment, we invoke the Python interpreter to run
the python application script defined in section: The Application Script.
We use ``Experiment.create_run_settings()`` to create a configuration object that will define the
operation of a ``Model``. The function returns a ``RunSettings`` object.
When initializing the ``RunSettings`` object, we specify the path to the application file,
`application_script.py`, to ``exe_args``, and the run command to ``exe``.

.. code-block:: python

  # Initialize a RunSettings object
  model_settings = exp.create_run_settings(exe=exe_ex, exe_args="application_script.py")
  model_settings.set_nodes(1)

Step 2: Initialize
''''''''''''''''''
Next, create a ``Model`` instance using the ``Experiment.create_model()``.
Pass the ``model_settings`` object as an argument
to the ``create_model()`` function and assign to the variable ``model``:

.. code-block:: python

  # Initialize the Model
  model = exp.create_model("model", model_settings)

Step 3: Start
'''''''''''''
Next, launch the model instance using the ``Experiment.start()`` function:

.. code-block:: python

  # Launch the Model
  exp.start(model, block=True, summary=True)

.. note::
    We specify `block=True` to ``exp.start()`` because our experiment
    requires that the ``Model`` finish before the experiment continues.
    This is because we will request tensors from the database that
    are inputted by the Model we launched.

Poll Data Using Clients
-----------------------
Next, check if the tensor exists in the standard database using ``Client.poll_tensor()``.
This function queries for data in the database. The function requires the tensor name (`name`),
how many milliseconds to wait in between queries (`poll_frequency_ms`),
and the total number of times to query (`num_tries`). Check if the data exists in the database by
polling every 100 milliseconds until 10 attempts are completed:

.. code-block:: python

  # Retrieve the tensors placed by the Model
  value_2 = driver_client_standard_db.poll_key("tensor_2", 100, 10)
  # Validate that the tensor exists
  logger.info(f"The tensor is {value_2}")

The output will be as follows::
  noted to run and replace asap

Cleanup Experiment
------------------
Finally, use the ``Experiment.stop()`` function to stop the database instances. Print the
workflow summary with ``Experiment.summary()``:

.. code-block:: python

  # Cleanup the database
  exp.stop(standard_db)
  logger.info(exp.summary())

When you run the experiment, the following output will appear::
 again noted to fill in, oops

====================
.. _colocated_orch_doc:
Colocated Deployment
====================
--------
Overview
--------
During colocated deployment, a SmartSim ``Orchestrator`` (the database) is launched on
the ``Model`` compute node(s).
The orchestrator is non-clustered and each ``Model`` compute node hosts an instance of the database.
Processes on the compute host individually address the database.

Communication between a colocated Orchestrator and Model
is initialized in the application script via a SmartRedis client. Since a colocated Orchestrator is launched when the Model
is started by the experiment, you may only connect a SmartRedis client to a colocated database from within
the associated colocated Model script. The client establishes a connection using the database address detected
by SmartSim or provided by the user. In multiple database experiments, users provide the `db_identifier` used to create the colocated
Model when creating a client connection.

.. |colo-orc| image:: images/co-located-orc-diagram.png
  :width: 700
  :alt: Alternative text


|colo-orc|

Colocated deployment is designed for highly performant online inference scenarios where
a distributed process (likely MPI processes) are performing inference with
data local to each process. Data produced by these processes and stored in the colocated database
can be transferred via a SmartRedis client to a standard database to become available for consumption
by other applications. A tradeoff of colocated deployment is the ability to scale to a large workload.
Colocated deployment rather benefits small/medium simulations with low latency requirements.
By hosting the database and simulation on the same compute node, communication time is reduced which
contributes to quicker processing speeds.

This method is deemed ``locality based inference`` since data is local to each
process and the ``Orchestrator`` is deployed locally on each compute host where
the distributed application is running.

-------
Example
-------
In the following example, we provide a demonstration on automating the deployment of
a colocated Orchestrator using SmartSim from within a Python driver script. Once the colocated database is launched,
we demonstrate connecting a client to the database from within the application script to transmit and poll data.

The example is comprised of two script files:

- The Application Script
   The example application script is a Python file that contains
   instructions to create and connect a SmartRedis
   client to the colocated Orchestrator.
- The Experiment Driver Script
   The experiment driver script launches and manages
   the example entities with the ``Experiment`` API.
   In the driver script, we use the ``Experiment``
   to create and launch a colocated ``Model``.

The Application Script
======================
To begin writing the application script, provide the imports:

.. code-block:: python

  from smartredis import ConfigOptions, Client, log_data
  from smartredis import *
  import numpy as np

Initialize the Clients
----------------------
To establish a connection with the colocated database,
initialize a new SmartRedis client and specify `cluster=False`
since our database is single-sharded:

.. code-block:: python

  # Initialize a Client
  colo_client = Client(cluster=False)

.. note::
    Since there is only one database launched in the Experiment
    (the colocated database), specifying a a database address
    is not required when initializing the client.
    SmartRedis will handle the connection.

.. note::
   To create a client connection to the colocated database, the colocated Model must be launched
   from within the driver script. You must execute the Python driver script, otherwise, there will
   be no database to connect the client to.

Store Data
----------
Next, using the SmartRedis client instance, we create and store a NumPy tensor using
``Client.put_tensor()``:

.. code-block:: python

    # Create NumPy array
    array_1 = np.array([1, 2, 3, 4])
    # Store the NumPy tensor
    colo_client.put_tensor("tensor_1", array_1)

Retrieve Data
-------------
Next, retrieve the tensor using ``Client.get_tensor()``:

.. code-block:: python

    # Retrieve tensor from driver script
    value_1 = colo_client.get_tensor("tensor_1")
    # Log tensor
    colo_client.log_data(LLInfo, f"The colocated db tensor is: {value_1}")

When the Experiment completes, you can find the following log message in `colo_model.out`::
    Default@21-48-01:The colocated db tensor is: [1 2 3 4]

The Experiment Driver Script
============================
To run the application, specify a Model workload from
within the workflow (Experiment).
Defining workflow stages requires the utilization of functions associated
with the ``Experiment`` object.
In this example, we instantiate an ``Experiment`` object with the name ``getting-started``.
We setup the SmartSim ``logger`` to output information from the Experiment.

.. code-block:: python

    import numpy as np
    from smartredis import Client
    from smartsim import Experiment
    from smartsim.log import get_logger
    import sys

    exe_ex = sys.executable
    logger = get_logger("Example Experiment Log")
    # Initialize the Experiment
    exp = Experiment("getting-started", launcher="auto")

Initialize a Colocated Model
----------------------------
In the next stage of the experiment, we
create and launch a colocated ``Model`` that
runs the application script with a database
on the same compute node.

Step 1: Configure
"""""""""""""""""
In this experiment, we invoke the Python interpreter to run
the python script defined in section: The Application Script.
To configure this into a ``Model``, we use the ``Experiment.create_run_settings()`` function.
The function returns a ``RunSettings`` object.
A ``RunSettings`` allows you to configure
the run settings of a SmartSim entity.
We initialize a RunSettings object and
specify the path to the application file,
`application_script.py`, to the argument
``exe_args``, and the run command to ``exe``.

.. note::
  Change the `exe_args` argument to the path of the application script
  on your file system to run the example.

Use the ``RunSettings`` helper functions to
configure the the distribution of computational tasks (``RunSettings.set_nodes()``). In this
example, we specify to SmartSim that we intend the Model to run on a single compute node.

.. code-block:: python

    # Initialize a RunSettings object
    model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/lus/scratch/richaama/clustered_model.py")
    # Configure RunSettings object
    model_settings.set_nodes(1)

Step 2: Initialize
""""""""""""""""""
Next, create a ``Model`` instance using the ``Experiment.create_model()``.
Pass the ``model_settings`` object as an argument
to the ``create_model()`` function and assign to the variable ``model``.
.. code-block:: python

    # Initialize a SmartSim Model
    model = exp.create_model("colo_model", model_settings)

Step 2: Colocate
""""""""""""""""
To colocate the model, use the ``Model.colocate_db_tcp()`` function.
This function will colocate an Orchestrator instance with this Model over
a Unix domain socket connection.

.. code-block:: python

    # Colocate the Model
    model.colocate_db_tcp()

Step 3: Start
"""""""""""""
Next, launch the colocated model instance using the ``Experiment.start()`` function.
.. code-block:: python

    # Launch the colocated Model
    exp.start(model, block=True, summary=True)

test

Cleanup Experiment
------------------

.. code-block:: python

    logger.info(exp.summary())

When you run the experiment, the following output will appear::

  |    | Name   | Entity-Type   | JobID     | RunID   | Time    | Status    | Returncode   |
  |----|--------|---------------|-----------|---------|---------|-----------|--------------|
  | 0  | model  | Model         | 1592652.0 | 0       | 10.1039 | Completed | 0            |

======================
Multiple Orchestrators
======================
SmartSim supports automating the deployment of multiple Orchestrators
from within an Experiment. Communication with the database via a SmartRedis client is possible with the
`db_identifier` argument that is required when initializing an Orchestrator or
colocated Model during a multiple database experiment. When initializing a SmartRedis
client during the Experiment, first create a ``ConfigOptions`` object
with the `db_identifier` argument created during before passing object to the Client()
init call.

Multiple Orchestrator Example
=============================
SmartSim offers functionality to automate the deployment of multiple
databases, supporting workloads that require multiple
``Orchestrators`` for a ``Experiment``. For instance, a workload may consist of a
simulation with high inference performance demands (necessitating a co-located deployment),
along with an analysis and
visualization workflow connected to the simulation (requiring a standard orchestrator).
In the following example, we simulate a simple version of this use case.

The example is comprised of two script files:

* The Application Script
* The Experiment Driver Script

**The Application Script Overview:**
In this example, the application script is a python file that
contains instructions to complete computational
tasks. Applications are not limited to Python
and can also be written in C, C++ and Fortran.
This script specifies creating a Python SmartRedis client for each
standard orchestrator and a colocated orchestrator. We use the
clients to request data from both standard databases, then
transfer the data to the colocated database. The application
file is launched by the experiment driver script
through a ``Model`` stage.

**The Application Script Contents:**

1. Connecting SmartRedis clients within the application to retrieve tensors
   from the standard databases to store in a colocated database. Details in section:
   :ref:`Initialize the Clients<Initialize the Clients>`.

**The Experiment Driver Script Overview:**
The experiment driver script holds the stages of the workflow
and manages their execution through the ``Experiment`` API.
We initialize an Experiment
at the beginning of the Python file and use the ``Experiment`` to
iteratively create, configure and launch computational kernels
on the system through the `slurm` launcher.
In the driver script, we use the ``Experiment`` to create and launch a ``Model`` instance that
runs the application.

**The Experiment Driver Script Contents:**

1. Launching two standard Orchestrators with unique identifiers. Details in section:
   :ref:`Launch Multiple Orchestrators<Launch Multiple Orchestrators>`.
2. Launching the application script with a co-located database. Details in section:
   :ref:`Initialize a Colocated Model<Initialize a Colocated Model>`.
3. Connecting SmartRedis clients within the driver script to send tensors to standard Orchestrators
   for retrieval within the application. Details in section:
   :ref:`Create Client Connections to Orchestrators<Create Client Connections to Orchestrators>`.

Setup and run instructions can be found :ref:`here<How to Run the Example>`

The Application Script
----------------------
Applications interact with the databases
through a SmartRedis client.
In this section, we write an application script
to demonstrate how to connect SmartRedis
clients in the context of multiple
launched databases. Using the clients, we retrieve tensors
from two databases launched in the driver script, then store
the tensors in the colocated database.

.. note::
   The Experiment must be started to use the Orchestrators within the
   application script.  Otherwise, it will fail to connect.
   Find the instructions on how to launch :ref:`here<How to Run the Example>`

To begin, import the necessary packages:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 1-3

Initialize the Clients
^^^^^^^^^^^^^^^^^^^^^^
To establish a connection with each database,
we need to initialize a new SmartRedis client for each
``Orchestrator``.

Step 1: Initialize ConfigOptions
""""""""""""""""""""""""""""""""
Since we are launching multiple databases within the experiment,
the SmartRedis ``ConfigOptions`` object is required when initializing
a client in the application.
We use the ``ConfigOptions.create_from_environment()``
function to create three instances of ``ConfigOptions``,
with one instance associated with each launched ``Orchestrator``.
Most importantly, to associate each launched Orchestrator to a ConfigOptions object,
the ``create_from_environment()`` function requires specifying the unique database identifier
argument named `db_identifier`.

For the single-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 5-6

For the multi-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 10-11

For the colocated database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 15-16

Step 2: Initialize the Client Connections
"""""""""""""""""""""""""""""""""""""""""
Now that we have three ``ConfigOptions`` objects, we have the
tools necessary to initialize three SmartRedis clients and
establish a connection with the three databases.
We use the SmartRedis ``Client`` API to create the client instances by passing in
the ``ConfigOptions`` objects and assigning a `logger_name` argument.

Single-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 7-8

Multi-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 12-13

Colocated database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 17-18

Retrieve Data and Store Using SmartRedis Client Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To confirm a successful connection to each database, we will retrieve the tensors
that we plan to store in the python driver script. After retrieving, we
store both tensors in the colocated database.
The ``Client.get_tensor()`` method allows
retrieval of a tensor. It requires the `name` of the tensor assigned
when sent to the database via ``Client.put_tensor()``.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 20-26

Later, when you run the experiment driver script the following output will appear in ``tutorial_model.out``
located in ``getting-started-multidb/tutorial_model/``::

  Model: single shard logger@00-00-00:The single sharded db tensor is: [1 2 3 4]
  Model: multi shard logger@00-00-00:The multi sharded db tensor is: [5 6 7 8]

This output showcases that we have established a connection with multiple Orchestrators.

Next, take the tensors retrieved from the standard deployment databases and
store them in the colocated database using  ``Client.put_tensor(name, data)``.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 28-30

Next, check if the tensors exist in the colocated database using ``Client.poll_tensor()``.
This function queries for data in the database. The function requires the tensor name (`name`),
how many milliseconds to wait in between queries (`poll_frequency_ms`),
and the total number of times to query (`num_tries`):

.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:
  :lines: 32-37

The output will be as follows::

  Model: colo logger@00-00-00:The colocated db has tensor_1: True
  Model: colo logger@00-00-00:The colocated db has tensor_2: True

The Experiment Driver Script
----------------------------
To run the previous application, we must define workflow stages within a workload.
Defining workflow stages requires the utilization of functions associated
with the ``Experiment`` object. The Experiment object is intended to be instantiated
once and utilized throughout the workflow runtime.
In this example, we instantiate an ``Experiment`` object with the name ``getting-started-multidb``.
We setup the SmartSim ``logger`` to output information from the Experiment.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 1-10

Launch Multiple Orchestrators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the context of this ``Experiment``, it's essential to create and launch
the databases as a preliminary step before any other components since
the application script requests tensors from the launched databases.

We aim to showcase the multi-database automation capabilities of SmartSim, so we
create two databases in the workflow: a single-sharded database and a
multi-sharded database.

Step 1: Initialize Orchestrators
""""""""""""""""""""""""""""""""
To create an database, utilize the ``Experiment.create_database()`` function.
The function requires specifying a unique
database identifier argument named `db_identifier` to launch multiple databases.
This step is necessary to connect to databases outside of the driver script.
We will use the `db_identifier` names we specified in the application script.

For the single-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 12-14

For the multi-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 16-18

.. note::
  Calling ``exp.generate()`` will create two subfolders
  (one for each Orchestrator created in the previous step)
  whose names are based on the db_identifier of that Orchestrator.
  In this example, the Experiment folder is
  named ``getting-started-multidb/``. Within this folder, two Orchestrator subfolders will
  be created, namely ``single_shard_db_identifier/`` and ``multi_shard_db_identifier/``.

Step 2: Start Databases
"""""""""""""""""""""""
Next, to launch the databases,
pass the database instances to ``Experiment.start()``.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 20-21

The ``Experiment.start()`` function launches the ``Orchestrators`` for use within the workflow. In other words, the function
deploys the databases on the allocated compute resources.

.. note::
  By setting `summary=True`, SmartSim will print a summary of the
  experiment before it is launched. After printing the experiment summary,
  the experiment is paused for 10 seconds giving the user time to
  briefly scan the summary contents. If we set `summary=False`, then the experiment
  would be launched immediately with no summary.

Create Client Connections to Orchestrators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The SmartRedis ``Client`` object contains functions that manipulate, send, and receive
data within the database. Each database has a single, dedicated SmartRedis ``Client``.
Begin by initializing a SmartRedis ``Client`` object per launched database.

To create a designated SmartRedis ``Client``, you need to specify the address of the target
running database. You can easily retrieve this address using the ``Orchestrator.get_address()`` function.

For the single-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 23-24

For the multi-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 25-26

Store Data Using Clients
^^^^^^^^^^^^^^^^^^^^^^^^
In the application script, we retrieved two NumPy tensors.
To support the apps functionality, we will create two
NumPy arrays in the python driver script and send them to the a database. To
accomplish this, we use the ``Client.put_tensor()`` function with the respective
database client instances.

For the single-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 28-31

For the multi-sharded database:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 33-36

Lets check to make sure the database tensors do not exist in the incorrect databases:

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 38-42

When you run the experiment, the following output will appear::

  00:00:00 system.host.com SmartSim[#####] INFO The multi shard array key exists in the incorrect database: False
  00:00:00 system.host.com SmartSim[#####] INFO The single shard array key exists in the incorrect database: False

Initialize a Colocated Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the next stage of the experiment, we
launch the application script with a co-located database
by configuring and creating
a SmartSim colocated ``Model``.

Step 1: Configure
"""""""""""""""""
You can specify the run settings of a model.
In this experiment, we invoke the Python interpreter to run
the python script defined in section: :ref:`The Application Script<The Application Script>`.
To configure this into a ``Model``, we use the ``Experiment.create_run_settings()`` function.
The function returns a ``RunSettings`` object.
When initializing the RunSettings object,
we specify the path to the application file,
`application_script.py`, for
``exe_args``, and the run command for ``exe``.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 44-45

.. note::
  You will have to change the `exe_args` argument to the path of the application script
  on your machine to run the example.

With the ``RunSettings`` instance,
configure the the distribution of computational tasks (``RunSettings.set_nodes()``) and the number of instances
the script is execute on each node (``RunSettings.set_tasks_per_node()``). In this
example, we specify to SmartSim that we intend to execute the script once on a single node.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 46-48

Step 2: Initialize
""""""""""""""""""
Next, create a ``Model`` instance using the ``Experiment.create_model()``.
Pass the ``model_settings`` object as an argument
to the ``create_model()`` function and assign to the variable ``model``.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 49-50

Step 2: Colocate
""""""""""""""""
To colocate the model, use the ``Model.colocate_db_uds()`` function to
Colocate an Orchestrator instance with this Model over
a Unix domain socket connection.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 51-52

This method will initialize settings which add an unsharded
database to this Model instance. Only this Model will be able
to communicate with this colocated database by using the loopback TCP interface.

Step 3: Start
"""""""""""""
Next, launch the colocated model instance using the ``Experiment.start()`` function.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 53-54

.. note::
  We set `block=True`,
  so that ``Experiment.start()`` waits until the last Model has finished
  before returning: it will act like a job monitor, letting us know
  if processes run, complete, or fail.

Cleanup Experiment
^^^^^^^^^^^^^^^^^^
Finally, use the ``Experiment.stop()`` function to stop the database instances. Print the
workflow summary with ``Experiment.summary()``.

.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos:
  :lines: 56-59

When you run the experiment, the following output will appear::

  00:00:00 system.host.com SmartSim[#####]INFO
  |    | Name                         | Entity-Type   | JobID       | RunID   | Time    | Status    | Returncode   |
  |----|------------------------------|---------------|-------------|---------|---------|-----------|--------------|
  | 0  | colo_model                   | Model         | 1556529.5   | 0       | 1.7437  | Completed | 0            |
  | 1  | single_shard_db_identifier_0 | DBNode        | 1556529.3   | 0       | 68.8732 | Cancelled | 0            |
  | 2  | multi_shard_db_identifier_0  | DBNode        | 1556529.4+2 | 0       | 45.5139 | Cancelled | 0            |

How to Run the Example
----------------------
Below are the steps to run the experiment. Find the
:ref:`experiment source code<Experiment Source Code>`
and :ref:`application source code<Application Source Code>`
below in the respective subsections.

.. note::
  The example assumes that you have already installed and built
  SmartSim and SmartRedis. Please refer to Section :ref:`Basic Installation<Basic Installation>`
  for further details. For simplicity, we assume that you are
  running on a SLURM-based HPC-platform. Refer to the steps below
  for more details.

Step 1 : Setup your directory tree
    Your directory tree should look similar to below::

      SmartSim/
      SmartRedis/
      Multi-db-example/
        application_script.py
        experiment_script.py

    You can find the application and experiment source code in subsections below.

Step 2 : Install and Build SmartSim
    This example assumes you have installed SmartSim and SmartRedis in your
    Python environment. We also assume that you have built SmartSim with
    the necessary modules for the machine you are running on.

Step 3 : Change the `exe_args` file path
    When configuring the colocated model in `experiment_script.py`,
    we pass the file path of `application_script.py` to the `exe_args` argument
    on line 33 in :ref:`experiment_script.py<Experiment Source Code>`.
    Edit this argument to the file path of your `application_script.py`

Step 4 : Run the Experiment
    Finally, run the experiment with ``python experiment_script.py``.


Application Source Code
^^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../tutorials/getting_started/multi_db_example/application_script.py
  :language: python
  :linenos:

Experiment Source Code
^^^^^^^^^^^^^^^^^^^^^^
.. literalinclude:: ../tutorials/getting_started/multi_db_example/multidb_driver.py
  :language: python
  :linenos: