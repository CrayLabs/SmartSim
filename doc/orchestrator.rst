************
Orchestrator
************


The ``Orchestrator`` is an in-memory database that is launched prior to all other
entities within an ``Experiment``. The ``Orchestrator`` can be used to store and retrieve
data during the course of an experiment and across multiple entities. In order to
stream data into or receive data from the ``Orchestrator``, one of the SmartSim clients
(SmartRedis) has to be used within a Model.

.. |orchestrator| image:: images/Orchestrator.png
  :width: 700
  :alt: Alternative text

|orchestrator|

Combined with the SmartRedis clients, the ``Orchestrator`` is capable of hosting and executing
AI models written in Python on CPU or GPU. The ``Orchestrator`` supports models written with
TensorFlow, Pytorch, TensorFlow-Lite, or models saved in an ONNX format (e.g. sci-kit learn).


Cluster Orchestrator
====================

The ``Orchestrator`` supports single node and distributed memory settings. This means
that a single compute host can be used for the database or multiple by specifying
``db_nodes`` to be greater than 1.

.. |cluster-orc| image:: images/clustered-orc-diagram.png
  :width: 700
  :alt: Alternative text

|cluster-orc|


With a clustered ``Orchestrator``, multiple compute hosts memory can be used together
to store data. As well, the CPU or GPU(s) where the ``Orchestrator`` is running can
be used to execute the AI models, and Torchscript code on data stored within it.

Users do not need to know how the data is stored in a clustered configuration and
can address the cluster with the SmartRedis clients like a single block of memory
using simple put/get semantics in SmartRedis. SmartRedis will ensure that data
is evenly distributed amoungst all nodes in the cluster.

The cluster deployment is optimal for high data throughput scenarios such as
online analysis, training and processing.


Colocated Orchestrator
=======================

A colocated Orchestrator is a special type of Orchestrator that is deployed on
the same compute hosts an a ``Model`` instance defined by the user. In this
deployment, the database is *not* connected together in a cluster and each
shard of the database is addressed individually by the processes running
on that compute host.

.. |colo-orc| image:: images/co-located-orc-diagram.png
  :width: 700
  :alt: Alternative text


|colo-orc|

This deployment is designed for highly performant online inference scenarios where
a distributed process (likely MPI processes) are performing inference with
data local to each process.

This method is deemed ``locality based inference`` since data is local to each
process and the ``Orchestrator`` is deployed locally on each compute host where
the distributed application is running.


To create a colocated model, first, create a ``Model`` instance and then call
the ``Model.colocate_db_tcp`` or ``Model.colocate_db_uds`` function.

.. currentmodule:: smartsim.entity.model

.. automethod:: Model.colocate_db_tcp
    :noindex:

.. automethod:: Model.colocate_db_uds
    :noindex:

Here is an example of creating a simple model that is colocated with an
``Orchestrator`` deployment using Unix Domain Sockets

.. code-block:: python

  from smartsim import Experiment
  exp = Experiment("colo-test", launcher="auto")

  colo_settings = exp.create_run_settings(exe="./some_mpi_app")

  colo_model = exp.create_model("colocated_model", colo_settings)
  colo_model.colocate_db_uds(
          db_cpus=1,              # cpus given to the database on each node
          debug=False             # include debug information (will be slower)
          ifname=network_interface # specify network interface(s) to use (i.e. "ib0" or ["ib0", "lo"])
  )
  exp.start(colo_model)


By default, SmartSim will pin the database to the first _N_ CPUs according to ``db_cpus``. By
specifying the optional argument ``custom_pinning``, an alternative pinning can be specified
by sending in a list of CPU ids (e.g [0,2,range(5,8)]). For optimal performance, most users
will want to also modify the RunSettings for the model to pin their application to cores not
occupied by the database.

.. warning::

  Pinning is not supported on MacOS X. Setting ``custom_pinning`` to anything
  other than ``None`` will raise a warning and the input will be ignored.

.. note::

  Pinning _only_ affects the co-located deployment because both the application and the database
  are sharing the same compute node. For the clustered deployment, a shard occupies the entirerty
  of the node.

Redis
=====

.. _Redis: https://github.com/redis/redis
.. _RedisAI: https://github.com/RedisAI/RedisAI

The ``Orchestrator`` is built on `Redis`_. Largely, the job of the ``Orchestrator`` is to
create a Python reference to a Redis deployment so that users can launch, monitor
and stop a Redis deployment on workstations and HPC systems.

Redis was chosen for the Orchestrator because it resides in-memory, can be distributed on-node
as well as across nodes, and provides low latency data access to many clients in parallel. The
Redis ecosystem was a primary driver as the Redis module system provides APIs for languages,
libraries, and techniques used in Data Science. In particular, the ``Orchestrator``
relies on `RedisAI`_ to provide access to Machine Learning runtimes.

At its core, Redis is a key-value store. This means that put/get semantics are used to send
messages to and from the database. SmartRedis clients use a specific hashing algorithm, CRC16, to ensure
that data is evenly distributed amongst all database nodes. Notably, a user is not required to
know where (which database node) data or Datasets (see Dataset API) are stored as the
SmartRedis clients will infer their location for the user.


KeyDB
=====

.. _KeyDB: https://github.com/EQ-Alpha/KeyDB

`KeyDB`_ is a multi-threaded fork of Redis that can be swapped in as the database for
the ``Orchestrator`` in SmartSim. KeyDB can be swapped in for Redis by setting the
``REDIS_PATH`` environment variable to point to the ``keydb-server`` binary.

A full example of configuring KeyDB to run in SmartSim is shown below

.. code-block:: bash

  # build KeyDB
  # see https://github.com/EQ-Alpha/KeyDB

  # get KeyDB configuration file
  wget https://github.com/CrayLabs/SmartSim/blob/d3d252b611c9ce9d9429ba6eeb71c15471a78f08/smartsim/_core/config/keydb.conf

  export REDIS_PATH=/path/to/keydb-server
  export REDIS_CONF=/path/to/keydb.conf

  # run smartsim workload

Multi-database Example
======================
SmartSim offers functionality to automate the deployment of multiple
databases on an HPC cluster, supporting workloads that require multiple
``Orchestrators`` for a ``Experiment``. For instance, a workload may consist of a
simulation with high inference performance demands (necessitating a co-located deployment),
along with an analysis and
visualization workflow connected to the simulation (requiring a standard orchestrator).

Below is a simple example demonstrating the process of:
1. Launching two std deployment Orchestrators within an experiment with unique identifiers
2. Launching one colo deployment Orchestrator within an experiment
3. Connecting SmartRedis clients within the driver script and sending tensors to databases
4. Connecting SmartRedis clients within the application and retrieving tensors from databases

The Application
---------------
To store and retrieve data from the two databases
during the course of application, you must correctly
initialize two SmartSim clients (SmartRedis).
In this section, we will write a application script
to demonstrate how to connect SmartRedis
clients in the context of multiple
launched databases.

To begin, import the necessary packages:

.. code-block:: python

  from smartredis import ConfigOptions, Client, log_data

The SmartRedis ``log_data`` will be used to monitor the application status
through the course of the Experiment.

Initialize the Clients
^^^^^^^^^^^^^^^^^^^^^^
To establish a connection with each databases,
we need to initialize a new SmartRedis client for each
``Orchestrator``.

Step 1: Initialize ConfigOptions
""""""""""""""""""""""""""""""""
Since we are launching multiple databases within the experiment,
when initializing a client in the application,
the SmartRedis ``ConfigOptions`` object is required.
We create two ``ConfigOptions`` instances through the
``ConfigOptions.create_from_environment()`` function,
one instance per launched Orchestrator.
Most importantly, the ``create_from_environment()`` function requires specifying the unique database identifier
argument named `db_identifier`. The `db_identifier` argument
serves as a unique identifier for each database launched, guiding
SmartSim in assigning clients to their respective databases.

For the single-sharded database:

.. code-block:: python

  single_shard_config = ConfigOptions.create_from_environment("single_shard_db_identifier")

For the multi-sharded database:

.. code-block:: python

  multi_shard_config = ConfigOptions.create_from_environment("multi-shard-db-identifier")

Step 2: Initialize the Clients
""""""""""""""""""""""""""""""
Now that we have two ``ConfigOptions`` objects, we have the
tools necessary to initialize two SmartRedis clients and
establish a connection with the two databases.
We use the SmartRedis ``Client`` API to create the client instances by passing in
the ``ConfigOptions`` objects and assigning a `logger_name` argument.
It is good practice to use SmartRedis logging
capabilities within your script to help monitor
your ``Model`` during the course of an experiment.

Single-sharded database:

.. code-block:: python

  app_single_shard_client = Client(single_shard_config, logger_name="Model: multi shard logger")

Multi-sharded database:

.. code-block:: python

  app_multi_shard_client = Client(multi_shard_config, logger_name="Model: single shard logger")


Retrieve Data Using Clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^
To confirm a successful connection to each database, we will retrieve the tensors
that we store in the python driver script later in the example.
After initializing a client instance, you have access to the Client API functions.
The ``Client.get_tensor()`` method allows
retrieval of a tensor. It requires the `name` of the tensor assigned
when sent to the database via ``Client.put_tensor()``.

.. code-block:: python

  val1 = app_single_shard_client.get_tensor("tensor_1")
  val2 = app_multi_shard_client.get_tensor("tensor_2")
  app_single_shard_client.log_data(LLInfo, f"The colocated db tensor is: {val1}")
  app_multi_shard_client.log_data(LLInfo, f"The clustered db tensor is: {val2}")

Later, when you run the experiment the following output will appear in ``tutorial_mode.out``
located in ``getting-started-multidb/tutorial_model/``.

.. code-block:: bash

  Model: multi shard logger@00-00-00:The colocated db tensor is: [1 2 3 4]
  Model: single shard logger@00-00-00:The clustered db tensor is: [5 6 7 8]

This output showcases that we have established a connection with multiple Orchestrators.

The Experiment Script
---------------------
To run the previous application, we must define workflow stages within a workload.
Defining workflow stages requires the utilization of functions associated
with the ``Experiment`` object. The Experiment object is intended to be instantiated
once and utilized throughout the workflow runtime.
In this example, we instantiate an ``Experiment`` object with the name ``getting-started-multidb``.
We setup the SmartSim ``logger`` to output information from the Experiment.

.. code-block:: python

  from smartsim import Experiment
  from smartredis import Client
  from smartredis.log import get_logger
  import numpy as np

  exp = Experiment("getting-started-multidb", launcher="auto")
  logger = get_logger("Multidb Experiment Log")

Launch Multiple databases
^^^^^^^^^^^^^^^^^^^^^^^^^
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

.. code-block:: python

  single_shard_db = exp.create_database(db_nodes=1, db_identifier="single_shard_db_identifier")
  exp.generate(single_shard_db, overwrite=True)

For the multi-sharded database:

.. code-block:: python

  multi_shard_db = exp.create_database(db_nodes=3, db_identifier="multi-shard-db-identifier")
  exp.generate(multi_shard_db, overwrite=True)

.. note::
  Calling exp.generate will create two subfolders
  (one for each Orchestrator created in the previous step)
  whose names are based on the db_identifier of that Orchestrator.
  In this example, the Experiment folder is
  named ``getting-started-multidb/``. Within this folder, two Orchestrator subfolders will
  be created, namely ``single_shard_db_identifier/`` and ``multi_shard_db_identifier/``. It's
  important to note that the names of the Orchestrator subfolders are determined by
  the db_identifier value we set when instantiating an ``Orchestrator``.

Step 2: Start databases
"""""""""""""""""""""""
Next, to launch the databases,
pass the database instances to ``Experiment.start()``.

.. code-block:: python

  exp.start(single_shard_db, multi_shard_db, summary=True)

The ``Experiment.start()`` function launches the ``Orchestrators`` for use within the workflow. In other words, the function
deploys the databases on the allocated compute resources.

.. note::
  By setting `summary=True`, we can see a summary of the
  experiment printed before it is launched. The summary
  will stay for 10 seconds, and it is useful as a last
  check. If we set `summary=False`, then the experiment
  would be launched immediately.

Initialize Clients
^^^^^^^^^^^^^^^^^^
The SmartRedis ``Client`` object contains functions that manipulate, send, and receive
data within the database. Each database has a single, dedicated SmartRedis ``Client``.
Begin by initializing a SmartRedis ``Client`` object per launched database.

To create a designated SmartRedis ``Client``, you need to specify the address of the target
running database. You can easily retrieve this address using the ``Orchestrator.get_address()`` function.

For the single-sharded database:

.. code-block:: python

  driver_client_single_shard = Client(cluster=False, address=single_shard_db.get_address()[0], logger_name="Single shard db logger")

For the multi-sharded database:

.. code-block:: python

  driver_client_multi_shard = Client(cluster=True, address=multi_shard_db.get_address()[0], logger_name="Multi shard db logger")

Send Data
^^^^^^^^^
In the application script, we retrieved two NumPy tensors.
To support the apps functionality, we will create two
NumPy arrays in the python driver script and send them to the a database. To
accomplish this, we use the ``Client.put_tensor()`` function with the respective
database client instances.

For the single-sharded database:

.. code-block:: python

  array_1 = np.array([1, 2, 3, 4])
  driver_client_single_shard.put_tensor("tensor_1", array_1)

For the multi-sharded database:

.. code-block:: python

  array_2 = np.array([5, 6, 7, 8])
  driver_client_multi_shard.put_tensor("tensor_2", array_2)

Lets check to make sure the database tensors do not exist in the incorrect databases:

.. code-block:: python

  check_single_shard_db_tensor_incorrect = driver_client_single_shard.key_exists("tensor_2")
  check_multi_shard_db_tensor_incorrect = driver_client_multi_shard.key_exists("tensor_1")
  logger.info(f"The multi shard array key exists in the incorrect database: {check_single_shard_db_tensor_incorrect}")
  logger.info(f"The single shard array key exists in the incorrect database: {check_multi_shard_db_tensor_incorrect}")

When you run the experiment, the following output will appear:

.. code-block:: bash

  00:00:00 system.host.com SmartSim[#####] INFO The multi shard array key exists in the incorrect database: False
  00:00:00 system.host.com SmartSim[#####] INFO The single shard array key exists in the incorrect database: False

Initializing a Model
^^^^^^^^^^^^^^^^^^^^
In the next stage of the experiment, we will utilize the
application script by configuring and creating
a two SmartSim ``Models``.

Step 1: Configure
"""""""""""""""""
You can specify the run settings of a model.
In this example, we invoke the Python interpreter to run a
python script. To specify this, we use the ``Experiment.create_run_settings()`` function.
This function returns a ``RunSettings`` object.
Here, we specify the path to the source code file (script) to
``exe_args`` and the run command to ``exe``.

.. code-block:: python

  model_settings = exp.create_run_settings(exe="python", exe_args="./model_multidb_example.py")

With the ``RunSettings`` instance,
configure the the distribution of computational tasks (``RunSettings.set_nodes()``) and the number of instances
the script is execute on each node (``RunSettings.set_tasks_per_node()``). In this
example, we specify to SmartSim that we intend to execute the script once on a single node.

.. code-block:: python

  model_settings.set_nodes(1)
  model_settings.set_tasks_per_node(1)

Step 2: Initialize
""""""""""""""""""
Next, create a ``Model`` instance using the ``Experiment.create_model()``.
Pass the ``model_settings`` object as an argument
to the ``create_model()`` function and assign to the variable ``model``.

.. code-block:: python

  model = exp.create_model("colocated_model", model_settings)
  model.colocate_db_tcp()
  exp.generate(model, overwrite=True)

Step 3: Start
"""""""""""""
Next, launch the model instance using the ``Experiment.start()`` function.

.. code-block:: python

  exp.start(model, block=True, summary=True)

.. note::
  We set `block=True`,
  so that ``Experiment.start()`` waits until the last Model has finished
  before returning: it will act like a job monitor, letting us know
  if processes run, complete, or fail.

Initializing a Model
^^^^^^^^^^^^^^^^^^^^
Next we will create a colocated model
that echos "Hello World" to stdout.

.. note::
  The model is not a colocated model until
  we apply the ``Model.colocate_db_tcp()``.

Step 1: Configure
"""""""""""""""""
Once again, use the ``Experiment.create_run_settings()`` function
to specify the executable argument (`exe_args`) and run command (`exe`).

.. code-block:: python

  model_settings_2 = exp.create_run_settings("echo", exe_args="Hello World")

Step 2: Initialize
""""""""""""""""""
Next, to create the ``Model`` instance using the ``Experiment.create_model()``.
Pass the ``model_settings_2`` object as an argument
to the ``create_model()`` function and assign to the variable ``model_colo``.
The pass using the ``Model.colocate_db_tcp()`` provided by the ``Model`` API
to colocate an ``Orchestrator`` instance with this Model over TCP/IP.

.. code-block:: python

  model_colo = exp.create_model("colo_model", model_settings_2)
  model_colo.colocate_db_tcp()

Step 3: Start
"""""""""""""
Next, launch the model instance using the ``Experiment.start()`` function.

.. code-block:: python

  exp.start(model_colo, block=True, summary=True)

.. note::
  We set `block=True`,
  so that ``Experiment.start()`` waits until the last Model has finished
  before returning: it will act like a job monitor, letting us know
  if processes run, complete, or fail.

Clobber the databases
^^^^^^^^^^^^^^^^^^^^^
Finally, use the ``Experiment.stop()`` function to stop the database instances. Print the
workflow summary with ``Experiment.summary()``.

.. code-block:: python

  exp.stop(single_shard_db, multi_shard_db)
  logger.info(exp.summary())

When you run the experiment, the following output will appear.

.. code-block:: bash

  00:00:00 system.host.com SmartSim[#####]INFO
  |    | Name                         | Entity-Type   | JobID       | RunID   | Time    | Status    | Returncode   |
  |----|------------------------------|---------------|-------------|---------|---------|-----------|--------------|
  | 0  | std_model                    | Model         | 1538905.7   | 0       | 1.7516  | Completed | 0            |
  | 1  | colo_model                   | Model         | 1538905.8   | 0       | 3.5465  | Completed | 0            |
  | 2  | single_shard_db_identifier_0 | DBNode        | 1538905.5   | 0       | 73.1798 | Cancelled | 0            |
  | 3  | multi-shard-db-identifier    | DBNode        | 1538905.6+2 | 0       | 49.8503 | Cancelled | 0            |

Source Code
-----------

Experiment script
^^^^^^^^^^^^^^^^^
.. code-block:: python

  import numpy as np
  from smartredis import Client
  from smartsim import Experiment
  from smartsim.log import get_logger
  import sys

  exe_ex = sys.executable
  logger = get_logger("Multidb Experiment Log")
  exp = Experiment("getting-started-multidb", launcher="auto")

  single_shard_db = exp.create_database(port=6379, db_nodes=1, interface="ib0", db_identifier="single_shard_db_identifier")
  exp.generate(single_shard_db, overwrite=True)

  multi_shard_db = exp.create_database(port=6380, db_nodes=3, interface="ib0", db_identifier="multi-shard-db-identifier")
  exp.generate(multi_shard_db, overwrite=True)

  exp.start(single_shard_db, multi_shard_db)

  driver_client_single_shard = Client(cluster=False, address=single_shard_db.get_address()[0], logger_name="Single shard db logger")
  driver_client_multi_shard = Client(cluster=True, address=multi_shard_db.get_address()[0], logger_name="Multi shard db logger")

  array_1 = np.array([1, 2, 3, 4])
  driver_client_single_shard.put_tensor("tensor_1", array_1)

  array_2 = np.array([5, 6, 7, 8])
  driver_client_multi_shard.put_tensor("tensor_2", array_2)

  check_single_shard_db_tensor_incorrect = driver_client_single_shard.key_exists("tensor_2")
  check_multi_shard_db_tensor_incorrect = driver_client_multi_shard.key_exists("tensor_1")
  logger.info(f"The multi shard array key exists in the incorrect database: {check_single_shard_db_tensor_incorrect}")
  logger.info(f"The single shard array key exists in the incorrect database: {check_multi_shard_db_tensor_incorrect}")

  model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/lus/scratch/richaama/model_ex.py")
  model_settings.set_nodes(1)
  model_settings.set_tasks_per_node(1)
  model_example = exp.create_model("colocated_model", model_settings)
  model_example.colocate_db_tcp()
  exp.start(model_example, block=True, summary=True)

  exp.stop(single_shard_db, multi_shard_db)
  logger.info(exp.summary())

Application Script
^^^^^^^^^^^^^^^^^^
.. code-block:: python

  from smartredis import ConfigOptions, Client, log_data
  from smartredis import *
  from smartredis.error import *

  single_shard_config = ConfigOptions.create_from_environment("single_shard_db_identifier")
  app_single_shard_client = Client(single_shard_config, logger_name="Model: single shard logger")

  multi_shard_config = ConfigOptions.create_from_environment("multi-shard-db-identifier")
  app_multi_shard_client = Client(multi_shard_config, logger_name="Model: multi shard logger")

  val1 = app_single_shard_client.get_tensor("tensor_1")
  val2 = app_multi_shard_client.get_tensor("tensor_2")

  app_single_shard_client.log_data(LLInfo, f"The colocated db tensor is: {val1}")
  app_multi_shard_client.log_data(LLInfo, f"The clustered db tensor is: {val2}")