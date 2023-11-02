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

Multidb
=======

To support computationally intensive tasks like scientific simulations,
SmartSim provides functionality to automate the deployment of multiple
databases on an HPC cluster. As data volume increases, consider augmenting
the number of sharded databases or provision multiple standalone databases,
to ensure the workload can scale to meet resource demands. To ensure that data
retrieval, updates, and synchronization are carried out smoothly and efficiently
across multiple database nodes, unique identifier names are given to each database
during initialization. Below is a general example of launching multiple databases
with a unique identifier and distributing data between the launched databases.

Defining workflow stages requires the utilization of functions associated
with the Experiment object. The Experiment object is intended to be instantiated
once and utilized throughout the workflow runtime. Begin by creating an Experiment
object and assign it to the 'exp' variable. In this example, we instantiate the
Experiment object with the name 'getting-started-multidb' and the 'launcher=slurm'
configuration.

.. code-block:: python
  import numpy as np
  from smartredis import Client
  from smartsim import Experiment

  exp = Experiment("getting-started-multidb", launcher="slurm")


In the context of this Experiment, it's essential to create and launch
the databases as a preliminary step before any other components. We aim
to showcase the multi-database automation capabilities of SmartSim, so we
create two types of databases in the workflow: a standalone database and a
clustered database.

To create a database, you can utilize the create_database() function located
in the Experiment class. For launching multiple databases, this function requires
specifying a unique database identifier argument named db_identifier. The db_identifier
argument plays a crucial role in SmartSim multi-database support by serving as a unique
identifier for each database being launched. The db_identifier will be utilized later
in the example.

Here's how to create the databases:

For the standalone database:

.. code-block:: python
  standalone_database = exp.create_database(port=6379, db_nodes=1, interface="ib0", db_identifier="std-deployment")
  exp.generate(standalone_database, overwrite=True)

For the clustered database:

.. code-block:: python
  clustered_database = exp.create_database(port=6380, db_nodes=3, interface="ib0", db_identifier="clus-deployment")
  exp.generate(clustered_database, overwrite=True)

Next, pass the database instances to the following command:

.. code-block:: python
  exp.start(standalone_database, clustered_database, block=False, summary=True)

The Experiment.start() function launches the Orchestrators for use within the workflow.

.. note::

  The Orchestrator is an in-memory database that can be launched alongside entities in SmartSim

Next in the example, learn to distribute data between the launched databases by using the Client SmartRedis API.

The SmartRedis Client object contains functions that manipulate, send, and receive
data within the database. Each database has a single, dedicated SmartRedis Client.
Begin by initializing a SmartRedis Client object per launched database.

To create a designated SmartRedis Client, you need to specify the address of the target
running database. You can easily retrieve this address using the get_address() function
provided by the initialized Orchestrator object.

.. note::

  Make sure to set the 'cluster' argument appropriately based on whether the database is in a clustered configuration or not.

Here's the code example:

.. code-block:: python
  client1 = Client(address=standalone_database.get_address()[0], cluster=False)
  client2 = Client(address=clustered_database.get_address()[0], cluster=True)

In this code, 'client1' is initialized for the standalone Orchestrator, while 'client2' is
initialized for the clustered Orchestrator.

In this example, we transmit, retrieve and manipulate a NumPy tensor. Begin by creating a
NumPy array and send it to each database using the associated SmartRedis client. To
accomplish this, use the put_tensor() function provided by the SmartRedis Client API.
We access this function through the SmartRedis Client objects, client1 and client2.
In this example, we are sending the same 'array' to both databases.

.. code-block:: python
  array = np.array([1, 2, 3, 4])
  client1.put_tensor("tensor1", array)
  client2.put_tensor("tensor2", array)

In the next stage of the workflow, we are going to write a software program
that utilizes the previously launched databases. In SmartSim terms, the application
is called a Model.

To run a software program, you must specify the run settings: run command and
source code file. In this example, we invoke the Python 3 interpreter to run a
python script. Create the run settings using the create_run_settings() function
provided by the exp (Experiment) object. Specify the path to the script to the argument
`exe_args` and the executable to the `exe_ex` argument.

.. code-block:: python
  srun_settings = exp.create_run_settings(exe="python3", exe_args="/path/to/file.py")

Next, set the number of nodes and tasks per node for the ‘Model’. For the application (Model),
the number of nodes determines the distribution of computational tasks while tasks per node
specifies how many individual instances of the program (tasks) can run on each node. In this
example, we specify to SmartSim that we intend to execute the `Model` once on a single node.

.. code-block:: python
  srun_settings.set_nodes(1)
  srun_settings.set_tasks_per_node(1)

Next, create an instance of the Model object using the create_model() function of the
exp object. Pass in the srun_settings object, which was configured earlier, as an argument
to the create_model() function and assign to the variable ‘model’.

.. code-block:: python
  model = exp.create_model("tutorial-model", srun_settings)

Next, launch the model instance using the exp.start() function.

.. code-block:: python
  exp.start(model, block=True, summary=True)

Next in the example, add SmartRedis multi-database functionality to the Model.
Navigate to the file specified in the exe_args argument of the create_run_settings()
function, exe_args="/path/to/model_multidb_example.py".

To begin, learn to retrieve data from the launched databases by using a SmartRedis Client
and the database identifier.In the Model script, we do not have access to the
Experiment.get_address() function. To establish a connection with the launched databases,
we still need to initialize a SmartRedis client for each Orchestrator with the address of
the launched database.

Previously, we added the db_identifier argument when creating the two databases.
When a database is created with a db_identifier name, SmartSim creates an environment
variable named SSDB for each database with the db_identifier variable value suffixed to
SSDB. In the case of the example, the environment variables created are SSDB_std-deployment
and SSDB_clus-deployment.

To create a SmartRedis Client for each database, first use the
ConfigOptions.create_from_environment() factory method, which accepts the
suffix to be applied to the SSDB environment variable. Then pass the ConfigOptions
object when creating the SmartRedis client.

For the standalone database:

.. code-block:: python
  std_config = ConfigOptions.create_from_environment("std-deployment")
  std_db_client = Client(std_config, "client_1")

For the clustered database:

.. code-block:: python
  clus_config = ConfigOptions.create_from_environment("clus-deployment")
  clus_db_client = Client(clus_config, "client_2")

These steps allow us to set up SmartRedis clients for each Orchestrator and
establish effective communication with the respective databases.

To confirm a successful connection to each database, retrieve the tensors previously
stored using the created SmartRedis clients. The `get_tensor` Client method allows
retrieval of a tensor by passing in the tensor name.

.. code-block:: python
  val1 = std_db_client.get_tensor("tensor1")
  val2 = clus_db_client.get_tensor("tensor2")
  print(f"{val1} + {val2}")

