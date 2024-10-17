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
is evenly distributed amongst all nodes in the cluster.

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
  are sharing the same compute node. For the clustered deployment, a shard occupies the entirety
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

* The :ref:`Application Script<The Application Script>`
* The :ref:`Experiment Driver Script<The Experiment Driver Script>`

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