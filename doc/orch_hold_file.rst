************
Orchestrator
************

========
Overview
========
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

======================
Clustered Orchestrator
======================
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

Example
-------
This example provides a demonstration on automating the deployment of
a standard Orchestrator, connecting a SmartRedis Client from
within the

The Application Script
----------------------

To begin writing the application script, import the necessary packages:
.. code-block:: python

  from smartredis import Client, log_data
  from smartredis import *
  import numpy as np

Initialize the Client
^^^^^^^^^^^^^^^^^^^^^
To establish a connection with the standard database,
we need to initialize a new SmartRedis client.
Since the standard database we launch in the driver script
multi-sharded, specify `cluster` as `True`:

.. code-block:: python

  # Initialize a Client
  standard_db_client = Client(cluster=True)

Retrieve Data
^^^^^^^^^^^^^
To confirm a successful connection to the database, we will retrieve the tensor
that we store in the python driver script.
Use the ``Client.get_tensor()`` method to
retrieve the tensor by specifying the name `tensor_1` we
used during ``Client.put_tensor()`` in the driver script:
.. code-block:: python

  # Retrieve tensor from driver script
    value_1 = standard_db_client.get_tensor("tensor_1")
    # Log tensor
    standard_db_client.log_data(LLInfo, f"The single sharded db tensor is: {value_1}")

Later, when you run the experiment driver script the following output will appear in ``model.out``
located in ``getting-started-multidb/tutorial_model/``::
    Default@17-11-48:The single sharded db tensor is: [1 2 3 4]

Store Data
^^^^^^^^^^
Next, create a NumPy tensor to send to the standard database to retrieve
in the driver script by using  ``Client.put_tensor(name, data)``:
.. code-block:: python

  # Create NumPy array
  array_2 = np.array([5, 6, 7, 8])
  # Use SmartRedis client to place tensor in single sharded db
  standard_db_client.put_tensor("tensor_2", array_2)

The Experiment Driver Script
----------------------------
To run the previous application, we must define workflow stages within a workload.
Defining workflow stages requires the utilization of functions associated
with the ``Experiment`` object. The Experiment object is intended to be instantiated
once and utilized throughout the workflow runtime.
In this example, we instantiate an ``Experiment`` object with the name ``getting-started-multidb``.
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
  exp = Experiment("tester", launcher="auto")

Launch Standard Orchestrator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In the context of this ``Experiment``, it's essential to create and launch
the databases as a preliminary step before any other components since
the application script requests tensors from the launched databases.

We aim to showcase the multi-database automation capabilities of SmartSim, so we
create two databases in the workflow: a single-sharded database and a
multi-sharded database.
Step 1: Initialize Orchestrator
"""""""""""""""""""""""""""""""
To create an database, utilize the ``Experiment.create_database()`` function.
.. code-block:: python

  # Initialize a multi sharded database
  standard_db = exp.create_database(port=6379, db_nodes=3, interface="ib0")
  exp.generate(standard_db, overwrite=True)

Step 2: Start Databases
"""""""""""""""""""""""
Next, to launch the databases,
pass the database instances to ``Experiment.start()``.
.. code-block:: python

  # Launch the multi sharded database
  exp.start(standard_db)

The ``Experiment.start()`` function launches the ``Orchestrators`` for use within the workflow. In other words, the function
deploys the databases on the allocated compute resources.

Create Client Connections to Orchestrator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The SmartRedis ``Client`` object contains functions that manipulate, send, and receive
data within the database. Each database has a single, dedicated SmartRedis ``Client``.
Begin by initializing a SmartRedis ``Client`` object per launched database.

To create a designated SmartRedis ``Client``, you need to specify the address of the target
running database. You can easily retrieve this address using the ``Orchestrator.get_address()`` function.

.. code-block:: python

  # Initialize SmartRedis client for multi sharded database
  driver_client_standard_db = Client(cluster=True, address=standard_db.get_address()[0])

Store Data Using Clients
^^^^^^^^^^^^^^^^^^^^^^^^
In the application script, we retrieved two NumPy tensors.
To support the apps functionality, we will create two
NumPy arrays in the python driver script and send them to the a database. To
accomplish this, we use the ``Client.put_tensor()`` function with the respective
database client instances.
.. code-block:: python

  # Create NumPy array
  array_1 = np.array([1, 2, 3, 4])
  # Use single shard db SmartRedis client to place tensor in single sharded db
  driver_client_standard_db.put_tensor("tensor_1", array_1)

Initialize a Model
^^^^^^^^^^^^^^^^^^
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
.. code-block:: python

  # Initialize a RunSettings object
  model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/lus/scratch/richaama/standard_orch_model.py")
  model_settings.set_nodes(1)

Step 2: Initialize
""""""""""""""""""
Next, create a ``Model`` instance using the ``Experiment.create_model()``.
Pass the ``model_settings`` object as an argument
to the ``create_model()`` function and assign to the variable ``model``.
.. code-block:: python

  # Initialize the Model
  model = exp.create_model("model", model_settings)

Step 3: Start
"""""""""""""
Next, launch the colocated model instance using the ``Experiment.start()`` function.
.. code-block:: python

  # Launch the Model
  exp.start(model, block=True, summary=True)

Retrieve Data Using Clients
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python

  # Retrieve the tensors placed by the Model
  value_2 = driver_client_standard_db.poll_key("tensor_2", 100, 100)
  # Validate that the tensor exists
  logger.info(f"The tensor is {value_2}")

Cleanup Experiment
^^^^^^^^^^^^^^^^^^
.. code-block:: python

  # Cleanup the database
  exp.stop(standard_db)
  logger.info(exp.summary())

How to Run the Example
----------------------
Source Code
-----------
.. sourcecode::
======================
Colocated Orchestrator
======================
A co-located Orchestrator is a special type of Orchestrator that is deployed on
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

Example
-------
The Application Script
----------------------
.. code-block:: python

  from smartredis import ConfigOptions, Client, log_data
  from smartredis import *
  import numpy as np

Initialize the Clients
^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: python
  # Initialize a Client
  colo_client = Client(cluster=False)

Store Data
^^^^^^^^^^
.. code-block:: python
    # Create NumPy array
    array_1 = np.array([1, 2, 3, 4])
    # Use SmartRedis client to place tensor in single sharded db
    colo_client.put_tensor("tensor_1", array_1)
Retrieve Data
^^^^^^^^^^^^^
.. code-block:: python
    # Retrieve tensor from driver script
    value_1 = colo_client.get_tensor("tensor_1")
    # Log tensor
    colo_client.log_data(LLInfo, f"The colocated db tensor is: {value_1}")

The Experiment Driver Script
----------------------------
.. code-block:: python
    import numpy as np
    from smartredis import Client
    from smartsim import Experiment
    from smartsim.log import get_logger
    import sys

    exe_ex = sys.executable
    logger = get_logger("Example Experiment Log")
    # Initialize the Experiment
    exp = Experiment("tester", launcher="auto")

Initialize a Colocated Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Step 1: Configure
"""""""""""""""""
.. code-block:: python
    # Initialize a RunSettings object
    model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/lus/scratch/richaama/clustered_model.py")
    # Configure RunSettings object
    model_settings.set_nodes(1)

Step 2: Initialize
""""""""""""""""""
.. code-block:: python
    # Initialize a SmartSim Model
    model = exp.create_model("colo_model", model_settings)
Step 2: Colocate
""""""""""""""""
.. code-block:: python
    # Colocate the Model
    model.colocate_db_tcp()

Step 3: Start
"""""""""""""
.. code-block:: python
    # Launch the colocated Model
    exp.start(model, block=True, summary=True)
Cleanup Experiment
^^^^^^^^^^^^^^^^^^
.. code-block:: python

    logger.info(exp.summary())

How to Run the Example
----------------------
Source Code
-----------

======================
Multiple Orchestrators
======================

Example
-------

Source Code
-----------