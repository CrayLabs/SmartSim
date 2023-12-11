*****
Model
*****
========
Overview
========
Model(s) are subclasses of SmartSimEntity(s) and are created through the
Experiment API. Models represent any computational kernel. Models are flexible
enough to support many different applications, however, to be used with our
clients (SmartRedis) the application will have to be written in Python, C, C++, or Fortran.

Models are given RunSettings objects that specify how a kernel should be executed
with regard to the workload manager (e.g. Slurm) and the available compute resources
on the system.

=====
Files
=====
--------
Overview
--------
SmartSim enables users to attach files to a ``Model`` for use by the entity
when launched by ``Experiment.start()`` via the ``Model.attach_generator_files()`` function.
When the ``Model`` is generated, the
files will be located in the path of the entity. Invoking this method after
files have already been attached will overwrite the previous list of entity files.

The function ``Model.attach_generator_files()`` accepts three parameters: `to_copy`, `to_symlink`
and `to_configure`.

* `to_copy`: files “to_copy” are copied into the path of the entity
* `to_symlink`: files “to_symlink” are symlinked into the path of the entity
* `to_configure`: text based model input files where parameters for the model
  are set. Note that only models support the “to_configure” field. These files
  must have fields tagged that correspond to the values the user would like to change.
  The tag is settable but defaults to a semicolon e.g. THERMO = ;10;

-------
Example
-------
Below we will provide a simple example on how to use the
``Model.attach_generator_files()`` `to_configure` argument,
to attach a file to a Model that will assign variables
existing in the application.

For the Model, we will be attached the following text file
named `configure_inputs.txt`.

.. code-block:: txt

   THERMO = ;10;

The text file has a single variable assignment `THERMO=10`.

Begin by importing the necessary modules and initializing
a SmartSim ``Experiment`` object:

.. code-block:: python

    from smartsim import Experiment
    from smartsim.log import get_logger

    logger = get_logger("Experiment Log")
    # Initialize the Experiment
    exp = Experiment("getting-started", launcher="auto")

In order to create a ``Model``, we must first create a ``RunSettings``
object to specify to SmartSim, run parameters for the application.
Begin by initializing a ``RunSettings`` object:

.. code-block:: python

    # Initialize a RunSettings object
    model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/application")

Next, initialize a ``Model`` object and pass in the `model_settings`:

.. code-block:: python

    # Initialize a Model object
    model = exp.create_model("model", model_settings)

Now that we have create the Model object, we have access to the function,
``Model.attach_generator_files()``. Use the ``attach_generator_files()``
function with the `to_configure` parameter to specify the path to the text
file from above, `configure_inputs.txt`:

.. code-block:: python

    # Attach value configuration file
    model.attach_generator_files(to_configure="path/to/configure_inputs.txt")

Now, when we execute the Model, the file, `configure_inputs.txt`, will be in the
same file path as the application and will fill in the application arguments with
the values set from within the text file.

===============
Colocated Model
===============
During colocated deployment, a Model and database share the same compute resources.
Meaning, a SmartRedis client does not have to travel off the compute node to access
either the database or model since they exist on the same compute node. During an
experiment, if a SmartSim user colocates a Model, then an Orchestrator will be
launched alongside on the resources allocated for the Model by the run settings object.

You may colocate a Model after initializing a ``Model`` object via the ``Model.colocate_db()``
function. If you would like to colocate an Orchestrator instance with the Model over TCP/IP,
use ``Model.colocate_db_tcp()``. To colocate an Orchestrator instance with the Model over UDS,
use the function ``Model.colocate_db_uds()``.

For an example of how to colocate a Model, navigate to the :ref:`Colocated Orchestrator<link>`
instructions.

===================
Model Key Prefixing
===================
--------
Overview
--------
SmartSim and SmartRedis can avoid key collision by prepending program-unique
prefixes to Model workloads launched through SmartSim. For example, if you were
to have two applications feeding data to a single database, who produced keys
of the same name, upon requesting this information there would be a key collision
since there is no yet uniqueness between the same tensor names. By enabling key
prefixing on a Model, SmartSim will append the model name to each key produced
by the application and sent to the database as such: `model_name.tensor_name`.

This is done simply through functions offered by the Model API:

* Model.register_incoming_entity(incoming_entity)
* Model.enable_key_prefixing()
* Model.disable_key_prefixing()
* Model.query_key_prefixing()

-------
Example
-------
We provide a producer/consumer example that demonstrates
two producer models, with key prefixing enabled, that
send tensor of the same name to a standard database. A
consumer Model requests both tensors and displays the
information using a SmartRedis logger to demonstrates
successful use of Model key prefixing.

During the example we will be creating four different files:

1. experiment.py : the Experiment driver script
2. producer_1.py : a Model producer application
3. producer_2.py : a Model producer application
4. consumer.py : a Model consumer application

Producer_1 Application
======================
Begin by importing the necessary modules and initializing a SmartRedis
Client:

.. code-block:: python

    from smartredis import Client
    from smartredis import *
    import numpy as np

    # Initialize a Client
    client = Client(cluster=False)

Next, create a NumPy array to place in the database with the key name `tensor`:

.. code-block:: python

    # Create NumPy array
    array = np.array([5, 6, 7, 8])
    # Use SmartRedis client to place tensor in single sharded db
    client.put_tensor("tensor", array)

With key prefixing enabled, the tensor stored will be under key
`producer_1.tensor`.

Producer_2 Application
======================
Begin by importing the necessary modules and initializing a SmartRedis
Client:

.. code-block:: python

    from smartredis import Client
    from smartredis import *
    import numpy as np

    # Initialize a Client
    client = Client(cluster=False)

Next, create a NumPy array to place in the database with the key name `tensor`:

.. code-block:: python

    # Create NumPy array
    array = np.array([1, 2, 3, 4])
    # Use SmartRedis client to place tensor in single sharded db
    client.put_tensor("tensor", array)

With key prefixing enabled, the tensor stored will be under key
`producer_2.tensor`.

Consumer Application
====================
Next, request the inputted tensors from the producer
applications within the consumer application.
Begin by importing the necessary modules and initializing a SmartRedis
Client:

.. code-block:: python

    from smartredis import Client
    from smartredis import *
    import numpy as np

    # Initialize a Client
    client = Client(cluster=False)

SmartRedis offers the function, ``Client.set_data_source()`` that
will add a prefix to the key name when using ``Client.get_tensor()``.

.. code-block:: python

    client.set_data_source("producer_1")
    # Searching for key name: producer_1.tensor
    client.get_tensor("tensor")
    client.set_data_source("producer_2")
    # Searching for key name: producer_2.tensor
    client.get_tensor("tensor")

Next, output the tensor values to validate correctness:

.. code-block:: python

    client.log_data(LLInfo, f"1: {val1}")
    client.log_data(LLInfo, f"2: {val2}")

The Experiment Driver script
============================
Begin by initializing a SmartSim Experiment object
with the required import modules and launching
a standalone database:

.. code-block:: python

    from smartsim import Experiment
    from smartsim.log import get_logger

    logger = get_logger("Experiment Log")
    # Initialize the Experiment
    exp = Experiment("getting-started", launcher="auto")

    # Initialize a single sharded database
    single_shard_db = exp.create_database(port=6379, db_nodes=1, interface="ib0")
    exp.generate(single_shard_db, overwrite=True)
    exp.start(single_shard_db)

Next, lets create each Model with runsettings and key prefixing
beginning with model 1: `producer_1.py`. Initialize the run
settings for the Model, then create the Model:

.. code-block:: python

    # Initialize a RunSettings object
    producer_settings_1 = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/producer_1.py")
    producer_settings_1.set_nodes(1)
    producer_settings_1.set_tasks_per_node(1)
    producer_1 = exp.create_model("producer_1", producer_settings_1)

Since this model will be producing and sending tensors to the standard Orchestrator,
we need to enable key prefixing like below:

.. code-block:: python

    producer_1.enable_key_prefixing()

Now repeat this process for the second producer model:
.. code-block:: python

    # Initialize a RunSettings object
    producer_settings_2 = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/producer_2.py")
    producer_settings_2.set_nodes(1)
    producer_settings_2.set_tasks_per_node(1)
    producer_2 = exp.create_model("producer_2", producer_settings_2)
    producer_2.enable_key_prefixing()

Next, lets create the consumer model that will be request the
keys from `producer_1` and `producer_2`:

.. code-block:: python

    # Initialize a RunSettings object
    consumer_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/consumer.py")
    consumer_settings.set_nodes(1)
    consumer_settings.set_tasks_per_node(1)
    consumer = exp.create_model("consumer", consumer_settings)

We do not need key prefixing enabled on this model, so we disable:

.. code-block:: python

    consumer.disable_key_prefixing()

Next, start the Experiment, then cleanup:

.. code-block:: python

    exp.start(model_1, model_2, model_3, block=True, summary=True)
    exp.stop(single_shard_db)
    logger.info(exp.summary())

===========
ML Features
===========
--------
Overview
--------
The SmartSim Model API offers three functions that enable ML and AI from within
the Experiment:

* ``Model.add_function()``
This Model helper function is used to launch TorchScript functions with Model
instances. Each script function to the model will be loaded into a non-converged
orchestrator prior to the execution of this Model instance. For converged orchestrators,
the add_script() method should be used. Device selection is either “GPU” or “CPU”.
If many devices are present, a number can be passed for specification e.g. “GPU:1”.
Setting devices_per_node=N, with N greater than one will result in the model being
stored in the first N devices of type device.

* ``Model.add_ml_model()``
A TF, TF-lite, PT, or ONNX model to load into the DB at runtime.
Each ML Model added will be loaded into an orchestrator (converged or not)
prior to the execution of this Model instance.
One of either model (in memory representation) or model_path (file) must be provided.

* ``Model.add_script()``
You must use TorchScript to launch with this Model instance.
Each script added to the model will be loaded into an orchestrator
(converged or not) prior to the execution of this Model instance.
Device selection is either “GPU” or “CPU”. If many devices are present,
a number can be passed for specification e.g. “GPU:1”.
Setting devices_per_node=N, with N greater than one will result in
the model being stored in the first N devices of type device.
One of either script (in memory string representation) or script_path
(file) must be provided.