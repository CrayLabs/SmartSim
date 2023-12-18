*****
Model
*****
========
Overview
========
SmartSim Models are an abstract representation for compiled applications,
scripts, and general computational tasks. Models can be launched with
other SmartSim entities and ML infrastructure to build AI-enabled workflows.
SmartSim enables AI integration in ``Model`` workflows by allowing users to dynamically load
TensorFlow (TF) functions, TorchScript, and various machine learning models
(TF, TF-lite, PyTorch, or ONNX) into the database at runtime.
Models are flexible enough to support many different applications, however,
to be used with SmartSim clients (SmartRedis) the application must be written
in Python, C, C++, or Fortran.

When initializing a ``Model``, users
provide executable simulation code to the Models run settings as well as
execution instructions with regard to the workload
manager (e.g. Slurm) and available compute resources. Users can specify
and assign values to parameters used within the simulation via the `params`
initialization argument. Parameters supplied in the `params` argument can either be
written into supplied configuration files or be assigned within configuration files
for use within the simulation at runtime. This functionality is supported by the
``Model.attach_generator_files()`` helper function users have access to
once creating a Model instance.

SmartSim supports **two** strategies for deploying Models:

- **Standard Model**: Operating on a separate compute node from a
  SmartSim orchestrator, Standard Models facilitate communication with a clustered
  Orchestrator through SmartRedis clients. This architecture is ideal for simulations
  leveraging distributed computing capabilities, ensuring parallelized and efficient
  execution across multiple nodes.

- **Colocated Model**: Sharing compute resources with a colocated Orchestrator on the
  same compute node, Colocated Models offer advantages in scenarios where minimizing communication
  latency is critical, such as online inference or runtime processing.

SmartSim manages ``Model`` instances through the :ref:`Experiment API<experiment_api>` by providing functions to
launch, monitor, and stop simulations. Additionally, Models can be launched individually
or as a group via an Ensemble. Once a Model instance has been initialized, users have access to
the :ref:`Model API<model_api>` helper functions.

======================
Initialize Model Types
======================
--------
Overview
--------

The :ref:`Experiment API<experiment_api>` is responsible for initializing all workflow components.
A ``Model`` is created using the ``Experiment.create_model()`` helper function. Users can customize the
the Model by specifying initializer arguments.

The key initializer arguments are:

-  `name` (Type: str): Specify the name of the model for unique identification.
-  `run_settings` (Type: RunSettings): Describe execution settings for a Model.
-  `params` (dict, optional): Provides a dictionary of parameters for Models.
-  `path` (str, optional): Path to where the model should be executed at runtime.
-  `enable_key_prefixing` (bool, optional): Prefix the model name to data sent to the database to prevent key collisions. Default is True.
-  `batch_settings` (BatchSettings | None): Describes settings for batch workload treatment.

To initialize a ``Model``, a `name` and :ref:`RunSettings<settings-info>` reference is required.
To instruct a Model to encapsulate a simulation, users must provide the simulation
executable to the Models run settings with instructions on how the application should be launched.

Users may also launch Models as a batch job by specifying :ref:`BatchSettings<settings-info>` when initializing.
When a Model with a ``BatchSettings`` reference is added to an Ensemble with a ``BatchSettings`` reference,
the Models batch settings are strategically ignored.

The `params` init argument in for Models lets users define simulation parameters and their
values through a dictionary. Using :ref:`Model API functions<model_api>`, users can write these parameters to
a file in the Model's working directory. Similarly, the Model API provides functionality to
instruct SmartSim to read a file and automatically fill in parameter values. Both behaviors are
provided by the ``Model.attach_generator_files()`` function.

Additionally, it's important to note that Model instances will be executed in the
current working directory by default if no `path` argument is supplied. When a Model
instance is passed to ``Experiment.generate()``, a directory within the Experiment directory
is automatically created to store input and output files from the model.

----------------
A Standard Model
----------------
For standard model deployment, the model runs
on separate compute nodes from SmartSim orchestrators.
Standard models are able to communicate with clustered or standalone
databases. SmartRedis clients connect to a clustered Orchestrator from
within the Models executable script
and travel off the simulation compute node to the database compute node to send/retrieve
data. To use a database within the standard model workload, a
SmartSim Orchestrator must be launched prior to the start of the Model.
Standard models are ideal for simulations benefiting from
distributed computing capabilities, enabling efficient
parallelized execution across multiple compute nodes.

In the `Instructions` subsection, we provide an example illustrating the deployment of a standard model.

Instructions
------------
This example provides a demonstration of how to initialize and launch a standard Model
within the Experiment workflow.

All workflow entities are initialized through the Experiment API. To create a Model,
the function ``Experiment.create_model()`` is used. To create Model, we must first
initialize an instance of the Experiment class. Import the SmartSim Experiment module
and initialize an Experiment instance. Provide a `name` and `launcher`:

.. code-block:: python

    from smartsim import Experiment

    # Init Experiment and specify to launch locally
    exp = Experiment(name="getting-started", launcher="local")

Models require run settings objects which are initialized through an Experiment instance.
Use the experiment instance, `exp`, to access the helper method ``Experiment.create_run_settings()``
to initialize a ``RunSettings`` object. In this example, we use the Python
executable to execute the application script, "script.py":

.. code-block:: python

    settings = exp.create_run_settings(exe="python", exe_args="script.py")

We now have an instance of run settings name `settings`. Using the instance, instruct SmartSim
to encapsulate the simulation as a Model through ``Experiment.create_model()``:

.. code-block:: python

    model = exp.create_model(name="example-model", run_settings=settings)

Remember that all entities are launched, monitored and stopped by the Experiment instance.
Specify the model instance, `model`, to ``Experiment.start()`` to start the standard model:

.. code-block:: python

    exp.start(model)

When the Experiment Python driver script is executed, two files from the standard model will be created
in the Experiment working directory:

1. `model.out` : this file will hold outputs produced by the Model workload
2. `model.err` : will hold any errors that occurred during Model execution

-----------------
A Colocated Model
-----------------
A colocated model is when both the application and the
database share compute resource. To create a colocated model,
users first initialize a Model instance with the ``Experiment.create_model()`` function.
A user must then colocate the model with the Model API function ``Model.colocate_db()``.
This instructs SmartSim to launch an Orchestrator on the same compute
nodes as the model when the model instance is deployed via ``Experiment.start()``.

There are **three** different Model API helper functions to colocate a Model:

- ``Model.colocate_db()``: An alias for `Model.colocate_db_tcp()`.
- ``Model.colocate_db_tcp()``: Colocate an Orchestrator instance with this Model over TCP/IP.
- ``Model.colocate_db_uds()``: Colocate an Orchestrator instance with this Model over UDS.

Each function will initialize an unsharded database to the Model instance. Only the partnered Model
may communicate with the database via a SmartRedis client by using the loopback TCP interface or
Unix Domain sockets. Extra parameters for the database can be passed into the helper functions
via kwargs.

.. code-block:: python

    example_kwargs = {
        "maxclients": 100000,
        "threads_per_queue": 1,
        "inter_op_threads": 1,
        "intra_op_threads": 1
    }

For a walkthrough of how to colocate a Model, navigate to the :ref:`Colocated Orchestrator<link>` for
instructions.

=====
Files
=====
--------
Overview
--------
The ``Model.attach_generator_files()`` function enables users to write model
parameters to a file or attachment a file of parameters with values to the
Model entity for use within the simulation.
Users can specify whether to copy, symlink, or configure files by passing parameters
one of the following parameters: `to_copy`, `to_symlink`, or `to_configure`.
It's worth noting that invoking
this function, ``attach_generator_files()``, post-attachment overwrites the existing list of entity files.

To write Model parameters to a file and attach to the Model, pass one of the
following values to the helper function:

* `to_copy`: Files that are copied into the path of the entity.
* `to_symlink`: Files that are symlinked into the path of the entity.

To attach a file to the Model and read the contents for use within the simulation, pass the
following value to the helper function:

* `to_configure`: Designed for text-based model input files,
  "to_configure" is exclusive to models. These files contain parameters for
  the model, with customizable tags corresponding to values users intend to
  modify. The default tag is a semicolon (e.g., THERMO = ;10;).

In the `Example` subsection, we provide an example illustrating using the value `to_configure`
within ``attach_generator_files()``.

-------
Example
-------
This example provides a demonstration of how to instruct SmartSim to attach a file
containing parameters for use within the Model using ``Model.attach_generator_files()``
and specifying `to_configure`.

We have a text file named `configure_inputs.txt`. Within the text, is the parameter, `THERMO`,
that is assigned a single value:

.. code-block:: txt

   THERMO = ;10;

The parameter `THERMO` is used within our application script and we would like to
instruct SmartSim to assign the parameter value upon Model execution.

To encapsulate our simulation using a Model, we must create an Experiment instance
to gain access to the Experiment helper function that creates the Model.
Begin by importing the Experiment module and SmartSim log module.
Initialize an Experiment:

.. code-block:: python

    from smartsim import Experiment
    from smartsim.log import get_logger

    logger = get_logger("Experiment Log")
    # Initialize the Experiment
    exp = Experiment("getting-started", launcher="auto")

Models require run settings. Create a simple ``RunSettings`` object, specifying
the path to our application script and the executable to run the script:

.. code-block:: python

    # Initialize a RunSettings object
    model_settings = exp.create_run_settings(exe="python", exe_args="/path/to/application.py")

Next, initialize a ``Model`` object with ``Experiment.create_model()``
and pass in the `model_settings` instance:

.. code-block:: python

    # Initialize a Model object
    model = exp.create_model("model", model_settings)

We now have a ``Model`` instance named `model`. Attach the above text file
to the Model for use at entity runtime. To do so, we use the
``Model.attach_generator_files()`` function and specify the `to_configure`
parameter with the path to the text file, `configure_params.txt`:

.. code-block:: python

    # Attach the file to the Model instance
    model.attach_generator_files(to_configure="path/to/configure_params.txt")

When we launch the Model, using ``Experiment.start()``, the file, `configure_inputs.txt`,
will be in the same file path as the application and SmartSim will read and assign
Model parameters from the text file.

=====================
ML Models and Scripts
=====================
--------
Overview
--------
SmartSim supports sending TorchScript functions, scripts, and
TensorFlow (TF), TensorFlow Lite (TF-lite), PyTorch (PT), or ONNX models to the database at runtime
prior to the execution of a Model for use within the workload.
The ``Model API`` provides a subset of helper functions that support
these capabilities:

* ``Model.add_ml_model()`` : Load a TF, TF-lite, PT, or ONNX model into the DB at runtime.
* ``Model.add_function()`` : Launch a TorchScript function with the Model.
* ``Model.add_script()`` : Launch a TorchScript with the Model.

---------
AI Models
---------
When configuring a Model, users can instruct SmartSim to load
TensorFlow (TF), TensorFlow Lite (TF-lite), PyTorch (PT), or ONNX
models dynamically to the database (colocated or standard). Machine Learning (ML) models added
are loaded prior to the execution of the model. SmartSim users may
provide the model in memory or specify the file path via the
``Model.add_ml_model()`` Model API helper function.

When specifying an ML model using ``Model.add_ml_model()``, the
following arguments are offered:

- name (str): key to store model under
- backend (str): name of the backend (TORCH, TF, TFLITE, ONNX)
- model (byte string, optional): A model in memory (only supported for non-colocated orchestrators)
- model_path (file path to model): serialized model
- device (str, optional): name of device for execution, defaults to “CPU”
- devices_per_node (int): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- first_device (int): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- batch_size (int, optional): batch size for execution, defaults to 0
- min_batch_size (int, optional): minimum batch size for model execution, defaults to 0
- min_batch_timeout (int, optional): time to wait for minimum batch size, defaults to 0
- tag (str, optional): additional tag for model information, defaults to “”
- inputs (list[str], optional): model inputs (TF only), defaults to None
- outputs (list[str], optional): model outputs (TF only), defaults to None

These arguments provide details to add and configure
ML models within the model simulation.

Example: Loading an In-Memory ML Model to the Model
---------------------------------------------------
In this example, we demonstrate how to instruct SmartSim to load
an in-memory ML model into the database at model runtime.

**Python Script: Creating a Keras CNN for Model Purposes**

To create an in-memory ML model, define the Model within the Python driver script.
For the purpose of the example, we define a Keras CNN within the experiment.

.. code-block:: python

    def create_tf_cnn():
        """Create a Keras CNN for model purposes

        """
        from smartsim.ml.tf import serialize_model
        n = Net()
        input_shape = (3,3,1)
        inputs = Input(input_shape)
        outputs = n(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs, name=n.name)

        return serialize_model(model)

    # Get and save TF model
    model, inputs, outputs = create_tf_cnn()

**SmartSim Model Integration:**

Assuming an initialized ``Model`` named `smartsim_model`, we specify the
following parameters to the ``Model.add_ml_model()`` function:

.. code-block:: python

    smartsim_model.add_ml_model(name="cnn", backend="TF", model=model, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)

In this integration, we provide the following details:

-  `name`: "cnn" - A key to uniquely identify the model within the database.
-  `backend`: "TF" - Indicating that the model is a TensorFlow model.
-  `model`: model - The in-memory representation of the TensorFlow model.
-  `device`: "GPU" - Specifying the device for ML model execution.
-  `devices_per_node`: 2 - Use two GPUs per node.
-  `first_device`: 0 - Start with 0 index GPU.
-  `inputs`: inputs - The names of the model inputs.
-  `outputs`: outputs - The names of the model outputs.

When the Model is started via ``Experiment.start()``, the ML model will be loaded to the
standard orchestrator that is launched prior to the start of the Model.

---------------------
TorchScript functions
---------------------
Users can instruct SmartSim to upload TorchScript functions to the database
at runtime. Script functions are loaded into
standard orchestrators prior to the execution of Models. If using a
colocated orchestrator, use the ``Model.add_script()`` function.
Users have the flexibility to choose
between `"GPU"` or `"CPU"` for device selection, and in environments with multiple
devices, specific device numbers can be specified via `devices_per_node`.

When specifying a TF function using ``Model.add_function()``, the
following arguments are offered:

- name (str): key to store function under
- function (str, optional): TorchScript function code
- device (str, optional): device for script execution, defaults to “CPU”
- devices_per_node (int): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- first_device (int): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

Example: Loading an TensorFlow Function to the Model
----------------------------------------------------
In this example, we demonstrate how to instruct SmartSim to load
an TensorFlow function into the database at model runtime. It's
important to note the function, ``Model.add_function()`` is supported
for non-colocated deployments and during a colocated deployment, ``Model.add_script()``
should be used.

**Python Script: Define a TF Function for Model Purposes**
To load a TF function, define the function within the Python driver script.

.. code-block:: python

    def timestwo(x):
        return 2*x

**SmartSim Model Integration:**

Assuming an initialized ``Model`` named `smartsim_model`, we specify the
following parameters to the ``Model.add_function()`` function:

.. code-block:: python

    smartsim_model.add_function(name="example_func", function=timestwo, device="GPU", devices_per_node=2, first_device=0)

In this integration, we provide the following details:

-  `name`: "example_func" - A key to uniquely identify the model within the database.
-  `function`: timestwo - Name of the TorchScript function defined in the Python driver script.
-  `device`: "CPU" - Specifying the device for ML model execution.
-  `devices_per_node`: 2 - Use two GPUs per node.
-  `first_device`: 0 - Start with 0 index GPU.

When the model is started via ``Experiment.start()``, the TF function will be loaded to the
standard orchestrator that is launched prior to the start of the model.

-------------------
TorchScript Scripts
-------------------
SmartSim supports the execution of TorchScripts
with Models. Regardless of whether
the orchestrator is colocated or not, each script added to the Model is
loaded into the orchestrator prior to the execution of entity.
Users have the flexibility to choose
between `"GPU"` or `"CPU"` for device selection, and in environments with multiple
devices, specific device numbers can be specified via `devices_per_node` such as `"GPU:1,"`.

When specifying a TorchScript using ``Model.add_script()``, the
following arguments are offered:

- name (str): key to store script under
- script (str, optional): TorchScript code (only supported for non-colocated orchestrators)
- script_path (str, optional): path to TorchScript code
- device (str, optional): device for script execution, defaults to “CPU”
- devices_per_node (int): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- first_device (int): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

You might use TorchScript scripts to represent individual models within the model.

Example: Loading an TorchScript to the Model
--------------------------------------------
In this example, we demonstrate how to instruct SmartSim to load
an TorchScript into the database at model runtime.

**Python Script: Define a TorchScript for Model Purposes**

Define the TorchScript code to a variable in the Python driver script:

.. code-block:: python

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

**SmartSim Model Integration:**

Assuming an initialized ``Model`` named `smartsim_model`, we specify the
following parameters to the ``Model.add_script()`` function:

.. code-block:: python

    smartsim_model.add_script(name="example_script", script=torch_script_str, device="GPU", devices_per_node=2, first_device=0)

In this integration, we provide the following details:

-  `name`: "example_script" - key to store script under
-  `script`: torch_script_str - TorchScript code
-  `device`: "CPU" - device for script execution
-  `devices_per_node`: 2 - Use two GPUs per node.
-  `first_device`: 0 - Start with 0 index GPU.

When the model is started via ``Experiment.start()``, the TorchScript will be loaded to the
orchestrator that is launched prior to the start of the model.

=========================
Data Collision Prevention
=========================
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

1. producer_1.py : a Model producer application
2. producer_2.py : a Model producer application
3. consumer.py : a Model consumer application
4. experiment.py : the Experiment driver script

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

Finally, launch the model instances:

.. code-block:: python

    exp.start(model_1, model_2, model_3, block=True, summary=True)

To end, tear down the database:

.. code-block:: python

    exp.stop(single_shard_db)
    logger.info(exp.summary())