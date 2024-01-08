*****
Model
*****
========
Overview
========
SmartSim ``Model`` objects enable users to execute computational tasks in an
``Experiment`` workflow, such as launching compiled applications,
running scripts, or performing general computational operations. ``Models`` can be launched with
other SmartSim entities and infrastructure to build AI-enabled workflows.
``Model`` objects can leverage ML capabilities by utilizing the SmartSim client (:ref:`SmartRedis<dead_link>`)
to transfer data to the ``Orchestrator``, enabling other running SmartSim ``Models`` to access the data.
Additionally, clients can execute ML models (TF, TF-lite, PyTorch, or ONNX) and TorchScripts stored in the
``Orchestrator``. SmartRedis is available in Python, C, C++, or Fortran.

To initialize a SmartSim ``Model``, use the ``Experiment.create_model()`` API function.
When creating a ``Model``, a :ref:`RunSettings<dead_link>` object must be provided. A ``RunSettings``
object specifies the ``Models`` executable simulation code (e.g. the full path to a compiled binary) as well as
application execution specifications. These specifications include :ref:`launch<dead_link>` commands (e.g. `srun`, `aprun`, `mpiexec`, etc),
compute resource requirements, and application command-line arguments.

When initializing a ``Model`` object, users can direct SmartSim to assign values at runtime to variables in the
application code. Simply specify the variables with their corresponding values in a Python dictionary and pass it to
the `params` factory method parameter during initialization. SmartSim then automatically writes these parameters and values into a
configuration file stored in the simulation's execution path.

SmartSim supports **two** strategies for deploying ``Models``:

1. **Standard Model**: When a Standard Model is launched, it does not use or share compute
resources on the same host (computer/server) where a SmartSim ``Orchestrator`` is running.
It is instead launched on its own compute resources.
Standard deployment is ideal for systems that have heterogeneous node types
(i.e. a mix of CPU-only and GPU-enabled compute nodes) where
ML model and TorchScript evaluation is more efficiently performed off-node. This
deployment is also ideal for workflows relying on data exchange between multiple
applications (e.g. online analysis, visualization, computational steering, or
producer/consumer application couplings).

2. **Colocated Model**: When the Colocated Model is launched, it shares compute resources with a colocated Orchestrator
on the same compute node. A colocated Model is ideal when the data and hardware accelerator
are located on the same compute node. This setup helps reduce latency in ML inference and TorchScript evaluation
by eliminating off-node communication.

Once a Model instance has been initialized, users have access to
the :ref:`Model API<model_api>` helper functions to further configure the ``Model``.
The models helper functions allow users to:

- attach files to a ``Model`` for use within the simulation
- launch a ``Orchestrator`` on the ``Models`` compute nodes
- add a ML model to launch with the ``Model`` instance
- add a TorchScript to launch with the ``Model`` instance
- add a TorchScript function to launch with the ``Model`` instance
- register communication with another ``Model``
- enable ``Model`` key collision prevention

SmartSim manages ``Model`` instances through the :ref:`Experiment API<experiment_api>` by providing functions to
launch, monitor, and stop applications. Additionally, Models can be launched individually
or as a group via an Ensemble.

====================
Model Initialization
====================
--------
Overview
--------

The :ref:`Experiment API<experiment_api>` is responsible for initializing all workflow entities.
A ``Model`` is created using the ``Experiment.create_model()`` factory method, and users can customize the
``Model`` via the factory method parameters.

The key initializer arguments are:

-  `name` (str): Specify the name of the model for unique identification.
-  `run_settings` (base.RunSettings): Describe execution settings for a Model.
-  `params` (t.Optional[t.Dict[str, t.Any]] = None): Provides a dictionary of parameters for Models.
-  `path` (t.Optional[str] = None): Path to where the model should be executed at runtime.
-  `enable_key_prefixing` (bool = False): Prefix the model name to data sent to the database to prevent key collisions. Default is True.
-  `batch_settings` (t.Optional[base.BatchSettings] = None): Describes settings for batch workload treatment.

A `name` and :ref:`RunSettings<dead_link>` reference are required to initialize a ``Model``.
Optionally, include a :ref:`BatchSettings<dead_link>` object to specify workload manager batch launching.

.. note::
    ``BatchSettings`` attached to a model are ignored when the model is executed as part of an ensemble.

The `params` factory method parameter for ``Models`` lets users define simulation parameters and their
values through a dictionary. Using :ref:`Model API<model_api>` functions, users can write these parameters to
a file in the Model's working directory.

.. note::
    Model instances will be executed in the current working directory by default if no `path` argument
    is supplied.

When a Model instance is passed to ``Experiment.generate()``, a
directory within the Experiment directory
is automatically created to store input and output files from the model.

--------------
Standard Model
--------------
A standard ``Model`` runs on separate compute nodes from SmartSim ``Orchestrators``.
A ``Model`` connects to an ``Orchestrator`` via the SmartSim client (:ref:`SmartRedis<dead_link>`).
For the client connection to be successful, the SmartSim ``Orchestrator`` must be launched prior to the start of the ``Model``.
Standard ``Model`` deployment is ideal for systems that have heterogeneous node types
(i.e. a mix of CPU-only and GPU-enabled compute nodes) where ML model and TorchScript
evaluation is more efficiently performed off-node. This deployment is also ideal for workflows
relying on data exchange between multiple applications (e.g. online analysis, visualization,
computational steering, or producer/consumer application couplings).

.. note::
    A ``Model`` can be launched without an ``Orchestrator`` if data transfer and ML capabilities are not
    required.

In the proceeding :ref:`Instructions<std_model_init_instruct>` subsection, we provide an example illustrating the deployment of a standard model.

.. _std_model_init_instruct:
Instructions
------------
This example provides a demonstration of how to initialize and launch a standard ``Model``
within an ``Experiment`` workflow. All workflow entities are initialized through the
:ref:`Experiment API<experiment_api>`. Consequently, initializing
a SmartSim ``Experiment`` is a prerequisite for ``Model`` initialization.

To initialize an instance of the ``Experiment`` class, import the SmartSim Experiment module and invoke the ``Experiment`` constructor
with a `name` and `launcher`:

.. code-block:: python

    from smartsim import Experiment

    # Init Experiment and specify to launch locally
    exp = Experiment(name="getting-started", launcher="local")

``Models`` require ``RunSettings`` objects. We use the `exp` instance to
call the factory method ``Experiment.create_run_settings()`` to initialize a ``RunSettings``
object. Finally, we specify the Python executable to run the executable simulation code named
`script.py`:

.. code-block:: python

    settings = exp.create_run_settings(exe="python", exe_args="script.py")

We now have a ``RunSettings`` instance named `settings` that we can use to create
a ``Model`` instance that contains all of the information required to launch our application:

.. code-block:: python

    model = exp.create_model(name="example-model", run_settings=settings)

To created an isolated output directory for the ``Model``, invoke ``Experiment.start()`` via the
``Experiment`` instance `exp` with `model` as an input parameter:

.. code-block:: python

    model = exp.generate(model)

Recall that all entities are launched, monitored and stopped by the ``Experiment`` instance.
To start ``Model``, invoke ``Experiment.start()`` via the ``Experiment`` instance `exp` with `model` as an
input parameter:

.. code-block:: python

    exp.start(model)

When the Experiment Python driver script is executed, two files from the standard model will be created
in the Experiment working directory:

1. `example-model.out` : this file will hold outputs produced by the Model workload
2. `example-model.err` : will hold any errors that occurred during Model execution

-----------------
A Colocated Model
-----------------
A colocated ``Model`` runs on the same compute node(s) as a SmartSim ``Orchestrator``.
With a colocated model, the Model and the Orchestrator share compute resources.
To create a colocated model,
users first initialize a ``Model`` instance with the ``Experiment.create_model()`` function.
A user must then colocate the ``Model`` with the Model API function ``Model.colocate_db()``.
This instructs SmartSim to launch an Orchestrator on the same compute
nodes as the model when the model instance is deployed via ``Experiment.start()``.

There are **three** different Model API helper functions to colocate a Model:

- ``Model.colocate_db_tcp()``: Colocate an Orchestrator instance and establish client communication using TCP/IP.
- ``Model.colocate_db_uds()``: Colocate an Orchestrator instance and establish client communication using UDS.
- ``Model.colocate_db()``: (deprecated) An alias for `Model.colocate_db_tcp()`.

Each function initializes an unsharded database accessible only to the model. When the model
is started, the database will be launched on the same compute resource as the model. Only the colocated Model
may communicate with the database via a SmartRedis client by using the loopback TCP interface or
Unix Domain sockets. Extra parameters for the database can be passed into the helper functions above
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
The ``Model.attach_generator_files()`` function enables the use of
external files with a model.

.. note::
    Post-attachment overwrites the existing list of entity files.

To attach a file to a model used within the simulation, provide one of the following values to the helper function:

* `to_copy` (t.Optional[t.List[str]] = None): Files that are copied into the path of the entity.
* `to_symlink` (t.Optional[t.List[str]] = None): Files that are symlinked into the path of the entity.

To specify a template file to the Model to store a dynamic set of `params` values at runtime, pass the
following value to the helper function:

* `to_configure` (t.Optional[t.List[str]] = None): Designed for text-based model input files,
  "to_configure" is exclusive to models. At runtime, these files are passed `params`
  used within the model, to replace customizable tags corresponding to values users intend to
  modify. The default tag is a semicolon (e.g., THERMO = ;THERMO;).

In the `Example` subsection, we provide an example illustrating using the value `to_configure`
within ``attach_generator_files()``.

-------
Example
-------
This example provides a demonstration of how to instruct SmartSim to attach a file
to load the Model `params` by using ``Model.attach_generator_files()`` to specify `to_configure`.

We have a text file named `params_inputs.txt`. Within the text, is the parameter, `THERMO`:

.. code-block:: txt

   THERMO = ;THERMO;

The parameter `THERMO` with value is passed in when initializing the Model. We would like to
instruct SmartSim to store the parameter value upon Model execution in `params_inputs.txt`
between the semi-colons.

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
    example_model = exp.create_model("model", model_settings, params={"THERMO":1})

We now have a ``Model`` instance named `model`. Attach the above text file
to the Model for use at entity runtime. To do so, we use the
``Model.attach_generator_files()`` function and specify the `to_configure`
parameter with the path to the text file, `params_inputs.txt`:

.. code-block:: python

    # Attach the file to the Model instance
    example_model.attach_generator_files(to_configure="path/to/params_inputs.txt")

Launching the model with ``exp.start(example_model)`` processes attached generator files. `configure_inputs.txt` will be
available in the model working directory and SmartSim will assign `example_model` `params` to the text file.

The contents of `params_inputs.txt` after Model completion are:

.. code-block:: txt

   THERMO = ;1;

=============
Model Outputs
=============
A Model creates two files when started:

* `<model_name>.out`
* `<model_name>.err`

The files are created in the working
directory. These files capture outputs and errors during execution. The filenames directly match the
model's name. The `<model_name>.out` file logs standard outputs and the
`<model_name>.err` logs errors for debugging.

.. note::
    You can move these files by specifying a string path using the `path` parameter when instantiating the model.
    You may also move these files by calling ``Experiment.generate(model)``. This will create a directory `model_name/`
    and will store the two files within the new folder.

=====================
ML Models and Scripts
=====================
--------
Overview
--------
The ``Model API`` provides a set of helper functions:

* ``Model.add_ml_model()``
* ``Model.add_function()``
* ``Model.add_script()``

The helper functions expose the capabilities:

* Load a TF, TF-lite, PT, or ONNX model into the DB at runtime
* Launch a TorchScript function with the Model.
* Launch a TorchScript with the Model.

--------
Runtimes
--------
* TensorFlow (TF)
* TensorFlow Lite (TF-lite)
* PyTorch (PT)
* ONNX

---------
AI Models
---------
When configuring a Model, users can instruct SmartSim to load
Machine Learning (ML) models dynamically to the database (colocated or standard). ML models added
are loaded prior to the execution of the model. SmartSim users may
provide the model in memory or specify the file path via the
``Model.add_ml_model()`` Model API helper function.

When specifying an ML model using ``Model.add_ml_model()``, the
following arguments are offered:

- name (str): key to store model under.
- backend (str): name of the backend (TORCH, TF, TFLITE, ONNX).
- model (t.Optional[str] = None): A model in memory (only supported for non-colocated orchestrators).
- model_path (t.Optional[str] = None): serialized model.
- device (t.Literal["CPU", "GPU"] = "CPU"): name of device for execution, defaults to “CPU”.
- devices_per_node (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- first_device (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- batch_size (int = 0): batch size for execution, defaults to 0.
- min_batch_size (int = 0): minimum batch size for model execution, defaults to 0.
- min_batch_timeout (int = 0): time to wait for minimum batch size, defaults to 0.
- tag (str = ""): additional tag for model information, defaults to “”.
- inputs (t.Optional[t.List[str]] = None): model inputs (TF only), defaults to None.
- outputs (t.Optional[t.List[str]] = None): model outputs (TF only), defaults to None.

These arguments provide details to add and configure
ML models within the model simulation.

Example: Loading an In-Memory ML Model to the Model
---------------------------------------------------
This example demonstrates how to instruct SmartSim to load
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

Assuming an initialized ``Model`` named `smartsim_model` exists, we add a TensorFlow model using
the ``Model.add_ml_model()`` function:

.. code-block:: python

    smartsim_model.add_ml_model(name="cnn", backend="TF", model=model, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)

In the above ``smartsim_model.add_ml_model()`` code snippet, we offer the following arguments:

-  name ("cnn"): A key to uniquely identify the model within the database.
-  backend ("TF): Indicating that the model is a TensorFlow model.
-  model (model): The in-memory representation of the TensorFlow model.
-  device ("GPU"): Specifying the device for ML model execution.
-  devices_per_node (2): Use two GPUs per node.
-  first_device (0): Start with 0 index GPU.
-  inputs (inputs): The names of the model inputs.
-  outputs (outputs): The names of the model outputs.

.. warning::
    Calling `exp.start(model)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the ML model to a non-existent database.

--------------------
Tensorflow functions
--------------------
Users can instruct SmartSim to upload TorchScript functions to the database
at runtime. Script functions are loaded into
standard orchestrators prior to the execution of Models. If using a
colocated orchestrator, use the ``Model.add_script()`` function.
Users have the flexibility to choose
between `"GPU"` or `"CPU"` for device selection, and in environments with multiple
devices, specific device numbers can be specified via `devices_per_node`.

When specifying a TF function using ``Model.add_function()``, the
following arguments are offered:

- name (str): key to store function under.
- function (t.Optional[str] = None): TorchScript function code.
- device (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- devices_per_node (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- first_device (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

Example: Loading a TensorFlow Function to the Model
---------------------------------------------------
This example demonstrates how to instruct SmartSim to load
a TensorFlow function into the database at model runtime.

.. note::
    - Use `Model.add_script` with a colocated deployment
    - Use `Model.add_function` on standard model deployment

**Python Script: Define a TF Function for Model Purposes**
To load a TF function, define the function within the Python driver script.

.. code-block:: python

    def timestwo(x):
        return 2*x

**SmartSim Model Integration:**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add a TensorFlow model using
the ``Model.add_function()`` function:

.. code-block:: python

    smartsim_model.add_function(name="example_func", function=timestwo, device="GPU", devices_per_node=2, first_device=0)

In the above ``smartsim_model.add_function()`` code snippet, we offer the following arguments:

-  name ("example_func"): A key to uniquely identify the model within the database.
-  function (timestwo): Name of the TorchScript function defined in the Python driver script.
-  device ("CPU"): Specifying the device for ML model execution.
-  devices_per_node (2): Use two GPUs per node.
-  first_device (0): Start with 0 index GPU.

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

- name (str): key to store script under.
- script (t.Optional[str] = None): TorchScript code (only supported for non-colocated orchestrators).
- script_path (t.Optional[str] = None): path to TorchScript code.
- device (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- devices_per_node (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- first_device (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

You might use TorchScript scripts to represent individual models within the model.

Example: Loading a TorchScript to the Model
--------------------------------------------
In this example, we demonstrate how to instruct SmartSim to load
an TorchScript into the database at model runtime.

**Python Script: Using Torch Scripts As ML Models**

Define the TorchScript code as a variable in the Python driver script:

.. code-block:: python

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

**SmartSim Model Integration:**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add a TensorFlow model using
the ``Model.add_script()`` function:

.. code-block:: python

    smartsim_model.add_script(name="example_script", script=torch_script_str, device="GPU", devices_per_node=2, first_device=0)

In the above ``smartsim_model.add_script()`` code snippet, we offer the following arguments:

-  name ("example_script"): key to store script under.
-  script (torch_script_str): TorchScript code.
-  device ("CPU"): device for script execution.
-  devices_per_node (2): Use two GPUs per node.
-  first_device (0): Start with 0 index GPU.

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