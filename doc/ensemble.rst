********
Ensemble
********

========
Overview
========
An ``Ensemble`` is a SmartSim object comprised of multiple
:ref:`Model<model_api>` objects, each representing an individual
simulation. An Ensemble can be be reference as a single instance and
enables users to launch a group of simulations
at runtime. Ensembles can be launched with other SmartSim
entities and ML infrastructure to build AI-enabled workflows.
The :ref:`Ensemble API<ensemble_api>` offers key Ensemble features,
including helper functions to:

- :ref:`Ensemble.add_ml_model()<load_ml_model>`: load a TF, TF-lite, PT, or ONNX model into the database at runtime.
- :ref:`Ensemble.add_function()<load_tf_function>`: load TorchScript functions to launch with each entity belonging to the ensemble.
- :ref:`Ensemble.add_script()<load_script>`: load TorchScripts to launch with each entity at runtime.
- :ref:`Ensemble.enable_key_prefixing()<prefix_ensemble>`: prevent key overlapping and enable ensemble members to run the same code.
- :ref:`Ensemble.attach_generator_files()<attach_file>`: attach configuration files to assign ensemble member parameters or load parameters to file.

An Ensemble is created through the ``Experiment.create_ensemble()`` function.
There are **three** Ensemble configuration that users can adjust to customize
Ensemble behavior:

1. :ref:`Manually<append_init>`: Create each Model individually and subsequently append the simulation to the ensemble.
2. :ref:`Replica creation<replicas_init>`: Generate a specified number of copies or instances of a simulation.
3. :ref:`Parameter expansion<param_expansion_init>`: Explore a range of simulation variable assignments by
   specifying values of input parameters and selecting an expansion strategy to rearrange
   across multiple Models within the ensemble.

==============
Initialization
==============
--------
Overview
--------
The creation of an Ensemble involves the use of various initializer arguments,
each serving a specific role in defining the characteristics and behavior of the
ensemble. These arguments facilitate three distinct methods of ensemble
creation: **manual model appending**, **parameter expansion**, and the utilization of **replicas**.

The key initializer arguments are:

-  `name` (Type: str): Specify the name of the ensemble, aiding in its unique identification.
-  `params` (Type: dict[str, Any]): Provides a dictionary of parameters:values for expanding into the ``Model`` members within the ensemble. Enables parameter expansion for diverse scenario exploration.
-  `params_as_args` (Type: list[str]): Specify which parameters from the `params` dictionary should be treated as command line arguments when executing the Models.
-  `batch_settings` (Type: BatchSettings, optional): Describes settings for batch workload treatment.
-  `run_settings` (Type: RunSettings, optional): Describes execution settings for individual Model members.
-  `replicas` (Type: int, optional): Declare the number of ``Model`` clones within the ensemble, crucial for the creation of simulation replicas.
-  `perm_strategy` (Type: str): Specifies a strategy for parameter expansion into ``Model`` instances, influencing the method of ensemble creation and number of ensemble members.
    The options are `"all_perm"`, `"step"`, and `"random"`.

-------------------
Parameter Expansion
-------------------
.. _param_expansion_init:
In ensemble simulations, parameter expansion is a technique that
allows users to set parameter values using the `params` initializer.
User's may control how the parameter values spread across the ensemble
members by using the `perm_strategy` initializer.

**Parameter Expansion Strategies:**

-  `"all_perm"`: Generates all possible parameter permutations for an exhaustive exploration.
-  `"step"`: Creates sets for each element in n arrays, providing a systematic exploration.
-  `"random"`: Allows random selection from predefined parameter spaces, offering a stochastic approach.

Examples
--------
Example 1 : Parameter Expansion with ``RunSettings`` and `params`
    .. note::
        The Ensemble requires an allocation.
    Expand same run settings and parameters to Ensemble members.
    Specify the parameter expansion strategy via `perm_strategy`.

    Begin by initializing a ``RunSettings`` object to expand to
    all Models:

    .. code-block:: python

        rs = exp.create_run_settings(exe="python", exe_args="output_my_parameter.py")

    Next, define the parameters to expand to all Models and with values to expand
    via a expansion strategy:

    .. code-block:: python

        params = {
            "name": ["Ellie", "John"],
            "parameter": [2, 11]
        }

    Finally, initialize an ``Ensemble`` by passing in the ``RunSettings``, `params` and `perm_strategy`:

    .. code-block:: python

        ensemble = exp.create_ensemble("ensemble", params=params, run_settings=rs, perm_strategy="all_perm")

    Notice that `perm_strategy="all_perm"` which means all permutations of the `params` key values will
    be calculated and distributed across Ensemble members. Here there are four permutations. Therefore,
    the Ensemble will have four ``Model`` members.

Example 2 : Parameter Expansion with ``RunSettings``, ``BatchSettings`` and `params`
    Submit the Ensemble as a batch job.
    Expand identical run settings and parameters to Ensemble members.
    Declare the parameter expansion strategy via `perm_strategy`.

    Begin by initializing and configuring a ``BatchSettings`` object to
    run the Ensemble instance:

    .. code-block:: python

        batch_args = {
            "distribution": "block"
            "exclusive": None
        }
        bs = exp.create_batch_settings(nodes=2,
                               time="10:00:00",
                               batch_args=batch_args)
    The above ``BatchSettings`` object will tell SmartSim to run the Ensemble on two
    nodes with a timeout of 10 hours.

    Next initialize a ``RunSettings`` object to expand to
    all Models:

    .. code-block:: python

        rs = exp.create_run_settings(exe="python", exe_args="output_my_parameter.py")
        rs.set_nodes(1)

    Next, define the parameters to include in all Models and with values to expand
    via a expansion strategy:

    .. code-block:: python

        params = {
            "name": ["Ellie", "John"],
            "parameter": [2, 11]
        }

    Finally, initialize an ``Ensemble`` by passing in the ``RunSettings``, `params` and `perm_strategy`:

    .. code-block:: python

        ensemble = exp.create_ensemble("ensemble", params=params, run_settings=rs, batch_settings=bs, perm_strategy="step")

    Notice that `perm_strategy="step"` which means values of the `params` key will be
    grouped into intervals and distributed across Ensemble members. Here there are two groups. Therefore,
    the Ensemble will have two ``Model`` members.

--------
Replicas
--------
.. _replicas_init:
In ensemble simulations, a replica strategy involves the creation of
identical or closely related models within an ensemble, allowing for the
assessment of how a system responds to the same set of parameters under
multiple instances. Users may use the `replicas` initializer argument
to create a specified number of identical Model members.

Examples
--------

Example 1 : Replica Creation with ``RunSettings`` and `replicas`
    To create an Ensemble of identical Models, begin by initializing a ``RunSettings``
    object:

    .. code-block:: python

        rs = exp.create_run_settings(exe="python", exe_args="output_my_parameter.py")

    Initialize the Ensemble by specifying the ``RunSettings`` object and number of clones to `replicas`:

    .. code-block:: python

        ensemble = exp.create_ensemble("ensemble-replica",
                               replicas=4,
                               run_settings=rs)

    By passing in `replicas=4`, four identical Ensemble members will be initialized.

Example 2 : Replica Creation with ``RunSettings``, ``BatchSettings`` and `replicas`
    To launch the Ensemble of identical Models as a batch job, begin by initializing a ``BatchSettings``
    object:

    .. code-block:: python

        batch_args = {
            "distribution": "block"
            "exclusive": None
        }
        bs = exp.create_batch_settings(nodes=4,
                               time="10:00:00",
                               batch_args=batch_args)
    The above ``BatchSettings`` object will tell SmartSim to run the Ensemble on four
    nodes with a timeout of 10 hours.

    Next, create a ``RunSettings`` object to expand to all Model replicas:

    .. code-block:: python

        rs = exp.create_run_settings(exe="python", exe_args="output_my_parameter.py")
        rs.set_nodes(4)

    Initialize the Ensemble by specifying the ``RunSettings`` object and number of clones to `replicas`:

    .. code-block:: python

        ensemble = exp.create_ensemble("ensemble-replica",
                               replicas=4,
                               run_settings=rs)

    By passing in `replicas=4`, four identical Ensemble members will be initialized.


---------------
Manually Append
---------------
.. _append_init:
Manually appending models involves the addition of user created model instances to an ensemble,
offering a level of customization in ensemble design. This approach is favorable when users
have distinct requirements for individual models, such as variations in parameters, run settings,
or model architectures.

Examples
--------
Example 1 : Append Models with ``BatchSettings``
    To create an empty Ensemble to append Models, initialize the Ensemble with
    a batch settings object:

    .. code-block:: python

        bs = exp.create_batch_settings(nodes=10,
                               time="01:00:00")
        ensemble = exp.create_ensemble("ensemble-append", batch_settings=bs)

    Next, create the Models to append to the Ensemble:

    .. code-block:: python

        srun_settings_1 = exp.create_run_settings(exe=exe, exe_args="path/to/script_1")
        srun_settings_2 = exp.create_run_settings(exe=exe, exe_args="path/to/script_2")
        model_1 = exp.create_model(name="model_1", run_settings=srun_settings_1)
        model_2 = exp.create_model(name="model_2", run_settings=srun_settings_2)

    Finally, append the ``Model`` object to the ``Ensemble``:

    .. code-block:: python

        ensemble.add_model(model_1)
        ensemble.add_model(model_2)
=====
Files
=====
.. _attach_files:
--------
Overview
--------
SmartSim enables users to attach files to Ensembles for use within the workflow
through the ``Ensemble.attach_generator_file()`` function. The function
accepts three input arguments:

1. `to_copy` (list, optional): files to copy, defaults to [].
2. `to_symlink` (list, optional): files to symlink, defaults to [].
3. `to_configure` (list, optional): input files with tagged parameters, defaults to [].

The `to_configure` argument accepts a list of files containing parameters and values
to use within each member of the Ensemble simulation. The distribution of the parameters
can be configured via the `perm_strategy` argument.

The `to_copy` and `to_symlink` arguments accept a list of files to write `params` to.

-------
Example
-------
This example provides a demonstration of using ``Ensemble.attach_generator_files()``
`to_configure` argument to load a text file to an Ensemble to set parameters used
required for each simulation for Ensemble members.

We want to load a text file named `configure_inputs.txt` to the workflow ensemble.
Within the file, is the parameter, `THERMO`, that is assigned multiple values:

.. code-block:: txt

   THERMO = ;[10,11,12];

The parameter `THERMO` is used within the application script that we will expand to
all Ensemble members. We would like to instruct SmartSim use all permutations of the
argument we passed in.

To encapsulate our application using an Ensemble, we must create an Experiment instance
to access Experiment helper function that create the Ensemble.
Begin by importing the Experiment module and SmartSim log module
to initialize the Experiment, `exp`:

.. code-block:: python

    from smartsim import Experiment
    from smartsim.log import get_logger

    logger = get_logger("Experiment Log")
    # Initialize the Experiment
    exp = Experiment("getting-started", launcher="auto")

We are applying the same run settings to all Ensemble members, and therefore,
will specify a run settings object when initializing the Ensemble.
We create the run settings object with the path to the application script
and the executable to run the script:

.. code-block:: python

    # Initialize a RunSettings object
    ensemble_settings = exp.create_run_settings(exe="python", exe_args="/path/to/application.py")

Next, initialize a ``Ensemble`` object with ``Experiment.create_ensemble()``
and pass in the `model_settings` instance:

.. code-block:: python

    # Initialize a Model object
    model = exp.create_ensemble("ensemble", run_settings=ensemble_settings, params={"THERMO"}, perm_strategy="all_perm")

We now have a ``Ensemble`` instance named `ensemble`. We specify that the simulations use the
variable `"THERMO"` and we specify that we would like the Ensemble to expand with all
possible permutations of the variable values.

Now attach the above text file to the Ensemble for use at entity runtime. We use the
``Ensemble.attach_generator_files()`` function and specify the `to_configure`
argument with the path to the text file, `configure_params.txt`:

.. code-block:: python

    # Attach the file to the Model instance
    model.attach_generator_files(to_configure="path/to/configure_params.txt")

When we launch the Ensemble using ``Experiment.start()``, SmartSim reads the values within the text
file, registered the value of `perm_strategy` which we set to `all_perm`, and will create three
different Models within the Ensemble.

=====================
ML Models and Scripts
=====================
--------
Overview
--------
SmartSim supports sending TorchScript functions, scripts, and
TF, TF-lite, PT, or ONNX models to the database at runtime
prior to the execution of ensemble members for use within the workload.
The ``Ensemble API`` provides a subset of helper functions that support
these capabilities:

* ``Ensemble.add_ml_model()`` : Load a TF, TF-lite, PT, or ONNX model into the DB at runtime.
* ``Ensemble.add_function()`` : Launch a TorchScript function with each ensemble member.
* ``Ensemble.add_script()`` : Launch a TorchScript with each ensemble member.

---------
AI Models
---------
.. _load_ml_model:
When configuring an ensemble, users can instruct SmartSim to load
TensorFlow (TF), TensorFlow Lite (TF-lite), PyTorch (PT), or ONNX
models dynamically to the database (colocated or standard). Machine Learning (ML) models added
are loaded prior to the execution of ensemble members and therefore
ready for use when a ensemble member is invoked. SmartSim users may
providing the model in memory or specifying its file path via the
``Ensemble.add_ml_model()`` Ensemble API helper function.

When specifying an ML model using ``Ensemble.add_ml_model()``, the
following arguments are offered:

-  `name` (str): Key used to store the model within the ensemble.
-  `model` (str | bytes | None): Model name in memory.
-  `model_path` (str): File path to the serialized model.
-  `backend` (str): Name of the model backend (TORCH, TF, TF-LITE, ONNX).
-  `device` (str, optional): Name of the device for execution (defaults to "CPU").
-  `batch_size` (int, optional): Batch size for execution (defaults to 0).
-  `min_batch_size` (int, optional): Minimum batch size for model execution (defaults to 0).
-  `tag` (str, optional): Additional tag for model information (defaults to an empty string).
-  `inputs` (list[str], optional): Names of model inputs (TF only, defaults to None).
-  `outputs` (list[str], optional): Names of model outputs (TF only, defaults to None).

These arguments provide details to add and configure
ML models within the ensemble simulation.

Example: Loading an In-Memory ML Model to the Ensemble
------------------------------------------------------
In this example, we demonstrate how to instruct SmartSim to load
an in-memory ML model into the database at ensemble runtime. It's
important to note that in-memory ML models are supported for
non-colocated deployments, making this example suitable for
standard orchestrators.

**Python Script: Creating a Keras CNN for Ensemble Purposes**

To create an in-memory ML model, define the Model within the Python driver script.
For the purpose of the example, we define a Keras CNN within the experiment.

.. code-block:: python

    def create_tf_cnn():
        """Create a Keras CNN for ensemble purposes

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

**SmartSim Ensemble Integration:**

Assuming an initialized ``Ensemble`` named `smartsim_ensemble`, we specify the
following parameters to the ``Ensemble.add_ml_model()`` function:

.. code-block:: python

    smartsim_ensemble.add_ml_model(name="cnn", backend="TF", model=model, device="CPU", inputs=inputs, outputs=outputs)

In this integration, we provide the following details:

-  `name`: "cnn" - A key to uniquely identify the model within the database.
-  `backend`: "TF" - Indicating that the model is a TensorFlow model.
-  `model`: model - The in-memory representation of the TensorFlow model.
-  `device`: "CPU" - Specifying the device for ML model execution.
-  `inputs`: inputs - The names of the model inputs.
-  `outputs`: outputs - The names of the model outputs.

When the ensemble is started via ``Experiment.start()``, the ML model will be loaded to the
standard orchestrator that is launched prior to the start of the ensemble.

---------------------
TorchScript functions
---------------------
.. _load_tf_function:
Users can instruct SmartSim to upload TorchScript functions to the database
at runtime. Script functions are loaded into
standard orchestrators prior to the execution of ensemble entities. If using a
colocated orchestrator, use the ``Ensemble.add_script()`` function.
Users have the flexibility to choose
between `"GPU"` or `"CPU"` for device selection, and in environments with multiple
devices, specific device numbers can be specified via `devices_per_node`.

When specifying a TF function using ``Ensemble.add_function()``, the
following arguments are offered:

-  `name`  (str) : key to store function under
-  `function` (str, optional) : Name of the TorchScript function
-  `device`  (str, optional) : device for script execution, defaults to “CPU”
-  `devices_per_node` (int) : assign the number of CPU's or GPU's to use on the node

Example: Loading an TensorFlow Function to the Ensemble
-------------------------------------------------------
In this example, we demonstrate how to instruct SmartSim to load
an TensorFlow function into the database at ensemble runtime. It's
important to note the function, ``Ensemble.add_function()`` is supported
for non-colocated deployments and during a colocated deployment, ``Ensemble.add_script()``
should be used.

**Python Script: Define a TF Function for Ensemble Purposes**
To load a TF function, define the function within the Python driver script.

.. code-block:: python

    def timestwo(x):
        return 2*x

**SmartSim Ensemble Integration:**

Assuming an initialized ``Ensemble`` named `smartsim_ensemble`, we specify the
following parameters to the ``Ensemble.add_function()`` function:

.. code-block:: python

    smartsim_ensemble.add_function(name="example_func", function=timestwo, device="CPU")

In this integration, we provide the following details:

-  `name`: "example_func" - A key to uniquely identify the model within the database.
-  `function`: timestwo - Name of the TorchScript function defined in the Python driver script.
-  `device`: "CPU" - Specifying the device for ML model execution.

When the ensemble is started via ``Experiment.start()``, the TF function will be loaded to the
standard orchestrator that is launched prior to the start of the ensemble.

-------------------
TorchScript Scripts
-------------------
.. _load_script:
SmartSim supports the execution of TorchScripts
with each entity belonging to the ensemble. Regardless of whether
the orchestrator is colocated or not, each script added to the ensemble is
loaded into the orchestrator prior to the execution of any ensemble member.
The flexibility of device selection further enhances
adaptability, offering the choice between "GPU" or "CPU." Users have the flexibility to choose
between `"GPU"` or `"CPU"` for device selection, and in environments with multiple
devices, specific device numbers can be specified via `devices_per_node` such as `"GPU:1,"`.

When specifying a TorchScript using ``Ensemble.add_script()``, the
following arguments are offered:

-  `name`  (str) : key to store script under
-  `script` (str, optional) : TorchScript code
-  `script_path` (str, optional) : file path to TorchScript code
-  `device`  (str, optional) : device for script execution, defaults to “CPU”
-  `devices_per_node` (int) : assign the number of CPU's or GPU's to use on the node

You might use TorchScript scripts to represent individual models within the ensemble:

Example: Loading an TorchScript to the Ensemble
-----------------------------------------------
In this example, we demonstrate how to instruct SmartSim to load
an TorchScript into the database at ensemble runtime.

**Python Script: Define a TorchScript for Ensemble Purposes**

Define the TorchScript code to a variable in the Python driver script:

.. code-block:: python

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

**SmartSim Ensemble Integration:**

Assuming an initialized ``Ensemble`` named `smartsim_ensemble`, we specify the
following parameters to the ``Ensemble.add_script()`` function:

.. code-block:: python

    smartsim_ensemble.add_script(name="example_script", script=torch_script_str, device="CPU")

In this integration, we provide the following details:

-  `name`: "example_script" - key to store script under
-  `script`: torch_script_str - TorchScript code
-  `device`: "CPU" - device for script execution

When the ensemble is started via ``Experiment.start()``, the TorchScript will be loaded to the
orchestrator that is launched prior to the start of the ensemble.

=========================
Data Collision Prevention
=========================
.. _prefix_ensemble:
--------
Overview
--------
When multiple ensemble members use the same code to access their respective models
in the Orchestrator, key overlapping can occur, leading to inadvertent data access
between ensemble members. To address this, the SmartSim Ensembles supports key prefixing
via the ``Ensemble.enable_key_prefixing()`` function,
which automatically adds the model name as a prefix to the keys used for access.
Enabling key prefixing eliminates issues related to key overlapping, allowing ensemble
members to use the same code without issue.

-------------------------------
Example: Ensemble Key Prefixing
-------------------------------
In this example, we explore ensemble key prefixing in SmartSim.
We create an ensemble of comprised of two Models that use identical code
and input tensors to a standard orchestrator. To prevent key collisions and ensure data
integrity, we enable key prefixing in the ensemble which automatically
appends the model name as a prefix to access keys. We then introduce
a consumer model within the Python driver script to demonstrate the effectiveness
of key prefixing in preventing conflicts during key requests
from the orchestrator.

The Application Producer Script
-------------------------------
In the Python driver script, we instruct SmartSim to create an Ensemble comprised of
two Models that execute the same `Application Producer Script`.
In the `Application Producer Script`, a SmartRedis client places a
tensor into the database. Since both Models use this script, two of the same
tensors with same tensor names will be placed into the database causing a key collision.
To prevent this, we enable key prefixing on the Ensemble in the driver script.
This means that when a Model places a tensor into the database, it will append
its name to the tensor key, such as `"model_1.tensor"`.

Below is the simulation code for each producer Model within the Ensemble:

.. code-block:: python

    from smartredis import Client, log_data
    from smartredis import *
    import numpy as np

    # Initialize a Client
    client = Client(cluster=False)

    # Create NumPy array
    array = np.array([1, 2, 3, 4])
    # Use SmartRedis client to place tensor in single sharded db
    client.put_tensor("tensor", array)

Continue to `The Application Consumer Script` utilize SmartSims key prefixing
features to retrieve each of the same named tensors.

The Application Consumer Script
===============================
In the Python driver script, we initialize a consumer ``Model`` that requests
the tensors produced from the producer script. To do so, we use SmartRedis
key prefixing functionality to instruct the SmartRedis client to append
the name of a model to the key being searched.

First specify the imports and initialize a SmartRedis Client:

.. code-block:: python

    from smartredis import Client, log_data
    from smartredis import *

    # Initialize a Client
    client = Client(cluster=False)

.. note::
    We launch a single-sharded database in the Experiment driver script
    and therefore do not need to use the ``ConfigOptions`` object here
    to connect the client to the database.

Retrieve the tensor from the first producer Model in the Ensemble. Use the
``Client.set_data_source()`` function to append the first Model name, `producer_0`, to the
key being searched. When ``Client.poll_tensor()`` is executed,
the client will poll for key, `producer_0.tensor`:

.. code-block:: python

    client.set_data_source("producer_0")
    val1 = client.poll_tensor("tensor", 100, 100)

Follow the same instructions above, however, change the prefix name to the name
of the second producer Model (`producer_1`):

.. code-block:: python

    client.set_data_source("producer_1")
    val2 = client.poll_tensor("tensor", 100, 100)

We print the boolean return to verify that the tensors were found:

.. code-block:: python

    client.log_data(LLInfo, f"producer_0.tensor was found: {val1}")
    client.log_data(LLInfo, f"producer_1.tensor was found: {val2}")

When the Experiment script is executed, the following output will appear in `consumer.out`::
    Default@11-46-05:producer_0.tensor was found: True
    Default@11-46-05:producer_1.tensor was found: True

The Experiment Script
=====================
To setup for the example in the Python driver script, we

-  initialize the Experiment `exp`
-  initialize the standard orchestrator `single_shard_db`
-  launch the `single_shard_db` using `exp.start()`

.. code-block:: python

    import numpy as np
    from smartredis import Client
    from smartsim import Experiment
    from smartsim.log import get_logger
    import sys

    exe_ex = sys.executable
    logger = get_logger("Experiment Log")
    # Initialize the Experiment
    exp = Experiment("getting-started", launcher="auto")

    # Initialize a single sharded database
    single_shard_db = exp.create_database(port=6379, db_nodes=1, interface="ib0")
    exp.generate(single_shard_db, overwrite=True)
    exp.start(single_shard_db)
    logger.info(exp.get_status(single_shard_db))

We are now setup to discuss key prefixing within the Experiment driver script.
Begin by initializing a ``RunSettings`` object expand to all ensemble members.
Specify the path to the application producer script discussed above.

.. code-block:: python

    # Initialize a RunSettings object
    ensemble_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/producer_script.py")

Next, initialize an ``Ensemble`` by specifying `ensemble_settings` and the number of clones to create:

.. code-block:: python

    producer_ensemble = exp.create_ensemble("producer", run_settings=ensemble_settings, replicas=2)

Enable ensemble key prefixing:

.. code-block:: python

    producer_ensemble.enable_key_prefixing()

Next, create the initialize the consumer Model that requests the tensors
produced by the ensemble:

.. code-block:: python

    # Initialize a RunSettings object
    model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/consumer_script.py")
    consumer_model = exp.create_model("consumer", model_settings)

Launch the ensemble:

.. code-block:: python

    exp.start(producer_ensemble, block=True, summary=True)

Set `block=True` so that ``Experiment.start()`` waits until the last Model has finished before continuing.

Register the Models that will be accessed within the consumer script:

.. code-block:: python

    for model in producer_ensemble:
        consumer_model.register_incoming_entity(model)

Launch the consumer Model:

.. code-block:: python

    exp.start(consumer_model, block=True, summary=True)

Tear down the database:

.. code-block:: python

    exp.stop(single_shard_db)
    logger.info(exp.summary())