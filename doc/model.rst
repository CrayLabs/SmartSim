*****
Model
*****
========
Overview
========
SmartSim ``Model`` objects enable users to execute computational tasks in an
``Experiment`` workflow, such as launching compiled applications,
running scripts, or performing general computational operations. ``Models`` can be launched with
other SmartSim entities and Orchestrators to build AI-enabled workflows.
``Model`` objects can leverage ML capabilities by utilizing the SmartSim client (:ref:`SmartRedis<dead_link>`)
to transfer data to a standard ``Orchestrator`` to enable other running ``Models`` to access the data.
Additionally, clients can execute ML models (TF, TF-lite, PyTorch, or ONNX) and TorchScripts stored in the
``Orchestrator``. SmartRedis is available in Python, C, C++, or Fortran.

To initialize a SmartSim ``Model``, use the ``Experiment.create_model()`` API function.
When creating a ``Model``, a :ref:`RunSettings<dead_link>` object must be provided. A ``RunSettings``
object specifies the ``Models`` executable simulation code (e.g. the full path to a compiled binary) as well as
application execution specifications. These specifications include :ref:`launch<dead_link>` commands (e.g. `srun`, `aprun`, `mpiexec`, etc),
compute resource requirements, and application command-line arguments.

A user can implement the use of an ``Orchestrator`` within a ``Model`` through **two** strategies:

- :ref:`Connect to an Orchestrator launched prior to the Model<std_model_doc>`
   When a ``Model`` is launched, it does not use or share compute
   resources on the same host (computer/server) where a SmartSim ``Orchestrator`` is running.
   Instead, it is launched on its own compute resources specified by the ``RunSettings`` object.
   The ``Model`` can connect via a SmartSim client to a launched standalone ``Orchestrator``.

- :ref:`Connect to an Orchestrator colocated with the Model<colo_model_doc>`
   When the colocated ``Model`` is started, SmartSim launches an ``Orchestrator`` on the ``Model`` compute
   nodes prior to the ``Models`` execution. The ``Model`` can then connect to the colocated ``Orchestrator``
   via a SmartSim client.

Once a ``Model`` instance has been initialized, users have access to
the :ref:`Model API<model_api>` functions to further configure the ``Model``.
The Models functions allow users to:

- :ref:`Attach files to a SmartSim Model for use within the simulation<files_doc>`
- :ref:`Launch an Orchestrator on the SmartSim Model compute nodes<colo_model_doc>`
- :ref:`Attach a ML model to the SmartSim Model instance<ai_model_doc>`
- :ref:`Attach a TorchScript function to the SmartSim Model instance<TS_doc>`
- :ref:`Register communication with another SmartSim Model instances<dead_link>`
- :ref:`Enable SmartSim Model key collision prevention<dead_link>`

SmartSim manages ``Model`` instances through the :ref:`Experiment API<experiment_api>` by providing functions to
launch, monitor, and stop applications. Additionally, ``Models`` can be launched individually
or as a group via an ``Ensemble``.

====================
Model Initialization
====================
Overview
========
The :ref:`Experiment API<experiment_api>` is responsible for initializing all workflow entities.
A ``Model`` is created using the ``Experiment.create_model()`` factory method, and users can customize the
``Model`` via the factory method parameters.

The key initializer arguments are:

-  `name` (str): Specify the name of the model for unique identification.
-  `run_settings` (base.RunSettings): Describe execution settings for a Model.
-  `params` (t.Optional[t.Dict[str, t.Any]] = None): Provides a dictionary of parameters for Models.
-  `path` (t.Optional[str] = None): Path to where the model should be executed at runtime.
-  `enable_key_prefixing` (bool = False): Prefix the model name to data sent to the database to prevent key collisions. Default is `True`.
-  `batch_settings` (t.Optional[base.BatchSettings] = None): Describes settings for batch workload treatment.

A `name` and :ref:`RunSettings<dead_link>` reference are required to initialize a ``Model``.
Optionally, include a :ref:`BatchSettings<dead_link>` object to specify workload manager batch launching.

.. note::
    ``BatchSettings`` attached to a model are ignored when the model is executed as part of an ensemble.

The `params` factory method parameter for ``Models`` lets users define simulation parameters and their
values through a dictionary. Using :ref:`Model API<model_api>` functions, users can write these parameters to
a file in the Model's working directory.

When a Model instance is passed to ``Experiment.generate()``, a
directory within the Experiment directory
is automatically created to store input and output files from the model.

.. note::
    Model instances will be executed in the current working directory by default if no `path` argument
    is supplied.

.. _std_model_doc:
Standard Model
==============
By default, a ``Model`` does not share compute resources with other ``Model`` entities or ``Orchestrator`` instances.
A ``Model`` connects to an ``Orchestrator`` via the SmartSim client (:ref:`SmartRedis<dead_link>`).
For the client connection to be successful, the SmartSim standalone ``Orchestrator`` must be launched
prior to the start of the ``Model``. To create a standard ``Model``, users initialize a
``Model`` instance with the ``Experiment.create_model()`` function.

.. note::
    A ``Model`` can be launched without an ``Orchestrator`` if data transfer and ML capabilities are not
    required.

In the proceeding :ref:`Instructions<std_model_init_instruct>` subsection, we provide an example illustrating the deployment of a standard model.

.. _std_model_init_instruct:
------------
Instructions
------------
This example provides a demonstration of how to initialize and launch a ``Model``
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

To created an isolated output directory for the ``Model``, invoke ``Experiment.generate()`` via the
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

.. _colo_model_doc:
A Colocated Model
=================
A colocated ``Model`` runs on the same compute node(s) as a SmartSim ``Orchestrator``.
With a colocated model, the Model and the Orchestrator share compute resources.
To create a colocated model,
users first initialize a ``Model`` instance with the ``Experiment.create_model()`` function.
A user must then colocate the ``Orchestrator`` and ``Model`` using the function ``Model.colocate_db()``.
This instructs SmartSim to launch an ``Orchestrator`` on the application compute
node(s) prior to the ``Models`` execution.

There are **three** different Model API functions to colocate a ``Model``:

- ``Model.colocate_db_tcp()``: Colocate an Orchestrator instance and establish client communication using TCP/IP.
- ``Model.colocate_db_uds()``: Colocate an Orchestrator instance and establish client communication using Unix domain sockets (UDS).
- ``Model.colocate_db()``: (deprecated) An alias for `Model.colocate_db_tcp()`.

Each function initializes an unsharded database accessible only to the model processes on the same compute node. When the model
is started, the ``Orchestrator`` will be launched on the same compute resource as the model. Only the colocated ``Model``
may communicate with the ``Orchestrator`` via a SmartRedis client by using the loopback TCP interface or
Unix Domain sockets. Extra parameters for the database can be passed into the functions above
via `kwargs`.

.. code-block:: python

    example_kwargs = {
        "maxclients": 100000,
        "threads_per_queue": 1,
        "inter_op_threads": 1,
        "intra_op_threads": 1
    }

For a walkthrough of how to colocate a Model, navigate to the :ref:`Colocated Orchestrator<dead_link>` for
instructions.

.. _files_doc:
=====
Files
=====
Overview
========
Applications often depend on external files (e.g. training datasets, evaluation datasets, etc)
to operate as intended. Users can instruct SmartSim to copy, symlink, or manipulate external files
prior to the ``Model`` launch via the ``Model.attach_generator_files()`` function.

.. note::
    Multiple calls to ``Model.attach_generator_files()`` will overwrite previous file configurations
    in the ``Model``.

To attach a file to a ``Model`` for use at runtime, provide one of the following arguments to the
``Model.attach_generator_files()`` function:

* `to_copy` (t.Optional[t.List[str]] = None): Files that are copied into the path of the entity.
* `to_symlink` (t.Optional[t.List[str]] = None): Files that are symlinked into the path of the entity.

To specify a template file in order to programmatically replace specified parameters during generation
of the ``Model`` directory, pass the following value to the ``Model.attach_generator_files()`` function:

* `to_configure` (t.Optional[t.List[str]] = None): Designed for text-based ``Model`` input files,
  "to_configure" is exclusive to the ``Model``. During ``Model`` directory generation, the attached
  files are parsed and specified tagged parameters are replaced with the `params` values that were
  specified in the ``Experiment.create_model()`` factory method of the ``Model``. The default tag is a semicolon
  (e.g., THERMO = ;THERMO;).

In the :ref:`Example<files_example_doc>` subsection, we provide an example using the value `to_configure`
within ``attach_generator_files()``.

.. _files_example_doc:
Example
=======
This example demonstrates how to attach a file to a ``Model`` for parameter replacement at time
of ``Model`` directory generation. This is accomplished using the `params` function parameter in
the ``Experiment.create_model()`` factory function and the `to_configure` function parameter
in ``Model.attach_generator_files()``.

In this example, we have a text file named `params_inputs.txt`. Within the text, is the parameter `THERMO`
that is required by the application at runtime:

.. code-block:: txt

   THERMO = ;THERMO;

In order to have the tagged parameter `;THERMO;` replaced with a usable value at runtime, two steps are required:

1. The `THERMO` variable must be included in ``Experiment.create_model()`` factory method as part of the `params` parameter.
2. The file containing the tagged parameter `;THERMO;`, `params_inputs.txt`, must be attached to the ``Model``
via the ``Model.attach_generator_files()`` method as part of the `to_configure` parameter.

To encapsulate our application within a ``Model``, we must create an ``Experiment`` instance
to gain access to the ``Experiment`` factory method that creates the ``Model``.
Begin by importing the ``Experiment`` module, importing SmartSim `log` module and initializing
an ``Experiment``:

.. code-block:: python

    from smartsim import Experiment
    from smartsim.log import get_logger

    logger = get_logger("Experiment Log")
    # Initialize the Experiment
    exp = Experiment("getting-started", launcher="auto")

Models require run settings. Create a simple ``RunSettings`` object to specify the path to
our application script as an executable argument and the executable to run the script:

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

To created an isolated directory for the ``Model`` outputs and configuration files, invoke ``Experiment.generate()`` via the
``Experiment`` instance `exp` with `model` as an input parameter:

.. code-block:: python

    model = exp.generate(model)

Launching the model with ``exp.start(example_model)`` processes attached generator files. `configure_inputs.txt` will be
available in the model working directory and SmartSim will assign `example_model` `params` to the text file.

The contents of `params_inputs.txt` after Model completion are:

.. code-block:: txt

   THERMO = 1

======================
Output and Error Files
======================
By default, SmartSim stores the standard output and error of the ``Model`` in two files:

* `<model_name>.out`
* `<model_name>.err`

The files are created in the working directory of the ``Model``, and the filenames directly match the
model's name. The `<model_name>.out` file logs standard outputs and the
`<model_name>.err` logs errors for debugging.

.. note::
    Invoking ``Experiment.generate(model)`` will create a directory `model_name/` and will store
    the two files within that directory. You can also specify a path for these files using the
    `path` parameter when invoking ``Experiment.create_model()``.

=====================
ML Models and Scripts
=====================
Overview
========
SmartSim users have the capability to utilize ML runtimes within a ``Model``.
Functions accessible through a ``Model`` object support the integration of ML
frameworks such as TensorFlow, TensorFlow-lite, PyTorch, and ONNX. Users can
load two types of data sources to ``Orchestrator``: ML models and TorchScripts.

Users can follow **two** processes to load a ML model to the ``Orchestrator``:

- :ref:`from memory<ai_model_doc>`
- :ref:`from file<ai_model_doc>`

Users can follow **three** processes to load a TorchScript to the ``Orchestrator``:

- :ref:`from memory<in_mem_TF_doc>`
- :ref:`from file<TS_from_file>`
- :ref:`from string<TS_raw_string>`

Once a ML model or TorchScript is loaded to the ``Orchestrator``, ``Model`` objects can
leverage ML capabilities by utilizing the SmartSim client (:ref:`SmartRedis<dead_link>`)
to execute the stored ML models or TorchScripts.

.. _ai_model_doc:
AI Models
=========
When configuring a ``Model``, users can instruct SmartSim to load
Machine Learning (ML) models dynamically to the ``Orchestrator`` (colocated or standard). ML models added
are loaded into the ``Orchestrator`` prior to the execution of the ``Model``. To load an ML model
to the database, SmartSim users can provide the ML model **in-memory** or specify the **file path**
when using the ``Model.add_ml_model()`` function. The supported ML frameworks are TensorFlow,
TensorFlow-lite, PyTorch, and ONNX.

When attaching an ML model using ``Model.add_ml_model()``, the
following arguments are offered to customize the storage and execution of the ML model:

- `name` (str): name to reference the model in the Orchestrator.
- `backend` (str): name of the backend (TORCH, TF, TFLITE, ONNX).
- `model` (t.Optional[str] = None): A model in memory (only supported for non-colocated orchestrators).
- `model_path` (t.Optional[str] = None): serialized model.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): name of device for execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `batch_size` (int = 0): batch size for execution, defaults to 0.
- `min_batch_size` (int = 0): minimum batch size for model execution, defaults to 0.
- `min_batch_timeout` (int = 0): time to wait for minimum batch size, defaults to 0.
- `tag` (str = ""): additional tag for model information, defaults to “”.
- `inputs` (t.Optional[t.List[str]] = None): model inputs (TF only), defaults to None.
- `outputs` (t.Optional[t.List[str]] = None): model outputs (TF only), defaults to None.

Continue for demonstrations on how to :ref:`load an in-memory ML model<in_mem_ML_model_ex>` and
:ref:`load an ML model from file<from_file_ML_model_ex>`.

.. _in_mem_ML_model_ex:
-------------------------------------
Example: Attach an in-memory ML Model
-------------------------------------
This example demonstrates how to attach an in-memory ML model to a SmartSim ``Model``
to load into an ``Orchestrator`` at ``Model`` runtime.

.. note::
    This example assumes:

    - an ``Orchestrator`` is launched prior to the ``Models`` execution
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define an in-memory Keras CNN**

The ML model must be defined using one of the supported ML frameworks. For the purpose of the example,
we define a Keras CNN in the same script as the SmartSim ``Experiment``:

.. code-block:: python

    def create_tf_cnn():
        """Create an in-memory Keras CNN for example purposes

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

**Attach the ML model to a SmartSim Model**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add the in-memory TensorFlow model using
the ``Model.add_ml_model()`` function and specify the in-memory model to the parameter `model`:

.. code-block:: python

    smartsim_model.add_ml_model(name="cnn", backend="TF", model=model, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)

In the above ``smartsim_model.add_ml_model()`` code snippet, we offer the following arguments:

-  `name` ("cnn"): A name to reference the model in the Orchestrator.
-  `backend` ("TF): Indicating that the model is a TensorFlow model.
-  `model` (model): The in-memory representation of the TensorFlow model.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the ML model to a non-existent database.

When the ``Model`` is started via ``Experiment.start()``, the ML model will be loaded to the
launched ``Orchestrator``. The ML model can then be executed on the ``Orhcestrator`` via a SmartSim
client (:ref:`SmartRedis<dead_link>`) within the application code.

.. _from_file_ML_model_ex:
----------------------------------------
Example: Attaching an ML Model from file
----------------------------------------
This example demonstrates how to attach a ML model from file to a SmartSim ``Model``
to load into an ``Orchestrator`` at ``Model`` runtime.

.. note::
    This example assumes:

    - a standard ``Orchestrator`` is launched prior to the ``Models`` execution
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define a Keras CNN script**

The ML model must be defined using one of the supported ML frameworks. For the purpose of the example,
we define the function `save_tf_cnn()` that saves a Keras CNN to a file named `model.pb` located in our
Experiment path:

.. code-block:: python

    def save_tf_cnn(path, file_name):
        """Create a Keras CNN and save to file for example purposes"""
        from smartsim.ml.tf import freeze_model

        n = Net()
        input_shape = (3, 3, 1)
        n.build(input_shape=(None, *input_shape))
        inputs = Input(input_shape)
        outputs = n(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs, name=n.name)

        return freeze_model(model, path, file_name)

    # Get and save TF model
    model_file, inputs, outputs = save_tf_cnn(model_dir, "model.pb")

**Attach the ML model to a SmartSim Model**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add a TensorFlow model using
the ``Model.add_ml_model()`` function and specify the model path to the parameter `model_path`:

.. code-block:: python

    smartsim_model.add_ml_model(name="cnn", backend="TF", model_path=model_file, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)

In the above ``smartsim_model.add_ml_model()`` code snippet, we offer the following arguments:

-  `name` ("cnn"): A name to reference the model in the Orchestrator.
-  `backend` ("TF): Indicating that the model is a TensorFlow model.
-  `model_path` (model_file): The path to the ML model script.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the ML model to a non-existent database.

When the ``Model`` is started via ``Experiment.start()``, the ML model will be loaded to the
launched ``Orchestrator``. The ML model can then be executed on the ``Orhcestrator`` via a SmartSim
client (:ref:`SmartRedis<dead_link>`) within the application code.

.. _TS_doc:
TorchScripts
============
When configuring a ``Model``, users can instruct SmartSim to load TorchScripts dynamically
to the ``Orchestrator``. TorchScripts added are loaded into the ``Orchestrator`` prior to
the execution of the ``Model``. To load a TorchScript to the database, SmartSim users
can follow one of the processes:

- :ref:`Define a TorchScript function in-memory<in_mem_TF_doc>`
   Use the ``Model.add_function()`` to instruct SmartSim to load an in-memory TorchScript to the ``Orchestrator``.
- :ref:`Define a TorchScript function from file<TS_from_file>`
   Provide file path to ``Model.add_script()`` to instruct SmartSim to load the TorchScript from file to the ``Orchestrator``.
- :ref:`Define a TorchScript function as string<TS_raw_string>`
   Provide function string to ``Model.add_script()`` to instruct SmartSim to load a raw string as a TorchScript function to the ``Orchestrator``.

Continue or select the respective process link to learn more on how each function (``Model.add_script()`` and ``Model.add_function()``)
dynamically loads TorchScripts to the ``Orchestrator``.

.. _in_mem_TF_doc:
-------------------------------
Attach an in-memory TorchScript
-------------------------------
Users can define TorchScript functions within the Python driver script
to attach to a ``Model``. This feature is supported by ``Model.add_function()`` which provides flexible
device selection, allowing users to choose between which device the the TorchScript is executed on, `"GPU"` or `"CPU"`.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

.. warning::
    ``Model.add_function()`` does **not** support loading in-memory TorchScript functions to a colocated ``Orchestrator``.
    If you would like to load a TorchScript function to a colocated ``Orchestrator``, define the function
    as a :ref:`raw string<TS_raw_string>` or :ref:`load from file<TS_from_file>`.

When specifying an in-memory TF function using ``Model.add_function()``, the
following arguments are offered:

- `name` (str): reference name for the script inside of the ``Orchestrator``.
- `function` (t.Optional[str] = None): TorchScript function code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

.. _in_mem_TF_ex:
Example: Loading a in-memory TorchScript function
-------------------------------------------------
This example walks through the steps of instructing SmartSim to load an in-memory TorchScript function
to a standard ``Orchestrator``.

.. note::
    The example assumes:

    - a standard ``Orchestrator`` is launched prior to the ``Models`` execution
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define an in-memory TF function**

To begin, define an in-memory TorchScript function within the Python driver script.
For the purpose of the example, we add a simple TorchScript function, `timestwo`:

.. code-block:: python

    def timestwo(x):
        return 2*x

**Attach the in-memory TorchScript function to a SmartSim Model**

We use the ``Model.add_function()`` function to instruct SmartSim to load the TorchScript function `timestwo`
onto the launched standard ``Orchestrator``. Specify the function `timestwo` to the `function`
parameter:

.. code-block:: python

    smartsim_model.add_function(name="example_func", function=timestwo, device="GPU", devices_per_node=2, first_device=0)

In the above ``smartsim_model.add_function()`` code snippet, we offer the following arguments:

-  `name` ("example_func"): A name to uniquely identify the model within the database.
-  `function` (timestwo): Name of the TorchScript function defined in the Python driver script.
-  `device` ("CPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the ML model to a non-existent database.

When the ``Model`` is started via ``Experiment.start()``, the TF function will be loaded to the
standard ``Orchestrator``. The function can then be executed on the ``Orhcestrator`` via a SmartSim
client (:ref:`SmartRedis<dead_link>`) within the application code.

.. _TS_from_file:
------------------------------
Attach a TorchScript from file
------------------------------
Users can attach TorchScript functions from a file to a ``Model`` and upload them to a
colocated or standard ``Orchestrator``. This functionality is supported by the ``Model.add_script()``
function's `script_path` parameter. The function supports
flexible device selection, allowing users to choose between `"GPU"` or `"CPU"` via the `device` parameter.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

When specifying a TorchScript using ``Model.add_script()``, the
following arguments are offered:

- `name` (str): Reference name for the script inside of the ``Orchestrator``.
- `script` (t.Optional[str] = None): TorchScript code (only supported for non-colocated orchestrators).
- `script_path` (t.Optional[str] = None): path to TorchScript code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

.. _TS_from_file_ex:
Example: Loading a TorchScript from File
----------------------------------------
This example walks through the steps of instructing SmartSim to load a TorchScript from file
to a ``Orchestrator``.

.. note::
    This example assumes:

    - a ``Orchestrator`` is launched prior to the ``Models`` execution
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define a TorchScript script**

For the example, we create the Python script `torchscript.py`. The file contains a
simple torch function shown below:

.. code-block:: python

    def negate(x):
        return torch.neg(x)

**Attach the TorchScript script to a SmartSim Model**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add a TorchScript script using
the ``Model.add_script()`` function and specify the script path to the parameter `script_path`:

.. code-block:: python

    smartsim_model.add_script(name="example_script", script_path="path/to/torchscript.py", device="GPU", devices_per_node=2, first_device=0)

In the above ``smartsim_model.add_script()`` code snippet, we offer the following arguments:

-  `name` ("example_script"): Reference name for the script inside of the ``Orchestrator``.
-  `script_path` ("path/to/torchscript.py"): Path to the script file.
-  `device` ("CPU"): device for script execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the ML model to a non-existent database.

When `smartsim_model` is started via ``Experiment.start()``, the TorchScript will be loaded from file to the
orchestrator that is launched prior to the start of the `smartsim_model`.

.. _TS_raw_string:
---------------------------------
Define TorchScripts as raw string
---------------------------------
Users can upload TorchScript functions from string to send to a colocated or
standard ``Orchestrator``. This feature is supported by the
``Model.add_script()`` function's `script` parameter. The function supports
flexible device selection, allowing users to choose between `"GPU"` or `"CPU"` via the `device` parameter.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

When specifying a TorchScript using ``Model.add_script()``, the
following arguments are offered:

- `name` (str): Reference name for the script inside of the ``Orchestrator``.
- `script` (t.Optional[str] = None): TorchScript code (only supported for non-colocated orchestrators).
- `script_path` (t.Optional[str] = None): path to TorchScript code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

.. _TS_from_file_ex:
Example: Loading a TorchScript from string
------------------------------------------
This example walks through the steps of instructing SmartSim to load a TorchScript function
from string to a ``Orchestrator`` before the execution of the associated ``Model``.

.. note::
    This example assumes:

    - a ``Orchestrator`` is launched prior to the ``Models`` execution
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define a string TorchScript**

Define the TorchScript code as a variable in the Python driver script:

.. code-block:: python

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

**Attach the TorchScript function to a SmartSim Model**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add a TensorFlow model using
the ``Model.add_script()`` function and specify the variable `torch_script_str` to the parameter
`script`:

.. code-block:: python

    smartsim_model.add_script(name="example_script", script=torch_script_str, device="GPU", devices_per_node=2, first_device=0)

In the above ``smartsim_model.add_script()`` code snippet, we offer the following arguments:

-  `name` ("example_script"): key to store script under.
-  `script` (torch_script_str): TorchScript code.
-  `device` ("CPU"): device for script execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the ML model to a non-existent database.

When the model is started via ``Experiment.start()``, the TorchScript will be loaded to the
orchestrator that is launched prior to the start of the model.

=========================
Data Collision Prevention
=========================

--------
Interact
--------
SmartSim's tensor prefixing simplifies data interaction by allowing users to easily manage
tensors in the same script that placed them and retrieve tensors placed by other scripts in
the orchestrator. The following subsections will explore these scenarios, providing guidance
for efficient tensor handling in a SmartSim workflow.

DataStructures
==============
------
Enable
------
.. tabs::

    .. group-tab:: Tensor
        Tensor prefixing in SmartSim provides an organized way to prevent naming conflicts and differentiate
        tensors produced by models or ensembles within a SmartSim simulation. It involves prepending the
        Model `name` to the tensor `name` when sending data to the orchestrator. The activation can
        be configured in two ways:

        - Activate tensor prefixing in the **driver script**
        - Activate tensor prefixing in the **application script**

        **Activate Tensor Prefixing in the Driver Script**

        Users can enable tensor prefixing on a SmartSim ``Model`` by utilizing the ``Model.enable_key_prefixing()``
        function. This functionality ensures that the ``Model`` `name` is prepended to the tensor `name` when
        sending data to the orchestrator.

        In the example below, we illustrate how to create a ``Model`` instance named `model` and activate tensor
        prefixing by setting it to `True` using the ``enable_key_prefixing()`` method.

        .. code-block:: python

            # Create the run settings for the model
            model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/application_script.py")

            # Create a Model instance named 'model'
            model = exp.create_model("model_name", model_settings)
            # Enable tensor prefixing
            model.enable_key_prefixing()

        In application script of `model`, two tensors named `tensor_1` and `tensor_2` are sent to the orchestrator.
        The contents of the orchestrator are as follows:

        .. code-block:: bash

            1) model_name.tensor_1
            2) model_name.tensor_2

        **Activate Tensor Prefixing in the Application Script**

        Users can enable tensor prefixing on a SmartRedis Client by utilizing the ``Client.use_tensor_ensemble_prefix()``
        function within the application script. This functionality ensures that the ``Model`` `name`
        is prepended to the tensor `name` when sending data to the orchestrator using
        ``Client.put_tensor()``, ``Client.rename_tensor()`` or ``Client.copy_tensor()``.

        .. warning::
            SmartSim users do not have access to ``Client.use_tensor_ensemble_prefix()`` unless prefixing is
            enabled on the ``Model`` via ``Model.enable_key_prefixing()``.

        In the following example, we demonstrate creating a ``Client`` instance named `client` and
        toggling tensor prefixing between `True` and `False` using the ``use_tensor_ensemble_prefix()`` method.

        .. code-block:: python

            # Initialize a Client
            client = Client(cluster=False)

            # Disable key prefixing
            client.use_tensor_ensemble_prefix(False)
            # Place a tensor in the orchestrator
            client.put_tensor("tensor_1", np.array([5, 6, 7, 8]))
            # Enable key prefixing
            client.use_tensor_ensemble_prefix(True)
            # Copy the tensor to a different tensor
            client.copy_tensor("tensor_1", "copied_tensor")

        In the above application script, the `client` transmits a tensor named `tensor_1`
        to the orchestrator. Prefixing is deactivated when sending `tensor_1` and reactivated when copying
        `tensor_1` to the new data source named `copied_tensor`. The orchestrator's contents are as follows:

        .. code-block:: bash

            1) tensor_1
            2) model_name.copied_tensor

    .. group-tab:: DataSet
        Tensor prefixing in SmartSim provides an organized way to prevent naming conflicts and differentiate
        tensors produced by models or ensembles within a SmartSim simulation. It involves prepending the
        Model `name` to the tensor `name` when sending data to the orchestrator. The activation can
        be configured in two ways:

        - Activate tensor prefixing in the **driver script**
        - Activate tensor prefixing in the **application script**

        **Activate Tensor Prefixing in the Driver Script**

        Users can enable tensor prefixing on a SmartSim ``Model`` by utilizing the ``Model.enable_key_prefixing()``
        function. This functionality ensures that the ``Model`` `name` is prepended to the tensor `name` when
        sending data to the orchestrator.

        In the example below, we illustrate how to create a ``Model`` instance named `model` and activate tensor
        prefixing by setting it to `True` using the ``enable_key_prefixing()`` method.

        .. code-block:: python

            # Create the run settings for the model
            model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/application_script.py")

            # Create a Model instance named 'model'
            model = exp.create_model("model_name", model_settings)
            # Enable tensor prefixing
            model.enable_key_prefixing()

        In application script of `model`, two tensors named `tensor_1` and `tensor_2` are sent to the orchestrator.
        The contents of the orchestrator are as follows:

        .. code-block:: bash

            1) model_name.tensor_1
            2) model_name.tensor_2

        **Activate Tensor Prefixing in the Application Script**

        Users can enable tensor prefixing on a SmartRedis Client by utilizing the ``Client.use_tensor_ensemble_prefix()``
        function within the application script. This functionality ensures that the ``Model`` `name`
        is prepended to the tensor `name` when sending data to the orchestrator using
        ``Client.put_tensor()``, ``Client.rename_tensor()`` or ``Client.copy_tensor()``.

        .. warning::
            SmartSim users do not have access to ``Client.use_tensor_ensemble_prefix()`` unless prefixing is
            enabled on the ``Model`` via ``Model.enable_key_prefixing()``.

        In the following example, we demonstrate creating a ``Client`` instance named `client` and
        toggling tensor prefixing between `True` and `False` using the ``use_tensor_ensemble_prefix()`` method.

        .. code-block:: python

            # Initialize a Client
            client = Client(cluster=False)

            # Disable key prefixing
            client.use_tensor_ensemble_prefix(False)
            # Place a tensor in the orchestrator
            client.put_tensor("tensor_1", np.array([5, 6, 7, 8]))
            # Enable key prefixing
            client.use_tensor_ensemble_prefix(True)
            # Copy the tensor to a different tensor
            client.copy_tensor("tensor_1", "copied_tensor")

        In the above application script, the `client` transmits a tensor named `tensor_1`
        to the orchestrator. Prefixing is deactivated when sending `tensor_1` and reactivated when copying
        `tensor_1` to the new data source named `copied_tensor`. The orchestrator's contents are as follows:

        .. code-block:: bash

            1) tensor_1
            2) model_name.copied_tensor

    .. group-tab:: Agg List

    .. group-tab:: ML Model

    .. group-tab:: Script

--------
Interact
--------
.. tabs::

    .. group-tab:: Tensor

        **driver script**
        **app script**
    .. group-tab:: DataSet

    .. group-tab:: Agg List

    .. group-tab:: ML Model

    .. group-tab:: Script