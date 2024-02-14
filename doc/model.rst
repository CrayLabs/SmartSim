.. _model_object_doc:
*****
Model
*****
========
Overview
========
SmartSim ``Model`` objects enable users to execute computational tasks in an
``Experiment`` workflow, such as launching compiled applications,
running scripts, or performing general computational operations. A ``Model`` can be launched with
other SmartSim ``Models`` and ``Orchestrators`` to build AI-enabled workflows.
With the SmartSim ``Client`` (:ref:`SmartRedis<smartredis-api>`), data can be transferred from a ``Model``
to the ``Orchestrator`` for use in an ML Model (TF, TF-lite, PyTorch, or ONNX), online
training process, or additional ``Model`` applications. SmartSim ``Clients`` (SmartRedis) are available in
Python, C, C++, or Fortran.

To initialize a SmartSim ``Model``, use the ``Experiment.create_model()`` API function.
When creating a ``Model``, a :ref:`RunSettings<run_settings_doc>` object must be provided. A ``RunSettings``
object specifies the ``Model`` executable (e.g. the full path to a compiled binary) as well as
executable arguments and launch parameters. These specifications include launch commands (e.g. `srun`, `aprun`, `mpiexec`, etc),
compute resource requirements, and application command-line arguments.

Once a ``Model`` instance has been initialized, users have access to
the :ref:`Model API<model_api>` functions to further configure the ``Model``.
The Model API functions provide users with the following capabilities:

- :ref:`Attach Files to a SmartSim Model<files_doc>`
- :ref:`Colocate an Orchestrator to a SmartSim Model<colo_model_doc>`
- :ref:`Attach a ML model to the SmartSim Model<ai_model_doc>`
- :ref:`Attach a TorchScript function to the SmartSim Model<TS_doc>`
- :ref:`Enable SmartSim Model data collision prevention<model_key_collision>`

Once the ``Model`` has been configured and launched, a user can leverage an ``Orchestrator`` within a ``Model``
through **two** strategies:

- :ref:`Connect to a standalone Orchestrator<standalone_orch_doc>`
   When a ``Model`` is launched, it does not use or share compute
   resources on the same host (computer/server) where a SmartSim ``Orchestrator`` is running.
   Instead, it is launched on its own compute resources specified by the ``RunSettings`` object.
   The ``Model`` can connect via a SmartRedis ``Client`` to a launched standalone ``Orchestrator``.

- :ref:`Connect to an colocated Orchestrator<colocated_orch_doc>`
   When the colocated ``Model`` is started, SmartSim launches an ``Orchestrator`` on the ``Model`` compute
   nodes prior to the ``Model`` execution. The ``Model`` can then connect to the colocated ``Orchestrator``
   via a SmartRedis ``Client``.

.. note::
    For the ``Client`` connection to be successful from within the ``Model`` application,
    the SmartSim ``Orchestrator`` must be launched prior to the start of the ``Model``.

.. note::
    A ``Model`` can be launched without an ``Orchestrator`` if data transfer and ML capabilities are not
    required.

SmartSim manages ``Model`` instances through the :ref:`Experiment API<experiment_api>` by providing functions to
launch, monitor, and stop applications. Additionally, a ``Model`` can be launched individually
or as a group via an :ref:`Ensemble<ensemble_doc>`.

==============
Initialization
==============
Overview
========
The :ref:`Experiment API<experiment_api>` is responsible for initializing all SmartSim entities.
A ``Model`` is created using the ``Experiment.create_model()`` factory method, and users can customize the
``Model`` via the factory method parameters.

The key initializer arguments of ``create_model()`` are:

-  `name` (str): Specify the name of the ``Model`` for unique identification.
-  `run_settings` (base.RunSettings): Describe execution settings for a ``Model``.
-  `params` (t.Optional[t.Dict[str, t.Any]] = None): Provides a dictionary of parameters for ``Model``.
-  `path` (t.Optional[str] = None): Path to where the ``Model`` should be executed at runtime.
-  `enable_key_prefixing` (bool = False): Prefix the ``Model`` name to data sent to the ``Orchestrator`` to prevent key collisions. Default is `False`.
-  `batch_settings` (t.Optional[base.BatchSettings] = None): Describes settings for batch workload treatment.

A `name` and :ref:`RunSettings<run_settings_doc>` reference are required to initialize a ``Model``.
Optionally, include a :ref:`BatchSettings<batch_settings_doc>` object to specify workload manager batch launching.

.. note::
    ``BatchSettings`` attached to a ``Model`` are ignored when the ``Model`` is executed as part of an ``Ensemble``.

The `params` factory method parameter for ``Model`` creation allows a user to define simulation parameters and
values through a dictionary. Using ``Model`` :ref:`file functions<files_doc>`, users can write these parameters to
a file in the ``Model`` working directory.

When a ``Model`` instance is passed to ``Experiment.generate()``, a
directory within the Experiment directory
is created to store input and output files from the ``Model``.

.. note::
    It is strongly recommended to invoke ``Experiment.generate()`` on the ``Model``
    instance before launching the ``Model``. If a path is not specified during
    ``Experiment.create_model()``, calling ``Experiment.generate()`` with the ``Model``
    instance will result in SmartSim generating a ``Model`` directory within the
    ``Experiment`` directory. This directory will be used to store the ``Model`` outputs
    and attached files.

.. _std_model_doc:
Example
=======
In this example, we provide a demonstration of how to initialize and launch a ``Model``
within an ``Experiment`` workflow.

.. dropdown:: Example Driver Script source code

    .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_init.py

All workflow entities are initialized through the :ref:`Experiment API<experiment_api>`.
Consequently, initializing a SmartSim ``Experiment`` is a prerequisite for ``Model``
initialization.

To initialize an instance of the ``Experiment`` class, import the SmartSim
``Experiment`` module and invoke the ``Experiment`` constructor
with a `name` and `launcher`:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_init.py
  :language: python
  :linenos:
  :lines: 1-4

A ``Model`` requires ``RunSettings`` objects to specify how the ``Model`` should be
executed within the workflow. We use the ``Experiment`` instance `exp` to
call the factory method ``Experiment.create_run_settings()`` to initialize a ``RunSettings``
object. Finally, we specify the executable `"echo"` to run the executable argument `"Hello World"`:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_init.py
    :language: python
    :linenos:
    :lines: 6-7

.. note::
    For more information on ``RunSettings`` objects, reference the :ref:`RunSettings<run_settings_doc>` documentation.

We now have a ``RunSettings`` instance named `model_settings` that contains all of the
information required to launch our application. Pass a `name` and the run settings instance
to the ``create_model`` factory method:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_init.py
  :language: python
  :linenos:
  :lines: 9-10

To create an isolated output directory for the ``Model``, invoke ``Experiment.generate()`` on the
``Model`` `model_instance`:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_init.py
  :language: python
  :linenos:
  :lines: 12-13

.. note::
    The ``Experiment.generate()`` step is optional; however, this step organizes the ``Experiment``
    entity output files into individual entity folders within the ``Experiment`` folder. Continue
    in the example for information on ``Model`` output generation or visit the
    :ref:`Output and Error Files<model_output_files>` section.

All entities are launched, monitored and stopped by the ``Experiment`` instance.
To start the ``Model``, invoke ``Experiment.start()`` on `model_instance`:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_init.py
  :language: python
  :linenos:
  :lines: 15-16

When the ``Experiment`` Python driver script is executed, two files from the `model_instance` will be created
in the Experiment working directory:

1. `model_instance.out` : this file will hold outputs produced by the `model_instance` workload.
2. `model_instance.err` : this file will hold any errors that occurred during `model_instance` execution.

.. _colo_model_doc:
======================
Colocated Orchestrator
======================
A SmartSim ``Model`` has the capability to share compute node(s) with a SmartSim ``Orchestrator`` in
a deployment known as a colocated ``Orchestrator``. In this scenario, the ``Orchestrator`` and ``Model`` share
compute resources. To achieve this, users need to initialize a ``Model`` instance using the
``Experiment.create_model()`` function and then utilize one of the three functions listed below to
colocate an ``Orchestrator`` with the ``Model``. This instructs SmartSim to launch an ``Orchestrator``
on the application compute node(s) before the ``Model`` execution.

There are **three** different Model API functions to colocate a ``Model``:

- ``Model.colocate_db_tcp()``: Colocate an ``Orchestrator`` instance and establish client communication using TCP/IP.
- ``Model.colocate_db_uds()``: Colocate an ``Orchestrator`` instance and establish client communication using Unix domain sockets (UDS).
- ``Model.colocate_db()``: (deprecated) An alias for `Model.colocate_db_tcp()`.

Each function initializes an unsharded ``Orchestrator`` accessible only to the ``Model`` processes on the same compute node. When the ``Model``
is started, the ``Orchestrator`` will be launched on the same compute resource as the ``Model``. Only the colocated ``Model``
may communicate with the ``Orchestrator`` via a SmartRedis ``Client`` by using the loopback TCP interface or
Unix Domain sockets. Extra parameters for the ``Orchestrator`` can be passed into the colocate functions above
via `kwargs`.

.. code-block:: python

    example_kwargs = {
        "maxclients": 100000,
        "threads_per_queue": 1,
        "inter_op_threads": 1,
        "intra_op_threads": 1
    }

For a walkthrough of how to colocate a ``Model``, navigate to the
:ref:`Colocated Orchestrator<colocated_orch_doc>` for instructions.

For users aiming to **optimize performance**, SmartSim offers the flexibility to specify the
number of CPUs to which the colocated ``Orchestrator`` should be pinned. This can be achieved using
the `custom_pinning` argument, which is recognized by both ``Model.colocate_db_uds()`` and
``Model.colocate_db_tcp()``. The `custom_pinning` argument accepts either a single integer or a
list of integers. If users want to explore different pinning scenarios, specify a list of integers
to the `custom_pinning` argument.

.. _files_doc:
=====
Files
=====
Overview
========
Applications often depend on external files (e.g. training datasets, evaluation datasets, etc)
to operate as intended. Users can instruct SmartSim to copy, symlink, or manipulate external files
prior to a ``Model`` launch via the ``Model.attach_generator_files()`` function.

.. note::
    Multiple calls to ``Model.attach_generator_files()`` will overwrite previous file configurations
    in the ``Model``.

To setup the run directory for the ``Model``, users should pass the list of files to
``Model.attach_generator_files()`` using the following arguments:

* `to_copy` (t.Optional[t.List[str]] = None): Files that are copied into the path of the ``Model``.
* `to_symlink` (t.Optional[t.List[str]] = None): Files that are symlinked into the path of the ``Model``.

User-formatted files can be attached using the `to_configure` argument. These files will be modified
during ``Model`` generation to replace tagged sections in the user-formatted files with
values from the `params` initializer argument used during ``Model`` creation:

* `to_configure` (t.Optional[t.List[str]] = None): Designed for text-based ``Model`` input files,
  `"to_configure"` is exclusive to the ``Model``. During ``Model`` directory generation, the attached
  files are parsed and specified tagged parameters are replaced with the `params` values that were
  specified in the ``Experiment.create_model()`` factory method of the ``Model``. The default tag is a semicolon
  (e.g., THERMO = ;THERMO;).

.. note::
    In the :ref:`Example<files_example_doc>` subsection, we provide an example using the value `to_configure`
    within ``attach_generator_files()``.

.. _files_example_doc:
Example
=======
This example demonstrates how to attach a file to a ``Model`` for parameter replacement at the time
of ``Model`` directory generation. This is accomplished using the `params` function parameter in
``Experiment.create_model()`` and the `to_configure` function parameter
in ``Model.attach_generator_files()``.

.. dropdown:: Example Driver Script source code

    .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_file.py

In this example, we have a text file named `params_inputs.txt`. Within the text file, is the parameter `THERMO`
that is required by the ``Model`` application at runtime:

.. code-block:: bash

   THERMO = ;THERMO;

In order to have the tagged parameter `;THERMO;` replaced with a usable value at runtime, two steps are required:

1. The `THERMO` variable must be included in ``Experiment.create_model()`` factory method as
   part of the `params` initializer argument.
2. The file containing the tagged parameter `;THERMO;`, `params_inputs.txt`, must be attached to the ``Model``
   via the ``Model.attach_generator_files()`` method as part of the `to_configure` function parameter.

To encapsulate our application within a ``Model``, we must first create an ``Experiment`` instance.
Begin by importing the ``Experiment`` module and initializing an ``Experiment``:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_file.py
  :language: python
  :linenos:
  :lines: 1-4

A SmartSim ``Model`` requires a ``RunSettings`` object to
specify the ``Model`` executable (e.g. the full path to a compiled binary) as well as
executable arguments and launch parameters. Create a simple ``RunSettings`` object
and specify the path to the executable script as an executable argument (`exe_args`):

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_file.py
  :language: python
  :linenos:
  :lines: 6-7

.. note::
    To read more on SmartSim ``RunSettings`` objects, reference the :ref:`RunSettings<run_settings_doc>` documentation.

Next, initialize a ``Model`` object via ``Experiment.create_model()``. Pass in the `model_settings` instance
and the `params` value:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_file.py
  :language: python
  :linenos:
  :lines: 9-10

We now have a ``Model`` instance named `model_instance`. Attach the text file, `params_inputs.txt`,
to the ``Model`` for use at entity runtime. To do so, use the
``Model.attach_generator_files()`` function and specify the `to_configure`
parameter with the path to the text file, `params_inputs.txt`:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_file.py
  :language: python
  :linenos:
  :lines: 12-13

To created an isolated directory for the ``Model`` outputs and configuration files, invoke ``Experiment.generate()``
on `model_instance` as an input parameter:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/model_file.py
  :language: python
  :linenos:
  :lines: 15-16

The contents of `getting-started/model_name/params_inputs.txt` at runtime are:

.. code-block:: bash

   THERMO = 1

.. _model_output_files:
======================
Output and Error Files
======================
By default, SmartSim stores the standard output and error of the ``Model`` in two files:

* `<model_name>.out`
* `<model_name>.err`

The files are created in the working directory of the ``Model``, and the filenames directly match the
``Model`` name. The `<model_name>.out` file logs standard outputs and the
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
Functions accessible through a ``Model`` object support loading ML models (TensorFlow, TensorFlow-lite,
PyTorch, and ONNX) and TorchScripts into standalone ``Orchestrators`` or colocated ``Orchestrators`` at
application runtime.

Users can follow **two** processes to load an ML model to the ``Orchestrator``:

- :ref:`from memory<in_mem_ML_model_ex>`
- :ref:`from file<from_file_ML_model_ex>`

Users can follow **three** processes to load a TorchScript to the ``Orchestrator``:

- :ref:`from memory<in_mem_TF_doc>`
- :ref:`from file<TS_from_file>`
- :ref:`from string<TS_raw_string>`

Once an ML model or TorchScript is loaded into the ``Orchestrator``, ``Model`` objects can
leverage ML capabilities by utilizing the SmartSim ``Client`` (:ref:`SmartRedis<smartredis-api>`)
to execute the stored ML models and TorchScripts.

.. _ai_model_doc:
AI Models
=========
When configuring a ``Model``, users can instruct SmartSim to load
Machine Learning (ML) models dynamically to the ``Orchestrator``. ML models added
are loaded into the ``Orchestrator`` prior to the execution of the ``Model``. To load an ML model
to the ``Orchestrator``, SmartSim users can provide the ML model **in-memory** or specify the **file path**
when using the ``Model.add_ml_model()`` function. SmartSim solely supports loading an ML model from file
for use within standalone ``Orchestrators``. The supported ML frameworks are TensorFlow,
TensorFlow-lite, PyTorch, and ONNX.

When attaching an ML model using ``Model.add_ml_model()``, the
following arguments are offered to customize the storage and execution of the ML model:

- `name` (str): name to reference the model in the ``Orchestrator``.
- `backend` (str): name of the backend (TORCH, TF, TFLITE, ONNX).
- `model` (t.Optional[str] = None): A ML model in memory (only supported for non-colocated ``Orchestrators``).
- `model_path` (t.Optional[str] = None): serialized ML model.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): name of device for execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `batch_size` (int = 0): batch size for execution, defaults to 0.
- `min_batch_size` (int = 0): minimum batch size for ML model execution, defaults to 0.
- `min_batch_timeout` (int = 0): time to wait for minimum batch size, defaults to 0.
- `tag` (str = ""): additional tag for ML model information, defaults to “”.
- `inputs` (t.Optional[t.List[str]] = None): ML model inputs (TF only), defaults to None.
- `outputs` (t.Optional[t.List[str]] = None): ML model outputs (TF only), defaults to None.

.. _in_mem_ML_model_ex:
-------------------------------------
Example: Attach an in-memory ML Model
-------------------------------------
This example demonstrates how to attach an in-memory ML model to a SmartSim ``Model``
to load into an ``Orchestrator`` at ``Model`` runtime.

.. dropdown:: Example Driver Script source code

    .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/in_mem_ml_model.py

.. note::
    This example assumes:

    - an ``Orchestrator`` is launched prior to the ``Model`` execution (colocated or standalone)
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define an in-memory Keras CNN**

The ML model must be defined using one of the supported ML frameworks. For the purpose of the example,
we define a Keras CNN in the ``Experiment`` driver script:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/in_mem_ml_model.py
  :language: python
  :linenos:
  :lines: 14-28

**Attach the ML model to a SmartSim Model**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add the in-memory TensorFlow model using
the ``Model.add_ml_model()`` function and specify the in-memory ML model to the function parameter `model`:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/in_mem_ml_model.py
  :language: python
  :linenos:
  :lines: 39-40

In the above ``smartsim_model.add_ml_model()`` code snippet, we pass in the following arguments:

-  `name` ("cnn"): A name to reference the ML model in the ``Orchestrator``.
-  `backend` ("TF"): Indicating that the ML model is a TensorFlow model.
-  `model` (model): The in-memory representation of the TensorFlow model.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When the ``Model`` is started via ``Experiment.start()``, the ML model will be loaded to the
launched ``Orchestrator``. The ML model can then be executed on the ``Orchestrator`` via a SmartSim
``Client`` (:ref:`SmartRedis<smartredis-api>`) within the application code.

.. _from_file_ML_model_ex:
----------------------------------------
Example: Attaching an ML Model from file
----------------------------------------
This example demonstrates how to attach a ML model from file to a SmartSim ``Model``
to load into an ``Orchestrator`` at ``Model`` runtime.

.. note::
    SmartSim supports loading ML models from file to standalone ``Orchestrators``.
    This feature is **not** supported for colocated ``Orchestrators``.

.. dropdown:: Example Driver Script source code

    .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/from_file_ml_model.py

.. note::
    This example assumes:

    - a standalone ``Orchestrator`` is launched prior to the ``Model`` execution
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define a Keras CNN script**

The ML model must be defined using one of the supported ML frameworks. For the purpose of the example,
we define the function `save_tf_cnn()` that saves a Keras CNN to a file named `model.pb` located in our
``Experiment`` path:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/from_file_ml_model.py
  :language: python
  :linenos:
  :lines: 14-25, 35-37

**Attach the ML model to a SmartSim Model**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add the TensorFlow model using
the ``Model.add_ml_model()`` function and specify the TensorFlow model path to the parameter `model_path`:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/from_file_ml_model.py
  :language: python
  :linenos:
  :lines: 39-40

In the above ``smartsim_model.add_ml_model()`` code snippet, we pass in the following arguments:

-  `name` ("cnn"): A name to reference the ML model in the ``Orchestrator``.
-  `backend` ("TF"): Indicating that the ML model is a TensorFlow model.
-  `model_path` (model_file): The path to the ML model script.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When the ``Model`` is started via ``Experiment.start()``, the ML model will be loaded to the
launched standalone ``Orchestrator``. The ML model can then be executed on the ``Orchestrator``
via a SmartRedis ``Client`` (:ref:`SmartRedis<smartredis-api>`) within the application code.

.. _TS_doc:
TorchScripts
============
When configuring a ``Model``, users can instruct SmartSim to load TorchScripts dynamically
to the ``Orchestrator``. TorchScripts added are loaded into the ``Orchestrator`` prior to
the execution of the ``Model``. To load a TorchScript to the ``Orchestrator``, SmartSim users
can follow one of the processes:

- :ref:`Define a TorchScript function in-memory<in_mem_TF_doc>`
   Use the ``Model.add_function()`` to instruct SmartSim to load an in-memory TorchScript to the ``Orchestrator``.
- :ref:`Define a TorchScript function from file<TS_from_file>`
   Provide file path to ``Model.add_script()`` to instruct SmartSim to load the TorchScript from file to the ``Orchestrator``.
- :ref:`Define a TorchScript function as string<TS_raw_string>`
   Provide function string to ``Model.add_script()`` to instruct SmartSim to load a raw string as a TorchScript function to the ``Orchestrator``.

.. note::
    SmartSim does **not** support loading in-memory TorchScript functions to colocated ``Orchestrators``.
    Users should instead load TorchScripts to a colocated ``Orchestrator`` from file or as a raw string.

Continue or select a process link to learn more on how each function (``Model.add_script()`` and ``Model.add_function()``)
dynamically load TorchScripts to launched ``Orchestrators``.

.. _in_mem_TF_doc:
-------------------------------
Attach an in-memory TorchScript
-------------------------------
Users can define TorchScript functions within the Python driver script
to attach to a ``Model``. This feature is supported by ``Model.add_function()`` which provides flexible
device selection, allowing users to choose between which device the TorchScript is executed on, `"GPU"` or `"CPU"`.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` function parameter.

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
to a standalone ``Orchestrator``.

.. dropdown:: Example Driver Script source code

    .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/in_mem_script.py

.. note::
    The example assumes:

    - a standalone ``Orchestrator`` is launched prior to the ``Model`` execution
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define an in-memory TF function**

To begin, define an in-memory TorchScript function within the ``Experiment`` driver script.
For the purpose of the example, we add a simple TorchScript function named `timestwo`:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/in_mem_script.py
  :language: python
  :linenos:
  :lines: 3-4

**Attach the in-memory TorchScript function to a SmartSim Model**

We use the ``Model.add_function()`` function to instruct SmartSim to load the TorchScript function `timestwo`
onto the launched standalone ``Orchestrator``. Specify the function `timestwo` to the `function`
parameter:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/in_mem_script.py
  :language: python
  :linenos:
  :lines: 15-16

In the above ``smartsim_model.add_function()`` code snippet, we input the following arguments:

-  `name` ("example_func"): A name to uniquely identify the ML model within the ``Orchestrator``.
-  `function` (timestwo): Name of the TorchScript function defined in the Python driver script.
-  `device` ("CPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the TorchScript to a non-existent ``Orchestrator``.

When the ``Model`` is started via ``Experiment.start()``, the TF function will be loaded to the
standalone ``Orchestrator``. The function can then be executed on the ``Orchestrator`` via a SmartRedis
``Client`` (:ref:`SmartRedis<smartredis-api>`) within the application code.

.. _TS_from_file:
------------------------------
Attach a TorchScript from file
------------------------------
Users can attach TorchScript functions from a file to a ``Model`` and upload them to a
colocated or standalone ``Orchestrator``. This functionality is supported by the ``Model.add_script()``
function's `script_path` parameter. The function supports
flexible device selection, allowing users to choose between `"GPU"` or `"CPU"` via the `device` parameter.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

When specifying a TorchScript using ``Model.add_script()``, the
following arguments are offered:

- `name` (str): Reference name for the script inside of the ``Orchestrator``.
- `script` (t.Optional[str] = None): String of function code (e.g. TorchScript code string).
- `script_path` (t.Optional[str] = None): path to TorchScript code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

.. _TS_from_file_ex:
Example: Loading a TorchScript from File
----------------------------------------
This example walks through the steps of instructing SmartSim to load a TorchScript from file
to a launched ``Orchestrator``.

.. dropdown:: Example Driver Script source code

    .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/from_file_script.py

.. note::
    This example assumes:

    - a ``Orchestrator`` is launched prior to the ``Model`` execution (Colocated or standalone)
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define a TorchScript script**

For the example, we create the Python script `torchscript.py`. The file contains a
simple torch function shown below:

.. code-block:: python

    def negate(x):
        return torch.neg(x)

**Attach the TorchScript script to a SmartSim Model**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add the TorchScript script using
``Model.add_script()`` by specifying the script path to the `script_path` parameter:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/from_file_script.py
  :language: python
  :linenos:
  :lines: 13-14

In the above ``smartsim_model.add_script()`` code snippet, we include the following arguments:

-  `name` ("example_script"): Reference name for the script inside of the ``Orchestrator``.
-  `script_path` ("path/to/torchscript.py"): Path to the script file.
-  `device` ("CPU"): device for script execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When `smartsim_model` is started via ``Experiment.start()``, the TorchScript will be loaded from file to the
``Orchestrator`` that is launched prior to the start of `smartsim_model`. The function can then be executed
on the ``Orchestrator`` via a SmartRedis ``Client`` (:ref:`SmartRedis<smartredis-api>`) within the application code.

.. _TS_raw_string:
---------------------------------
Define TorchScripts as raw string
---------------------------------
Users can upload TorchScript functions from string to colocated or
standalone ``Orchestrators``. This feature is supported by the
``Model.add_script()`` function's `script` parameter. The function supports
flexible device selection, allowing users to choose between `"GPU"` or `"CPU"` via the `device` parameter.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

When specifying a TorchScript using ``Model.add_script()``, the
following arguments are offered:

- `name` (str): Reference name for the script inside of the ``Orchestrator``.
- `script` (t.Optional[str] = None): String of function code (e.g. TorchScript code string).
- `script_path` (t.Optional[str] = None): path to TorchScript code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

.. _TS_from_file_ex:
Example: Loading a TorchScript from string
------------------------------------------
This example walks through the steps of instructing SmartSim to load a TorchScript
from string to a ``Orchestrator``.

.. dropdown:: Example Driver Script source code

    .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/string_script.py

.. note::
    This example assumes:

    - a ``Orchestrator`` is launched prior to the ``Model`` execution (standalone or colocated)
    - an initialized ``Model`` named `smartsim_model` exists within the ``Experiment`` workflow

**Define a string TorchScript**

Define the TorchScript code as a variable in the ``Experiment`` driver script:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/string_script.py
  :language: python
  :linenos:
  :lines: 12-13

**Attach the TorchScript function to a SmartSim Model**

Assuming an initialized ``Model`` named `smartsim_model` exists, we add a TensorFlow model using
the ``Model.add_script()`` function and specify the variable `torch_script_str` to the parameter
`script`:

.. literalinclude:: ../tutorials/doc_examples/model_doc_examples/string_script.py
  :language: python
  :linenos:
  :lines: 15-16

In the above ``smartsim_model.add_script()`` code snippet, we offer the following arguments:

-  `name` ("example_script"): key to store script under.
-  `script` (torch_script_str): TorchScript code.
-  `device` ("CPU"): device for script execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_model)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When the ``Model`` is started via ``Experiment.start()``, the TorchScript will be loaded to the
``Orchestrator`` that is launched prior to the start of the ``Model``.

.. _model_key_collision:
=========================
Data Collision Prevention
=========================
Overview
========
If an ``Experiment`` consists of multiple ``Models`` that use the same key names,
the names used to reference data, ML models, and scripts will be
identical, and without the use of SmartSim and SmartRedis prefix methods, ``Models``
will end up inadvertently accessing or overwriting each other’s data. To prevent this
situation, the SmartSim ``Model`` object supports key prefixing, which prepends
the name of the ``Model`` to the keys sent to the ``Orchestrator`` to create unique key names.
With this enabled, collision is avoided and ``Models`` can use same names within their applications.

The key components of SmartSim prefixing functionality include:

1. **Sending Data to the Orchestrator**: Users can send data to an ``Orchestrator``
   with the ``Model`` name prepended to the data name through SmartSim :ref:`Model functions<model_prefix_func>` and
   SmartRedis :ref:`Client functions<client_prefix_func>`.
2. **Retrieving Data from the Orchestrator**: Users can instruct a ``Client`` to prepend a
   ``Model`` name to a key during data retrieval, polling, or check for existence on the ``Orchestrator``
   through SmartRedis :ref:`Client functions<client_prefix_func>`.

For example, assume you have two ``Models`` in an ``Experiment``, named `model_0` and `model_1`. In each
application code you use the function ``Client.put_tensor("tensor_0", data)`` to send a tensor named `"tensor_0"`
to the same ``Orchestrator``. With ``Model`` key prefixing turned on, the `model_0` and `model_1`
applications can access their respective tensor `"tensor_0"` by name without overwriting or accessing
the other ``Models`` `"tensor_0"` tensor. In this scenario, the two tensors placed in the
``Orchestrator`` are `model_0.tensor_0` and `model_1.tensor_0`.

Enabling and Disabling
======================
SmartSim provides support for toggling prefixing on a ``Model`` for tensors, ``Datasets``,
lists, ML models, and scripts. Prefixing functions from the SmartSim :ref:`Model API<model_api>` and SmartRedis :ref:`Client API<smartredis-api>` rely on
each other to fully support SmartSim key prefixing. For example, to use the ``Client`` prefixing
functions, a user must enable prefixing on the ``Model`` through ``Model.enable_key_prefixing()``.
This function enables prefixing for tensors, ``Datasets`` and lists placed in an ``Orchestrator``
by the ``Model``. This configuration can be toggled within the ``Model`` application through
``Client`` functions, such as disabling tensor prefixing via ``Client.use_tensor_ensemble_prefix(False)``.

The interaction between the prefix SmartSim `Model functions<model_prefix_func>` and SmartRedis
`Client functions<client_prefix_func>` are documentation below.

.. _model_prefix_func:
---------------
Model functions
---------------
A ``Model`` object supports two prefixing functions: ``Model.enable_key_prefixing()`` and
``Model.register_incoming_entity()``.

To enable prefixing on a ``Model``, users must use the ``Model.enable_key_prefixing()``
function in the ``Experiment`` driver script. The key components of this function include:

- Activates prefixing for tensors, ``Datasets``, and lists sent to a ``Orchestrator`` from within
  the ``Model`` application.
- Enables access to prefixing ``Client`` functions within the ``Model`` application. This excludes
  the ``Client.set_data_source()`` function, where ``enable_key_prefixing()`` is not require access.

.. note::
    ML model and script prefixing is not automatically enabled through ``Model.enable_key_prefixing()``
    and rather must be enabled within the ``Model`` application using ``Client.use_model_ensemble_prefix()``.

To enable a SmartRedis ``Client`` to search a number of ``Model`` prefixes when requesting data from the ``Orchestrator``,
execute ``Model.register_incoming_entity()`` on the designated ``Model``. The key components of this function include:

- If a ``Model`` application requests prefixed tensors placed by another ``Model``, the interaction
  between the two entities must be registered in the ``Experiment`` driver script. If the interaction is not
  properly registered, ``Client.set_data_source()`` will not be able to find the ``Models`` prefix
  that placed the tensors in the ``Orchestrator``.
- This function enables users to execute ``Client`` functions such as ``Client.delete_tensor(tensor_name)`` on
  prefixed tensors without having to specify the prefix. The prefix is instead set via ``Client.set_data_source(model_name)``
  in the application and registered via ``Model.register_incoming_entity()`` in the ``Experiment`` driver script.

.. _client_prefix_func:
----------------
Client functions
----------------
A ``Client`` object supports five prefixing functions: ``Client.use_tensor_ensemble_prefix()``,
``Client.use_dataset_ensemble_prefix()``,  ``Client.use_list_ensemble_prefix()``,
``Client.use_model_ensemble_prefix()`` and ``Client.set_data_source()``.

To enable or disable SmartRedis data structure prefixing for tensors, ``Datasets``, aggregation lists, ML models
and scripts, SmartRedis ``Client`` offers functions per data structure:

- Tensor: ``Client.use_tensor_ensemble_prefix()``
- ``Dataset``: ``Client.use_dataset_ensemble_prefix()``
- Aggregation lists: ``Client.use_list_ensemble_prefix()``
- ML models/scripts: ``Client.use_model_ensemble_prefix()``

.. warning::
    To access the ``Client`` prefixing functions, prefixing must be enabled on the
    ``Model`` through ``Model.enable_key_prefixing()``. This function activates prefixing
    for tensors, ``Datasets`` and lists.

For examples on instructing SmartSim to send prefixed data to the ``Orchestrator``, navigate to the
:ref:`put/set operations<put_set_prefix>` section.

Users can instruct the SmartRedis ``Client`` to search a number of prefixes when requesting data from the ``Orchestrator``.
This is achieved through the ``Client.set_data_source()`` function used within the ``Model``
application. To implement this functionality:

1. Use ``Model.register_incoming_entity()`` on the ``Model`` intending to search for data in the ``Orchestrator``
   placed by a separate ``Model``.
2. Pass the SmartSim entity (e.g., another ``Model``) to ``Model.register_incoming_entity()`` in order to
   reference the ``Model`` prefix in the application code.
3. In the ``Model`` application, instruct the ``Client`` to prepend the specified ``Model`` name during key searches
   using ``Client.set_data_source("model_name")``.

For examples on instructing a ``Client`` to append a ``Model`` name to a key when searching for data, read the
:ref:`get operations<get_prefix>` section, :ref:`run operations<run_prefix>` section, or :ref:`copy/rename/delete
operations<copy_rename_del_prefix>` section.

.. _put_set_prefix:
Put/Set Operations
==================
In the following tabs we provide snippets of driver script and application code to demonstrate
activating and deactivating prefixing for tensors, ``Datasets``, lists, ML models and scripts using
SmartRedis put/get semantics.

.. tabs::

    .. group-tab:: Tensor
        **Activate Tensor Prefixing in the Driver Script**

        To activate prefixing on a ``Model`` in the driver script, a user must use the function
        ``Model.enable_key_prefixing()``. This functionality ensures that the ``Model`` name
        is prepended to each tensor name sent to the ``Orchestrator`` from within the ``Model``
        executable code.

        .. dropdown:: Example Driver Script source code

            .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/prefix_data.py

        In the driver script snippet below, we take an initialized ``Model`` and activate tensor
        prefixing through the ``enable_key_prefixing()`` function:

        .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/prefix_data.py
            :language: python
            :linenos:
            :lines: 6-12

        In the `model` application, two tensors named `tensor_1` and `tensor_2` are sent to a launched ``Orchestrator``.
        The contents of the ``Orchestrator`` after ``Model`` completion are:

        .. code-block:: bash

            1) "model_name.tensor_1"
            2) "model_name.tensor_2"

        You will notice that the ``Model`` name `model_name` has been prepended to each tensor name
        and stored in the ``Orchestrator``.

        **Activate Tensor Prefixing in the Application**

        Users can further configure tensor prefixing in the application by using
        the ``Client`` function ``use_tensor_ensemble_prefix()``. By specifying a boolean
        value to the function, users can turn prefixing on and off.

        .. note::
            To have access to ``Client.use_tensor_ensemble_prefix()``, prefixing must be enabled
            on the ``Model`` in the driver script via ``Model.enable_key_prefixing()``.

        In the application snippet below, we demonstrate enabling and disabling tensor prefixing:

        .. code-block:: python

            # Disable key prefixing
            client.use_tensor_ensemble_prefix(False)
            # Place a tensor in the Orchestrator
            client.put_tensor("tensor_1", np.array([1, 2, 3, 4]))
            # Enable key prefixing
            client.use_tensor_ensemble_prefix(True)
            # Place a tensor in the Orchestrator
            client.put_tensor("tensor_2", np.array([5, 6, 7, 8]))

        In the application, two tensors named `tensor_1` and `tensor_2` are sent to a launched ``Orchestrator``.
        The contents of the ``Orchestrator`` after ``Model`` completion are:

        .. code-block:: bash

            1) "tensor_1"
            2) "model_name.tensor_2"

        You will notice that the ``Model`` name `model_name` is **not** prefixed to `tensor_1` since
        we disabled tensor prefixing before sending the tensor to the ``Orchestrator``. However,
        when we enabled tensor prefixing and sent the second tensor, the ``Model`` name was prefixed
        to `tensor_2`.

    .. group-tab:: Dataset
        **Activate Dataset Prefixing in the Driver Script**

        To activate prefixing on a ``Model`` in the driver script, a user must use the function
        ``Model.enable_key_prefixing()``. This functionality ensures that the ``Model`` name
        is prepended to each ``Dataset`` name sent to the ``Orchestrator`` from within the ``Model``.

        .. dropdown:: Example Driver Script source code

            .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/prefix_data.py

        In the driver script snippet below, we take an initialized ``Model`` and activate ``Dataset``
        prefixing through the ``enable_key_prefixing()`` function:

        .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/prefix_data.py
            :language: python
            :linenos:
            :lines: 6-12

        In the `model` application, two Datasets named `dataset_1` and `dataset_2` are sent to a launched ``Orchestrator``.
        The contents of the ``Orchestrator`` after ``Model`` completion are:

        .. code-block:: bash

            1) "model_name.{dataset_1}.dataset_tensor_1"
            2) "model_name.{dataset_1}.meta"
            3) "model_name.{dataset_2}.dataset_tensor_2"
            4) "model_name.{dataset_2}.meta"

        You will notice that the ``Model`` name `model_name` has been prefixed to each ``Dataset`` name
        and stored in the ``Orchestrator``.

        **Activate Dataset Prefixing in the Application**

        Users can further configure ``Dataset`` prefixing in the application by using
        the ``Client`` function ``use_dataset_ensemble_prefix()``. By specifying a boolean
        value to the function, users can turn prefixing on and off.

        .. note::
            To have access to ``Client.use_dataset_ensemble_prefix()``, prefixing must be enabled
            on the ``Model`` in the driver script via ``Model.enable_key_prefixing()``.

        In the application snippet below, we demonstrate enabling and disabling ``Dataset`` prefixing:

        .. code-block:: python

            # Disable key prefixing
            client.use_dataset_ensemble_prefix(False)
            # Place a Dataset in the Orchestrator
            client.put_dataset(dataset_1)
            # Enable key prefixing
            client.use_dataset_ensemble_prefix(True)
            # Place a Dataset in the Orchestrator
            client.put_dataset(dataset_2)

        In the application, we have two ``Datasets`` named `dataset_1` and `dataset_2`.
        We then send them to a launched ``Orchestrator``. The contents of the ``Orchestrator`` after ``Model`` completion are:

        .. code-block:: bash

            1) "{dataset_1}.dataset_tensor_1"
            2) "{dataset_1}.meta"
            3) "model_name.{dataset_2}.dataset_tensor_1"
            4) "model_name.{dataset_2}.meta"

        You will notice that the ``Model`` name `model_name` is **not** prefixed to `dataset_1` since
        we disabled ``Dataset`` prefixing before sending the ``Dataset`` to the ``Orchestrator``. However,
        when we enabled ``Dataset`` prefixing and sent the second ``Dataset``, the ``Model`` name was prefixed
        to `dataset_2`.

    .. group-tab:: Aggregation List
        **Activate Aggregation List Prefixing in the Driver Script**

        To activate prefixing on a ``Model`` in the driver script, a user must use the function
        ``Model.enable_key_prefixing()``. This functionality ensures that the ``Model`` name
        is prepended to each list name sent to the ``Orchestrator`` from within the ``Model``.

        .. dropdown:: Example Driver Script source code

            .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/prefix_data.py

        In the driver script snippet below, we take an initialized ``Model`` and activate list
        prefixing through the ``enable_key_prefixing()`` function:

        .. literalinclude:: ../tutorials/doc_examples/model_doc_examples/prefix_data.py
            :language: python
            :linenos:
            :lines: 6-12

        In the `model` application, a list named `dataset_list` is sent to a launched ``Orchestrator``.
        The contents of the ``Orchestrator`` after ``Model`` completion are:

        .. code-block:: bash

            1) "model_name.dataset_list"

        You will notice that the ``Model`` name `model_name` has been prefixed to the list name
        and stored in the ``Orchestrator``.

        **Activate Aggregation List Prefixing in the Application**

        Users can further configure list prefixing in the application by using
        the ``Client`` function ``use_list_ensemble_prefix()``. By specifying a boolean
        value to the function, users can turn prefixing on and off.

        .. note::
            To have access to ``Client.use_list_ensemble_prefix()``, prefixing must be enabled
            on the ``Model`` in the driver script via ``Model.enable_key_prefixing()``.

        In the application snippet below, we demonstrate enabling and disabling list prefixing:

        .. code-block:: python

            # Disable key prefixing
            client.use_list_ensemble_prefix(False)
            # Place a Dataset in the Orchestrator
            client.put_dataset(dataset_1)
            # Place a list in the Orchestrator
            client.append_to_list("list_1", dataset_1)
            # Enable key prefixing
            client.use_dataset_ensemble_prefix(True)
            # Place a Dataset in the Orchestrator
            client.put_dataset(dataset_2)
            # Append Dataset to list in the Orchestrator
            client.append_to_list("list_2", dataset_2)

        In the application, two lists named `list_1` and `list_2` are sent to the ``Orchestrator``.
        The contents of the ``Orchestrator`` after ``Model`` completion are:

        .. code-block:: bash

            1) "list_1"
            2) "model_name.{dataset_1}.meta"
            3) "model_name.{dataset_1}.dataset_tensor_1"
            4) "model_name.list_2"
            5) "model_name.{dataset_2}.meta"
            6) "model_name.{dataset_2}.dataset_tensor_2"

        You will notice that the ``Model`` name `model_name` is **not** prefixed to `list_1` since
        we disabled list prefixing before sending the list to the ``Orchestrator``. However,
        when we enabled list prefixing and sent the second list, the ``Model`` name was prefixed
        to `list_2` as well as the list ``Dataset`` members.

        .. note::
            The ``Datasets`` sent to the ``Orchestrator`` are all prefixed. This is because
            ``Model.enable_key_prefixing()`` turns on prefixing for tensors, ``Datasets`` and lists.

    .. group-tab:: ML Model
        **Activate ML model Prefixing in the Application**

        Users can configure ML model prefixing in the application by using
        the ``Client`` function ``use_model_ensemble_prefix()``. By specifying a boolean
        value to the function, users can turn prefixing on and off.

        .. note::
            To have access to ``Client.use_model_ensemble_prefix()``, prefixing must be enabled
            on the ``Model`` in the driver script via ``Model.enable_key_prefixing()``.

        In the application snippet below, we demonstrate enabling and disabling ML model prefixing:

        .. code-block:: python

            # Disable ML model prefixing
            client.use_model_ensemble_prefix(False)
            # Send ML model to the Orchestrator
            client.set_model(
                "ml_model_1", serialized_model_1, "TF", device="CPU", inputs=inputs, outputs=outputs
            )
            # Enable ML model prefixing
            client.use_model_ensemble_prefix(True)
            # Send prefixed ML model to the Orchestrator
            client.set_model(
                "ml_model_2", serialized_model_2, "TF", device="CPU", inputs=inputs, outputs=outputs
            )

        In the application, two ML models named `ml_model_1` and `ml_model_2` are sent
        to a launched ``Orchestrator``. The contents of the ``Orchestrator`` after ``Model`` completion are:

        .. code-block:: bash

            1) "ml_model_1"
            2) "model_name.ml_model_2"

        You will notice that the ``Model`` name `model_name` is **not** prefixed to `ml_model_1` since
        we disabled ML model prefixing before sending the ML model to the ``Orchestrator``. However,
        when we enabled ML model prefixing and sent the second ML model, the ``Model`` name was prefixed
        to `ml_model_2`.

    .. group-tab:: Script
        **Activate Script Prefixing in the Application**

        Users can configure script prefixing in the application by using
        the ``Client`` function ``use_model_ensemble_prefix()``. By specifying a boolean
        value to the function, users can turn prefixing on and off.

        .. note::
            To have access to ``Client.use_model_ensemble_prefix()``, prefixing must be enabled
            on the ``Model`` in the driver script via ``Model.enable_key_prefixing()``.

        In the application snippet below, we demonstrate enabling and disabling script prefixing:

        .. code-block:: python

            # Disable script prefixing
            client.use_model_ensemble_prefix(False)
            # Store a script in the Orchestrator
            client.set_function("script_1", script_1)
            # Enable script prefixing
            client.use_model_ensemble_prefix(True)
            # Store a prefixed script in the Orchestrator
            client.set_function("script_2", script_2)

        In the application, two ML models named `script_1` and `script_2` are sent
        to a launched ``Orchestrator``. The contents of the ``Orchestrator`` after ``Model`` completion are:

        .. code-block:: bash

            1) "script_1"
            2) "model_name.script_2"

        You will notice that the ``Model`` name `model_name` is **not** prefixed to `script_1` since
        we disabled script prefixing before sending the script to the ``Orchestrator``. However,
        when we enabled script prefixing and sent the second script, the ``Model`` name was prefixed
        to `script_2`.

.. _get_prefix:

Get Operations
==============
In the following sections, we walk through snippets of application code to demonstrate the retrieval
of prefixed tensors, ``Datasets``, lists, ML models, and scripts using SmartRedis put/get
semantics. The examples demonstrate retrieval within the same application where the data
structures were placed, as well as scenarios where data structures are placed by separate
applications.

.. tabs::

    .. group-tab:: Tensor
        **Retrieve A Tensor Placed By The Same Application**

        SmartSim supports retrieving prefixed tensors sent to the ``Orchestrator`` from within the
        same application where the tensor was placed. To achieve this, users must
        provide the ``Model`` name that stored the tensor to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key searches. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name
        in the driver script.

        As an example, we placed a prefixed tensor on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.tensor_name"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        In the application snippet below, we demonstrate retrieving the tensor:

        .. code-block:: python

            # Set the name to prepend to key searches
            client.set_data_source("model_1")
            # Retrieve the prefixed tensor
            tensor_data = client.get_tensor("tensor_name")
            # Log the tensor data
            client.log_data(LLInfo, f"The tensor value is: {tensor_data}")

        In the `model.out` file, the ``Client`` will log the message::
            Default@00-00-00:The tensor value is: [1 2 3 4]

        **Retrieve A Tensor Placed By An Alternate Application**

        SmartSim supports retrieving prefixed tensors sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the tensor
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data in the
        driver script.

        In the example, a ``Model`` named `model_1` has placed a tensor in a standalone
        ``Orchestrator`` with prefixing enabled on the ``Model``. The contents of the ``Orchestrator``
        are as follows:

        .. code-block:: bash

            1) "model_1.tensor_name"

        We create a separate ``Model``, named `model_2`, with the executable application code below.

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        Here we retrieve the stored tensor named `tensor_name`:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Retrieve the prefixed tensor
            tensor_data = client.get_tensor("tensor_name")
            # Log the tensor data
            client.log_data(LLInfo, f"The tensor value is: {tensor_data}")

        In the `model.out` file, the ``Client`` will log the message::
            Default@00-00-00:The tensor value is: [1 2 3 4]

    .. group-tab:: Dataset
        **Retrieve A Dataset Placed By The Same Application**

        SmartSim supports retrieving prefixed ``Datasets`` sent to the ``Orchestrator`` from within the
        same application where the ``Dataset`` was placed. To achieve this, users must
        provide the ``Model`` name that stored the ``Dataset`` to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key searches. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed ``Dataset`` on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.{dataset_name}.dataset_tensor"
            2) "model_1.{dataset_name}.meta"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        In the application snippet below, we demonstrate retrieving the ``Dataset``:

        .. code-block:: python

            # Set the name to prepend to key searches
            client.set_data_source("model_1")
            # Retrieve the prefixed Dataset
            dataset_data = client.get_dataset("dataset_name")
            # Log the Dataset data
            client.log_data(LLInfo, f"The Dataset value is: {dataset_data}")

        In the `model.out` file, the ``Client`` will log the message:

        .. code-block:: bash

            Default@00-00-00:Default@00-00-00:The dataset value is:

            DataSet (dataset_name):
                Tensors:
                    dataset_tensor:
                        type: 16 bit unsigned integer
                        dimensions: [4]
                        elements: 4
                Metadata:
                    none

        **Retrieve A Dataset Placed By An Alternate Application**

        SmartSim supports retrieving prefixed ``Datasets`` sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the ``Dataset``
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data.

        In the example, a ``Model`` named `model_1` has placed a ``Dataset`` in a standalone
        ``Orchestrator`` with prefixing enabled on the ``Model``. The contents of the ``Orchestrator``
        are as follows:

        .. code-block:: bash

            1) "model_1.{dataset_name}.dataset_tensor"
            2) "model_1.{dataset_name}.meta"

        We create a separate ``Model``, named `model_2`, with the executable application code below.

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        Here we retrieve the stored ``Dataset`` named `dataset_name`:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Retrieve the prefixed Dataset
            dataset_data = client.get_dataset("dataset_name")
            # Log the Dataset data
            client.log_data(LLInfo, f"The Dataset value is: {dataset_data}")

        In the `model.out` file, the ``Client`` will log the message:

        .. code-block:: bash

            Default@00-00-00:Default@00-00-00:The Dataset value is:

            DataSet (dataset_name):
                Tensors:
                    dataset_tensor:
                        type: 16 bit unsigned integer
                        dimensions: [4]
                        elements: 4
                Metadata:
                    none

    .. group-tab:: Aggregation List
        **Retrieve A Aggregation List Placed By The Same Application**

        SmartSim supports retrieving prefixed lists sent to the ``Orchestrator`` from within the
        same application where the list was placed. To achieve this, users must
        provide the ``Model`` name that stored the list to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key searches. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed list on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.dataset_list"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        In the application snippet below, we demonstrate checking the length of the list:

        .. code-block:: python

            # Set the name to prepend to key searches
            client.set_data_source("model_1")
            # Retrieve the prefixed list
            list_data = client.get_datasets_from_list("dataset_list")
            # Log the list data
            client.log_data(LLInfo, f"The length of the list is: {len(list_data)}")

        In the `model.out` file, the ``Client`` will log the message::
            The length of the list is: 1

        **Retrieve A Aggregation List Placed By An Alternate Application**

        SmartSim supports retrieving prefixed lists sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the list
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data.

        In the example, a ``Model`` named `model_1` has placed a list in a standalone
        ``Orchestrator`` with prefixing enabled on the ``Model``. The contents of the ``Orchestrator``
        are as follows:

        .. code-block:: bash

            1) "model_name.dataset_list"

        We create a separate ``Model``, named `model_2`, with the executable application code below.
        
        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.
        
        Here we check the length of the list named `dataset_list`:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Retrieve the prefixed list
            list_data = client.get_datasets_from_list("dataset_list")
            # Log the list data
            client.log_data(LLInfo, f"The length of the list is: {len(list_data)}")

        In the `model.out` file, the ``Client`` will log the message::
            The length of the list is: 1

    .. group-tab:: ML Model
        **Retrieve A ML Model Placed By The Same Application**

        SmartSim supports retrieving prefixed ML models sent to the ``Orchestrator`` from within the
        same application where the ML model was placed. To achieve this, users must
        provide the ``Model`` name that stored the ML model to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key searches. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed ML model on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.mnist_cnn"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        In the application snippet below, we demonstrate retrieving the ML model:

        .. code-block:: python

            # Set the name to prepend to key searches
            client.set_data_source("model_1")
            # Retrieve the prefixed ML model
            model_data = client.get_model("mnist_cnn")

        **Retrieve A ML Model Placed By An Alternate Application**

        SmartSim supports retrieving prefixed ML model sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the ML model
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data.

        In the example, a ``Model`` named `model_1` has placed a ML model in a standalone
        ``Orchestrator`` with prefixing enabled on the ``Model``. The contents of the ``Orchestrator``
        are as follows:

        .. code-block:: bash

            1) "model_1.mnist_cnn"

        We create a separate ``Model``, named `model_2`, with the executable application code below.

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        Here we retrieve the stored ML model named `mnist_cnn`:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Retrieve the prefixed model
            model_data = client.get_model("mnist_cnn")

    .. group-tab:: Script
        **Retrieve A Script Placed By The Same Application**

        SmartSim supports retrieving prefixed scripts sent to the ``Orchestrator`` from within the
        same application where the script was placed. To achieve this, users must
        provide the ``Model`` name that stored the script to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key searches. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed script on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.normalizer"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        In the application snippet below, we demonstrate retrieving the script:

        .. code-block:: python

            # Set the name to prepend to key searches
            client.set_data_source("model_1")
            # Retrieve the prefixed script
            script_data = client.get_script("normalizer")
            # Log the script data
            client.log_data(LLInfo, f"The script data is: {script_data}")

        In the `model.out` file, the ``Client`` will log the message:

        .. code-block:: bash

            The script data is: def normalize(X):
            """Simple function to normalize a tensor"""
            mean = X.mean()
            std = X.std()

            return (X-mean)/std

        **Retrieve A Script Placed By An Alternate Application**

        SmartSim supports retrieving prefixed scripts sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the script
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data.

        In the example, a ``Model`` named `model_1` has placed a script in a standalone
        ``Orchestrator`` with prefixing enabled on the ``Model``. The contents of the ``Orchestrator``
        are as follows:

        .. code-block:: bash

            1) "model_1.normalizer"

        We create a separate ``Model``, named `model_2`, with the executable application code below.

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        Here we retrieve the stored script named `normalizer`:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Retrieve the prefixed script
            script_data = client.get_script("model_1.normalizer")
            # Log the script data
            client.log_data(LLInfo, f"The script data is: {script_data}")

        In the `model.out` file, the ``Client`` will log the message:

        .. code-block:: bash

            The script data is: def normalize(X):
            """Simple function to normalize a tensor"""
            mean = X.mean()
            std = X.std()

            return (X-mean)/std

.. _run_prefix:
Run Operations
==============
In the following sections, we walk through snippets of application code to demonstrate executing
prefixed ML models and scripts using SmartRedis run semantics. The examples demonstrate
executing within the same application where the ML Model and Script were placed, as well as scenarios
where ML Model and Script are placed by separate applications.

.. tabs::

    .. group-tab:: ML Model
        **Access ML Models From The Application They Were Loaded In**

        SmartSim supports executing prefixed ML models with prefixed tensors sent to the ``Orchestrator`` from within
        the same application that the ML model was placed. To achieve this, users must
        provide the ``Model`` name that stored the ML model and input tensors to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key names. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed ML model and tensor on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.mnist_cnn"
            2) "model_1.mnist_images"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        In the application snippet below, we demonstrate running the ML model:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Run the ML model
            client.run_model(name="mnist_cnn", inputs=["mnist_images"], outputs=["Identity"])

        The ``Orchestrator`` now contains prefixed output tensors:

        .. code-block:: bash

            1) "model_1.Identity"
            2) "model_1.mnist_cnn"
            3) "model_1.mnist_images"

        .. note::
            The output tensors are prefixed because we executed ``model_1.enable_key_prefixing()``
            in the driver script which enables prefixing for tensors, ``Datasets`` and lists.

        **Access ML Models From Outside The Application They Were Loaded In**

        SmartSim supports executing prefixed ML models with prefixed tensors sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the ML model and tensor
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data.

        In the example, a ``Model`` named `model_1` has placed a ML model and tensor in a standalone
        ``Orchestrator`` with prefixing enabled on the ``Model``. The contents of the ``Orchestrator``
        are as follows:

        .. code-block:: bash

            1) "model_1.mnist_cnn"
            2) "model_1.mnist_images"

        We create a separate ``Model``, named `model_2`, with the executable application code below.

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        In the application snippet below, we demonstrate running the ML model:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Run the ML model
            client.run_model(name="mnist_cnn", inputs=["mnist_images"], outputs=["Identity"])

        The ``Orchestrator`` now contains prefixed output tensors:

        .. code-block:: bash

            1) "model_2.Identity"
            2) "model_1.mnist_cnn"
            3) "model_1.mnist_images"

        .. note::
            The output tensors are prefixed because we executed ``model_2.enable_key_prefixing()``
            in the driver script which enables prefixing for tensors, ``Datasets`` and lists.

    .. group-tab:: Script

        **Access Scripts From The Application They Were Loaded In**

        SmartSim supports executing prefixed scripts with prefixed tensors sent to the ``Orchestrator`` from within
        the same application that the script was placed. To achieve this, users must
        provide the ``Model`` name that stored the script and input tensors to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key names. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed script and tensor on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.normalizer"
            2) "model_1.X_rand"

        To run the script, the prefixed script name `"model_name.normalizer"` and prefixed
        input tensors `"model_name.X_rand"` must be provided, as demonstrated below:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Run the script
            client.run_script("normalizer", "normalize", inputs=["X_rand"], outputs=["X_norm"])

        The ``Orchestrator`` now contains prefixed output tensors:

        .. code-block:: bash

            1) "model_1.normalizer"
            2) "model_1.X_rand"
            3) "model_1.X_norm"

        .. note::
            The output tensors are prefixed because we executed ``model_1.enable_key_prefixing()``
            in the driver script which enables prefixing for tensors, ``Datasets`` and lists.

        **Access Scripts From Outside The Application They Were Loaded In**

        SmartSim supports executing prefixed scripts with prefixed tensors sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the script and tensor
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data.

        In the example, a ``Model`` named `model_1` has placed a script and tensor in a standalone
        ``Orchestrator`` with prefixing enabled on the ``Model``. The contents of the ``Orchestrator``
        are as follows:

        .. code-block:: bash

            1) "model_1.normalizer"
            2) "model_1.X_rand"

        We create a separate ``Model``, named `model_2`, with the executable application code below.

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for use in ``Client.set_data_source()``.

        In the application snippet below, we demonstrate running the script:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Run the script
            client.run_script("normalizer", "normalize", inputs=["X_rand"], outputs=["X_norm"])

        The ``Orchestrator`` now contains prefixed output tensors:

        .. code-block:: bash

            1) "model_1.normalizer"
            2) "model_1.X_rand"
            3) "model_2.X_norm"

        .. note::
            The output tensors are prefixed because we executed ``model_2.enable_key_prefixing()``
            in the driver script which enables prefixing for tensors, ``Datasets`` and lists.

.. _copy_rename_del_prefix:
Copy/Rename/Delete Operations
=============================
In the following sections, we walk through snippets of application code to demonstrate the copy, rename and delete
operations on prefixed tensors, ``Datasets``, lists, ML models, and scripts. The examples
demonstrate these operations within the same script where the data
structures were placed, as well as scenarios where data structures are placed by separate
scripts.

.. tabs::

    .. group-tab:: Tensor
        **Copy/Rename/Delete Operations On Tensors In The Same Application**

        SmartSim supports copy/rename/delete operations on prefixed tensors sent to the ``Orchestrator`` from within
        the same application that the tensor was placed. To achieve this, users must
        provide the ``Model`` name that stored the tensor to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key names. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed tensor on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.tensor"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        To rename the tensor in the ``Orchestrator``, we provide self ``Model`` name
        to ``Client.set_data_source()`` then execute the function ``rename_tensor()``:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Rename the tensor
            client.rename_tensor("tensor", "renamed_tensor")

        Because prefixing is enabled on the ``Model`` via ``enable_key_prefixing()`` in the driver script,
        SmartSim will keep the prefix on the tensor but replace the tensor name as shown in the ``Orchestrator``:

        .. code-block:: bash

            1) "model_1.renamed_tensor"

        Next, we copy the prefixed tensor to a new destination:

        .. code-block:: python

            client.copy_tensor("renamed_tensor", "copied_tensor")

        Since tensor prefixing is enabled on the ``Client``, the `copied_tensor` is prefixed:

        .. code-block:: bash

            1) "model_1.renamed_tensor"
            2) "model_1.copied_tensor"

        Next, delete `renamed_tensor`:

        .. code-block:: python

            client.delete_tensor("renamed_tensor")

        The contents of the ``Orchestrator`` are:

        .. code-block:: bash

            1) "model_1.copied_tensor"

        **Copy/Rename/Delete Operations On Tensors Placed By An Alternate Application**

        SmartSim supports copy/rename/delete operations on prefixed tensors sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the tensor
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data.

        In the example, a ``Model`` named `model_1` has placed a tensor in a standalone ``Orchestrator`` with prefixing enabled
        on the ``Client``. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.tensor"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        From within a separate ``Model`` named `model_2`, we perform basic copy/rename/delete operations.
        To instruct the ``Client`` to prepend a ``Model`` name to all key searches, use the
        ``Client.set_data_source()`` function. Specify the ``Model`` name `model_1`
        that placed the tensor in the ``Orchestrator``:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")

        To rename the tensor in the ``Orchestrator``, we provide the tensor name:

        .. code-block:: python

            client.rename_tensor("tensor", "renamed_tensor")

        SmartSim will replace the prefix with the current ``Model`` name since prefixing is enabled
        on the current ``Model``. The contents of the ``Orchestrator`` are:

        .. code-block:: bash

            1) "model_2.renamed_tensor"

        .. note::
            In the driver script, we also register `model_2` as an entity on itself via ``model_2.register_incoming_entity(model_2)``.
            This way we can use ``Client.set_data_source()`` to search for data placed by `model_2`.

        Next, we copy the prefixed tensor to a new destination:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_2")
            # Copy the tensor data
            client.copy_tensor("renamed_tensor", "copied_tensor")

        The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_2.renamed_tensor"
            2) "model_2.copied_tensor"

        Next, delete `copied_tensor` by specifying the name:

        .. code-block:: python

            client.delete_tensor("copied_tensor")

        The contents of the ``Orchestrator`` are:

        .. code-block:: bash

            1) "model_2.renamed_tensor"

    .. group-tab:: Dataset
        **Copy/Rename/Delete Operations On A Dataset In The Same Application**

        SmartSim supports copy/rename/delete operations on prefixed ``Datasets`` sent to the ``Orchestrator`` from within
        the same application that the ``Dataset`` was placed. To achieve this, users must
        provide the ``Model`` name that stored the ``Dataset`` to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key names. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed ``Dataset`` on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.{dataset}.dataset_tensor"
            2) "model_1.{dataset}.meta"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        To rename the ``Dataset`` in the ``Orchestrator``, we provide self ``Model`` name
        to ``Client.set_data_source()`` then execute the function ``rename_tensor()``:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Rename the Dataset
            client.rename_dataset("dataset", "renamed_dataset")

        Because prefixing is enabled on the ``Model`` via ``enable_key_prefixing()`` in the driver script,
        SmartSim will keep the prefix on the ``Dataset`` but replace the ``Dataset`` name as shown in the ``Orchestrator``:

        .. code-block:: bash

            1) "model_1.{renamed_dataset}.dataset_tensor"
            2) "model_1.{renamed_dataset}.meta"
        
        Next, we copy the prefixed ``Dataset`` to a new destination:

        .. code-block:: python

            client.copy_dataset("renamed_dataset", "copied_dataset")
        
        Since ``Dataset`` prefixing is enabled on the ``Client``, the `copied_dataset` is prefixed:

        .. code-block:: bash

            1) "model_1.{renamed_dataset}.dataset_tensor"
            2) "model_1.{renamed_dataset}.meta"
            3) "model_1.{copied_dataset}.dataset_tensor"
            4) "model_1.{copied_dataset}.meta"

        Next, delete `copied_dataset`:

        .. code-block:: python

            client.delete_dataset("model_name.copied_dataset")

        The contents of the ``Orchestrator`` are:

        .. code-block:: bash

            1) "model_1.{renamed_dataset}.dataset_tensor"
            2) "model_1.{renamed_dataset}.meta"

        **Copy/Rename/Delete Operations On Datasets Placed By An Alternate Application**

        SmartSim supports copy/rename/delete operations on prefixed ``Datasets`` sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the ``Dataset``
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data.

        In the example, a ``Model`` named `model_1` has placed a ``Dataset`` in a standalone ``Orchestrator`` with prefixing enabled
        on the ``Client``. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.{dataset}.dataset_tensor"
            2) "model_1.{dataset}.meta"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        From within a separate ``Model`` named `model_2`, we perform basic copy/rename/delete operations.
        To instruct the ``Client`` to prepend a ``Model`` name to all key searches, use the
        ``Client.set_data_source()`` function. Specify the ``Model`` name `model_1`
        that placed the ``Dataset`` in the ``Orchestrator``:

        .. code-block:: python

            client.set_data_source("model_1")

        To rename the ``Dataset`` in the ``Orchestrator``, we provide the ``Dataset`` `name`:

        .. code-block:: python

            client.rename_tensor("dataset", "renamed_dataset")

        SmartSim will replace the prefix with the current ``Model`` name since prefixing is enabled
        on the current ``Model`` via ``Model.enable_key_prefixing()`` in the driver script.
        The contents of the ``Orchestrator`` are:

        .. code-block:: bash

            1) "model_2.{renamed_dataset}.dataset_tensor"
            2) "model_2.{renamed_dataset}.meta"

        .. note::
            In the driver script, we also register `model_2` as an entity on itself via ``model_2.register_incoming_entity(model_2)``.
            This way we can use ``Client.set_data_source()`` to search for data placed by `model_2`.

        Next, we copy the prefixed ``Dataset`` to a new destination:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_2")
            # Copy the tensor data
            client.copy_dataset("renamed_dataset", "copied_dataset")

        The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_2.{renamed_dataset}.dataset_tensor"
            2) "model_2.{renamed_dataset}.meta"
            3) "model_2.{copied_dataset}.dataset_tensor"
            4) "model_2.{copied_dataset}.meta"

        Next, delete `copied_dataset` by specifying the name:

        .. code-block:: python

            client.delete_dataset("copied_tensor")

        The contents of the ``Orchestrator`` are:

        .. code-block:: bash

            1) "model_2.{renamed_dataset}.dataset_tensor"
            2) "model_2.{renamed_dataset}.meta"

    .. group-tab:: Aggregation List
        **Copy/Rename/Delete Operations On A Aggregation List In The Same Application**

        SmartSim supports copy/rename/delete operations on prefixed lists sent to the ``Orchestrator`` from within
        the same application that the list was placed. To achieve this, users must
        provide the ``Model`` name that stored the list to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key names. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed list on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.list_of_datasets"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        To rename the list in the ``Orchestrator``, we provide self ``Model`` name
        to ``Client.set_data_source()`` then execute the function ``rename_list()``:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Rename the list
            client.rename_list("list_of_datasets", "renamed_list")

        Because prefixing is enabled on the ``Model`` via ``enable_key_prefixing()`` in the driver script,
        SmartSim will keep the prefix on the list but replace the list name as shown in the ``Orchestrator``:

        .. code-block:: bash

            1) "model_1.renamed_list"

        Next, we copy the prefixed list to a new destination:

        .. code-block:: python

            client.copy_list("renamed_list", "copied_list")

        Since list prefixing is enabled on the ``Client``, the `copied_list` is prefixed:

        .. code-block:: bash

            1) "model_1.renamed_list"
            2) "model_1.copied_list"

        Next, delete `copied_list`:

        .. code-block:: python

            client.delete_list("copied_list")

        The contents of the ``Orchestrator`` are:

        .. code-block:: bash

            1) "model_1.renamed_list"

        **Copy/Rename/Delete Operations On Aggregation Lists Placed By An Alternate Application**

        SmartSim supports copy/rename/delete operations on prefixed lists sent to the ``Orchestrator`` by separate
        ``Models``. To achieve this, users need to provide the ``Model`` name that stored the list
        to ``Client.set_data_source()``. This action instructs the ``Client`` to prepend the ``Model``
        name to all key searches. For SmartSim to recognize the ``Model`` name as a data source,
        users must execute the ``Model.register_incoming_entity()`` function on the ``Model``
        responsible for the search and pass the ``Model`` instance that stored the data.

        In the example, a ``Model`` named `model_1` has placed a list in a standalone ``Orchestrator`` with prefixing enabled
        on the ``Client``. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.list_of_datasets"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_2`,
            we execute ``model_2.register_incoming_entity(model_1)``. By passing the producer ``Model``
            instance to the consumer ``Model``, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        From within a separate ``Model`` named `model_2`, we perform basic copy/rename/delete operations.
        To instruct the ``Client`` to prepend a ``Model`` name to all key searches, use the
        ``Client.set_data_source()`` function. Specify the ``Model`` name `model_1`
        that placed the list in the ``Orchestrator``:

        .. code-block:: python

            client.set_data_source("model_1")

        To rename the list in the ``Orchestrator``, we provide the list name:

        .. code-block:: python

            client.rename_list("list_of_datasets", "renamed_list")

        SmartSim will replace the prefix with the current ``Model`` name since prefixing is enabled
        on the current ``Model``. The contents of the ``Orchestrator`` are:

        .. code-block:: bash

            1) "model_2.renamed_list"

        .. note::
            In the driver script, we also register `model_2` as an entity on itself via ``model_2.register_incoming_entity(model_2)``.
            This way we can use ``Client.set_data_source()`` to search for data placed by `model_2`.

        Next, we copy the prefixed list to a new destination:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_2")
            # Copy the tensor data
            client.copy_dataset("renamed_list", "copied_list")

        The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_2.renamed_list"
            2) "model_2.copied_list"

        Next, delete `copied_list` by specifying the name:

        .. code-block:: python

            client.delete_list("copied_list")

        The contents of the ``Orchestrator`` are:

        .. code-block:: bash

            1) "model_2.renamed_list"

    .. group-tab:: ML Model
        **Delete ML Models From The Application They Were Loaded In**

        SmartSim supports delete operations on prefixed ML models sent to the ``Orchestrator`` from within
        the same application that the ML model was placed. To achieve this, users must
        provide the ``Model`` name that stored the ML model to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key names. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed ML model on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        .. code-block:: bash

            1) "model_1.ml_model"

        To delete the ML model in the ``Orchestrator``, we provide self ``Model`` name
        to ``Client.set_data_source()`` then execute the function ``delete_model()``:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Delete the ML model
            client.delete_model("ml_model")

        **Delete A ML Model Placed By An Alternate Application**

        SmartSim supports delete operations on prefixed ML models sent to the ``Orchestrator`` by separate ``Models``.
        To do so, users must provide the ``Model`` name that stored the ML model to ``Client.set_data_source()``.
        This will instruct the ``Client`` to prepend the ``Model`` name input to all key searches.

        In the example, a ``Model`` named `model_1` has placed a ML model in a standalone ``Orchestrator`` with prefixing enabled
        on the ``Client``. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.ml_model"

        From within a separate ``Model`` named `model_2`, we perform a basic delete operation.
        To instruct the ``Client`` to prepend a ``Model`` name to all key searches, use the
        ``Client.set_data_source()`` function. Specify the ``Model`` name `model_1`
        that placed the list in the ``Orchestrator``:

        .. code-block:: python

            client.set_data_source("model_1")

        To delete the ML model in the ``Orchestrator``, we provide the ML model name:

        .. code-block:: python

            client.delete_model("ml_model")

    .. group-tab:: Script

        **Delete Scripts From The Application They Were Loaded In**

        SmartSim supports delete operations on prefixed scripts sent to the ``Orchestrator`` from within
        the same application that the script was placed. To achieve this, users must
        provide the ``Model`` name that stored the script to ``Client.set_data_source()``. This action
        instructs the ``Client`` to prepend the ``Model`` name to all key names. For SmartSim to
        recognize the ``Model`` name as a data source, users must execute the
        ``Model.register_incoming_entity()`` function on the ``Model`` and pass the self ``Model`` name.

        As an example, we placed a prefixed script on the ``Orchestrator`` within a ``Model`` named
        `model_1`. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.script"

        .. note::
            In the driver script, after initializing the ``Model`` instance named `model_1`,
            we execute ``model_1.register_incoming_entity(model_1)``. By passing the ``Model``
            instance to itself, we instruct SmartSim to recognize the name of `model_1` as a valid data
            source for subsequent use in ``Client.set_data_source()``.

        To delete the script in the ``Orchestrator``, we provide the full list name:

        .. code-block:: python

            # Set the Model source name
            client.set_data_source("model_1")
            # Rename the script
            client.delete_script("script")

        **Delete A Script Placed By An Alternate Application**

        SmartSim supports delete operations on prefixed scripts sent to the ``Orchestrator`` by separate ``Models``.
        To do so, users must provide the ``Model`` name that stored the script to ``Client.set_data_source()``.
        This will instruct the ``Client`` to prepend the ``Model`` name input to all key searches.

        In the example, a ``Model`` named `model_1` has placed a ML model in a standalone ``Orchestrator`` with prefixing enabled
        on the ``Client``. The ``Orchestrator`` contents are:

        .. code-block:: bash

            1) "model_1.script"

        From within a separate ``Model`` named `model_2`, we perform a basic delete operation.
        To instruct the ``Client`` to prepend a ``Model`` name to all key searches, use the
        ``Client.set_data_source()`` function. Specify the ``Model`` name `model_1`
        that placed the list in the ``Orchestrator``:

        .. code-block:: python

            client.set_data_source("model_1")

        To delete the script in the ``Orchestrator``, we provide the script name:

        .. code-block:: python

            client.delete_model("script")