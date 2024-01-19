********
Ensemble
********
========
Overview
========
SmartSim ``Ensemble`` objects enable users to execute a group of computational tasks in an
``Experiment`` workflow. ``Ensembles`` are comprised of multiple ``Model`` objects,
each representing an individual simulation. An ``Ensemble`` can be be referenced as a single instance and
can be launched with other ``Models`` and ``Orchestrators`` to build AI-enabled workflows.
The :ref:`Ensemble API<ensemble_api>` offers key ``Ensemble`` features, including class methods to:

- :ref:`Load TF, TF-lite, PT, or ONNX models<ai_model_ensemble_doc>` into the ``Orchestrator`` at ``Ensemble`` runtime.
- :ref:`Load TorchScript functions<in_mem_TF_doc_ensemble>` to launch with each entity belonging to the ``Ensemble``.
- :ref:`Load TorchScripts<TS_from_file_ensemble>` to launch with each entity at runtime.
- :ref:`Prevent key overlapping<prefix_ensemble>` and enable ``Ensemble`` members to run the same code.
- :ref:`Attach configuration files<attach_files_ensemble>` to assign ``Ensemble`` member parameters or load parameters to file.

To initialize a SmartSim ``Ensemble``, use the ``Experiment.create_ensemble()`` API function.
When creating an ``Ensemble``, a user must consider one of the **three** ``Ensemble`` creation strategies:

1. :ref:`Parameter expansion<param_expansion_init>`: Generate a variable-sized set of unique simulation instances
   configured with user-defined input parameters.
2. :ref:`Replica creation<replicas_init>`: Generate a specified number of copies or instances of a simulation.
3. :ref:`Manually<append_init>`: Attach pre-configured ``Models`` to an ``Ensemble`` to manage as a single unit.

SmartSim manages ``Ensemble`` instances through the :ref:`Experiment API<experiment_api>` by providing functions to
launch, monitor, and stop applications.

==============
Initialization
==============
--------
Overview
--------
The :ref:`Experiment API<experiment_api>` is responsible for initializing all workflow entities.
A ``Ensemble`` is created using the ``Experiment.create_ensemble()`` factory method, and users can customize the
``Ensembles`` creation via the factory method parameters.

The key initializer arguments of ``Experiment.create_ensemble()`` are:

-  `name` (str): Specify the name of the ensemble, aiding in its unique identification.
-  `params` (dict[str, Any]): Provides a dictionary of parameters:values for expanding into the ``Model`` members within the ensemble. Enables parameter expansion for diverse scenario exploration.
-  `params_as_args` (list[str]): Specify which parameters from the `params` dictionary should be treated as command line arguments when executing the Models.
-  `batch_settings` (BatchSettings, optional): Describes settings for batch workload treatment.
-  `run_settings` (RunSettings, optional): Describes execution settings for individual Model members.
-  `replicas` (int, optional): Declare the number of ``Model`` clones within the ensemble, crucial for the creation of simulation replicas.
-  `perm_strategy` (str): Specifies a strategy for parameter expansion into ``Model`` instances, influencing the method of ensemble creation and number of ensemble members. The options are `"all_perm"`, `"step"`, and `"random"`.

By using certain combinations of initializer arguments from above, users may customize
the creation of an ``Ensemble`` to match one of the following creation strategies:

- :ref:`Parameter expansion<param_expansion_init>`
- :ref:`Manually Append<replicas_init>`
- :ref:`Replicas<append_init>`

.. _param_expansion_init:
-------------------
Parameter Expansion
-------------------
In ``Ensemble`` simulations, parameter expansion is a technique that
allows users to set parameter values using the `params` key initializer argument
in ``Experiment.create_ensemble()``. User's may control how the parameter values
spread across the ``Ensemble`` members by using the `perm_strategy` key initializer argument.
The `perm_strategy` argument accepts three values listed below.

**Parameter Expansion Strategy Options:**

-  `"all_perm"`: Generates all possible parameter permutations for an exhaustive exploration.
-  `"step"`: Creates sets for each element in n arrays, providing a systematic exploration.
-  `"random"`: Allows random selection from predefined parameter spaces, offering a stochastic approach.

Examples
--------
We provide two parameter expansion examples by using the `params` and `perm_strategy`
initializer arguments when creating an ``Ensemble``.

Example 1 : Parameter Expansion with ``RunSettings``, `params` and `perm_strategy`
    This example expands the same run settings and parameters to ``Ensemble`` members.
    To do so, we specify the parameter expansion strategy via the `perm_strategy` initializer
    argument.

    Begin by initializing a ``RunSettings`` object to expand to
    all ``Models``:

    .. code-block:: python

        rs = exp.create_run_settings(exe="python", exe_args="path/to/application_script.py")

    Next, define the parameters to expand to all ``Ensemble`` members:

    .. code-block:: python

        params = {
            "name": ["Ellie", "John"],
            "parameter": [2, 11]
        }

    Finally, initialize an ``Ensemble`` by passing in the ``RunSettings``, `params` and `perm_strategy`:

    .. code-block:: python

        ensemble = exp.create_ensemble("ensemble", params=params, run_settings=rs, perm_strategy="all_perm")

    By specifying `perm_strategy="all_perm"`, all permutations of the `params` key values will
    be calculated and distributed across ``Ensemble`` members. Here there are four permutations of the `params` values. Therefore,
    SmartSim will create four ``Model`` ``Ensemble`` members and assign a permutation set to each.

Example 2 : Parameter Expansion with ``RunSettings``, ``BatchSettings``, `params` and `perm_strategy`
    Submit the ``Ensemble`` as a batch job.
    Expand identical run settings and parameters to ``Ensemble`` members.
    Declare the parameter expansion strategy via `perm_strategy`.

    Begin by initializing and configuring a ``BatchSettings`` object to
    run the ``Ensemble`` instance:

    .. code-block:: python

        batch_args = {
            "distribution": "block"
            "exclusive": None
        }
        bs = exp.create_batch_settings(nodes=2,
                               time="10:00:00",
                               batch_args=batch_args)

    The above ``BatchSettings`` object will instruct SmartSim to run the ``Ensemble`` on two
    nodes with a timeout of `10 hours`.

    Next initialize a ``RunSettings`` object to expand to all ``Ensemble`` members:

    .. code-block:: python

        rs = exp.create_run_settings(exe="python", exe_args="path/to/application_script.py")
        rs.set_nodes(1)

    Next, define the parameters to include in ``Ensemble`` members:

    .. code-block:: python

        params = {
            "name": ["Ellie", "John"],
            "parameter": [2, 11]
        }

    Finally, initialize an ``Ensemble`` by passing in the ``RunSettings``, `params` and `perm_strategy`:

    .. code-block:: python

        ensemble = exp.create_ensemble("ensemble", params=params, run_settings=rs, batch_settings=bs, perm_strategy="step")

    By specifying `perm_strategy="step"`, the values of the `params` key will be
    grouped into intervals and distributed across ``Ensemble`` members. Here there are two groups. Therefore,
    the Ensemble will have two ``Model`` members each assigned a group.

.. _replicas_init:
--------
Replicas
--------
In ``Ensemble`` simulations, a replica strategy involves the creation of
identical or closely related ``Models`` within an ``Ensemble``, allowing for the
assessment of how a system responds to the same set of parameters under
multiple instances. Users may use the `replicas` initializer argument
to create a specified number of identical ``Model`` members.

Examples
--------
We demonstrate two examples for initializing an ``Ensemble`` using the replicas
strategy.

Example 1 : Replica Creation with ``RunSettings`` and `replicas`
    To create an ``Ensemble`` of identical ``Models``, begin by initializing a ``RunSettings``
    object:

    .. code-block:: python

        rs = exp.create_run_settings(exe="python", exe_args="path/to/application_script.py")

    Initialize the ``Ensemble`` by specifying the ``RunSettings`` object and number of clones to `replicas`:

    .. code-block:: python

        ensemble = exp.create_ensemble("ensemble-replica",
                               replicas=4,
                               run_settings=rs)

    By passing in `replicas=4`, four identical ``Ensemble`` members will be initialized.

Example 2 : Replica Creation with ``RunSettings``, ``BatchSettings`` and `replicas`
    To launch the ``Ensemble`` of identical ``Models`` as a batch job, begin by initializing a ``BatchSettings``
    object:

    .. code-block:: python

        batch_args = {
            "distribution": "block"
            "exclusive": None
        }
        bs = exp.create_batch_settings(nodes=4,
                               time="10:00:00",
                               batch_args=batch_args)

    The above ``BatchSettings`` object will instruct SmartSim to run the ``Ensemble`` on four
    nodes with a timeout of `10 hours`.

    Next, create a ``RunSettings`` object to expand to all ``Model`` replicas:

    .. code-block:: python

        rs = exp.create_run_settings(exe="python", exe_args="path/to/application_script.py")
        rs.set_nodes(4)

    Initialize the ``Ensemble`` by specifying the ``RunSettings`` object and number of clones to `replicas`:

    .. code-block:: python

        ensemble = exp.create_ensemble("ensemble-replica",
                               replicas=4,
                               run_settings=rs)

    By passing in `replicas=4`, four identical ``Ensemble`` members will be initialized.

.. _append_init:
---------------
Manually Append
---------------
Manually appending ``Models`` involves the addition of user created ``Model`` instance to an ``Ensemble``,
offering an in-depth level of customization in ``Ensemble`` design. This approach is favorable when users
have distinct requirements for individual ``Models``, such as variations in parameters, run settings,
or ``Model`` architectures.

Examples
--------
We provide an example for initializing an ``Ensemble`` and manually appending ``Models`` to
the ``Ensemble``.

Example 1 : Append ``Models`` with ``BatchSettings``
    To create an empty ``Ensemble`` to append ``Models``, initialize the ``Ensemble`` with
    a batch settings object:

    .. code-block:: python

        bs = exp.create_batch_settings(nodes=10,
                               time="01:00:00")
        ensemble = exp.create_ensemble("ensemble-append", batch_settings=bs)

    Next, create the ``Models`` to append to the ``Ensemble``:

    .. code-block:: python

        srun_settings_1 = exp.create_run_settings(exe=exe, exe_args="path/to/application_script_1.py")
        srun_settings_2 = exp.create_run_settings(exe=exe, exe_args="path/to/application_script_2.py")
        model_1 = exp.create_model(name="model_1", run_settings=srun_settings_1)
        model_2 = exp.create_model(name="model_2", run_settings=srun_settings_2)

    Finally, append the ``Model`` object to the ``Ensemble``:

    .. code-block:: python

        ensemble.add_model(model_1)
        ensemble.add_model(model_2)

    The new ``Ensemble`` is comprised of two appended ``Model`` members.

.. _attach_files_ensemble:
=====
Files
=====
Overview
========
``Ensembles`` often depend on external files (e.g. training datasets, evaluation datasets, etc)
to operate as intended. Users can instruct SmartSim to copy, symlink, or manipulate external files
prior to the ``Ensembles`` launch via the ``Ensemble.attach_generator_files()`` function.

.. note::
    Multiple calls to ``Ensemble.attach_generator_files()`` will overwrite previous file configurations
    in the ``Ensemble``.

To attach a file to a ``Ensemble`` for use at runtime, provide one of the following arguments to the
``Ensemble.attach_generator_files()`` function:

* `to_copy` (t.Optional[t.List[str]] = None): Files that are copied into the path of each ``Ensemble`` member.
* `to_symlink` (t.Optional[t.List[str]] = None): Files that are symlinked into the path each ``Ensemble`` member.

To specify a template file in order to programmatically replace specified parameters during generation
of the ``Ensemble`` member directory, pass the following value to the ``Ensemble.attach_generator_files()`` function:

* `to_configure` (t.Optional[t.List[str]] = None): Designed for text-based ``Ensemble`` input files,
  `"to_configure"` is exclusive to the ``Ensemble``. During ``Ensemble`` member directory generation, the attached
  files are parsed and specified tagged parameters are replaced with the `params` values that were
  specified in the ``Experiment.create_ensemble()`` factory method of the ``Ensemble``. The default tag is a semicolon
  (e.g., THERMO = ;THERMO;).

In the :ref:`Example<files_ensemble_example_doc>` subsection, we provide an example using the value `to_configure`
within ``Ensemble.attach_generator_files()``.

.. _files_ensemble_example_doc:
Example
=======
This example demonstrates how to attach a file to an ``Ensemble`` for parameter replacement at the time
of ``Ensemble`` member directory generation. This is accomplished using the `params` function parameter in
the ``Experiment.create_ensemble()`` factory function and the `to_configure` function input argument
in ``Ensemble.attach_generator_files()``.

In this example, we have a text file named `params_inputs.txt`. Within the text, is the parameter `THERMO`
that is required by each application at runtime:

.. code-block:: txt

   THERMO = ;THERMO;

In order to have the tagged parameter `;THERMO;` replaced with a usable value at runtime, two steps are required:

1. The `THERMO` variable must be included in ``Experiment.create_ensemble()`` factory method as
   part of the `params` parameter.
2. The file containing the tagged parameter `;THERMO;`, `params_inputs.txt`, must be attached to the ``Ensemble``
   via the ``Ensemble.attach_generator_files()`` method as part of the `to_configure` parameter.

To encapsulate our application within a ``Ensemble``, we must create an ``Experiment`` instance
to gain access to the ``Experiment`` factory method that creates the ``Ensemble``.
Begin by importing the ``Experiment`` module, importing SmartSim `log` module and initializing
an ``Experiment``:

.. code-block:: python

    from smartsim import Experiment
    from smartsim.log import get_logger

    logger = get_logger("Experiment Log")
    # Initialize the Experiment
    exp = Experiment("getting-started", launcher="auto")

We will create an ``Ensemble`` using the `replicas` strategy. This will require run settings to expand
to each ``Ensemble`` member. Create a simple ``RunSettings`` object to specify the path to
our application script as an executable argument and the executable to run the script:

.. code-block:: python

    # Initialize a RunSettings object
    ensemble_settings = exp.create_run_settings(exe="python", exe_args="/path/to/application.py")

Next, initialize a ``Ensemble`` object with ``Experiment.create_ensemble()``
and pass in the `ensemble_settings` instance:

.. code-block:: python

    # Initialize a Ensemble object
    example_ensemble = exp.create_ensemble("ensemble", ensemble_settings, params={"THERMO":1}, replicas=2)

We now have a ``Ensemble`` instance named `example_ensemble`. Attach the above text file
to the ``Ensemble`` for use at entity runtime. To do so, we use the
``Ensemble.attach_generator_files()`` function and specify the `to_configure`
parameter with the path to the text file, `params_inputs.txt`:

.. code-block:: python

    # Attach the file to the Ensemble instance
    example_ensemble.attach_generator_files(to_configure="path/to/params_inputs.txt")

To created an isolated directory for the ``Ensemble`` outputs and configuration files, invoke ``Experiment.generate()`` via the
``Experiment`` instance `exp` with `example_ensemble` as an input parameter:

.. code-block:: python

    exp.generate(example_ensemble)

Launching the ``Ensemble`` with ``exp.start(example_ensemble)`` processes attached generator files. `configure_inputs.txt` will be
available in each ``Ensemble`` member working directory and SmartSim will assign the `example_ensemble` `params` to each text file.

The contents of `params_inputs.txt` in each folder after ``Ensemble`` completion are:

.. code-block:: bash

   THERMO = 1

=====================
ML Models and Scripts
=====================
--------
Overview
--------
SmartSim users have the capability to utilize ML runtimes within a ``Ensemble``.
Functions accessible through an ``Ensemble`` object support loading ML models (TensorFlow, TensorFlow-lite,
PyTorch, and ONNX) and TorchScripts into standalone ``Orchestrators`` or colocated ``Orchestrators`` at
application runtime.

Users can follow **two** processes to load a ML model to the ``Orchestrator``:

- :ref:`from memory<in_mem_ML_ensemble_ex>`
- :ref:`from file<from_file_ML_ensemble_ex>`

Users can follow **three** processes to load a TorchScript to the ``Orchestrator``:

- :ref:`from memory<in_mem_TF_doc_ensemble>`
- :ref:`from file<TS_from_file_ensemble>`
- :ref:`from string<TS_raw_string_ensemble>`

Once a ML model or TorchScript is loaded to the ``Orchestrator``, ``Ensemble`` ``Model`` objects can
leverage ML capabilities by utilizing the SmartSim client (:ref:`SmartRedis<dead_link>`)
to execute the stored ML models or TorchScripts.

.. _ai_model_ensemble_doc:
---------
AI Models
---------
When configuring a ``Ensemble``, users can instruct SmartSim to load
Machine Learning (ML) models dynamically to the ``Orchestrator`` (colocated or standard). ML models added
are loaded into the ``Orchestrator`` prior to the execution of every ``Ensemble`` member. To load an ML model
to the orchestrator, SmartSim users can provide the ML model **in-memory** or specify the **file path**
when using the ``Ensemble.add_ml_model()`` function. The supported ML frameworks are TensorFlow,
TensorFlow-lite, PyTorch, and ONNX.

When attaching an ML model using ``Ensemble.add_ml_model()``, the
following arguments are offered to customize the storage and execution of the ML model:

- `name` (str): name to reference the model in the Orchestrator.
- `model` (t.Optional[str] = None): A model in memory (only supported for non-colocated orchestrators).
- `model_path` (t.Optional[str] = None): serialized model.
- `backend` (str): name of the backend (TORCH, TF, TFLITE, ONNX).
- `device` (t.Literal["CPU", "GPU"] = "CPU"): name of device for execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `batch_size` (int = 0): batch size for execution, defaults to 0.
- `min_batch_size` (int = 0): minimum batch size for model execution, defaults to 0.
- `min_batch_timeout` (int = 0): time to wait for minimum batch size, defaults to 0.
- `tag` (str = ""): additional tag for model information, defaults to “”.
- `inputs` (t.Optional[t.List[str]] = None): model inputs (TF only), defaults to None.
- `outputs` (t.Optional[t.List[str]] = None): model outputs (TF only), defaults to None.

.. _in_mem_ML_ensemble_ex:
Example: Attach an in-memory ML Model
-------------------------------------
This example demonstrates how to attach an in-memory ML model to a SmartSim ``Ensemble``
to load into an ``Orchestrator`` at ``Ensemble`` runtime.

.. note::
    This example assumes:

    - an ``Orchestrator`` is launched prior to the ``Ensembles`` execution
    - an initialized ``Ensemble`` named `smartsim_ensemble` exists within the ``Experiment`` workflow

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

**Attach the ML model to a SmartSim Ensemble**

Assuming an initialized ``Ensemble`` named `smartsim_ensemble` exists, we add the in-memory TensorFlow model using
the ``Ensemble.add_ml_model()`` function and specify the in-memory model to the parameter `model`:

.. code-block:: python

    smartsim_ensemble.add_ml_model(name="cnn", backend="TF", model=model, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)

In the above ``smartsim_ensemble.add_ml_model()`` code snippet, we offer the following arguments:

-  `name` ("cnn"): A name to reference the model in the Orchestrator.
-  `backend` ("TF): Indicating that the model is a TensorFlow model.
-  `model` (model): The in-memory representation of the TensorFlow model.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the ML model to a non-existent database.

When the ``Ensemble`` is started via ``Experiment.start()``, the ML model will be loaded to the
launched ``Orchestrator``. The ML model can then be executed on the ``Orchestrator`` via a SmartSim
client (:ref:`SmartRedis<dead_link>`).

.. _from_file_ML_ensemble_ex:
Example: Attaching an ML Model from file
----------------------------------------
This example demonstrates how to attach a ML model from file to a SmartSim ``Ensemble``
to load into an ``Orchestrator`` at ``Ensemble`` runtime.

.. note::
    This example assumes:

    - a standard ``Orchestrator`` is launched prior to the ``Ensembles`` execution
    - an initialized ``Ensemble`` named `smartsim_ensemble` exists within the ``Experiment`` workflow

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

Assuming an initialized ``Ensemble`` named `smartsim_ensemble` exists, we add a TensorFlow model using
the ``Ensemble.add_ml_model()`` function and specify the model path to the parameter `model_path`:

.. code-block:: python

    smartsim_ensemble.add_ml_model(name="cnn", backend="TF", model_path=model_file, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)

In the above ``smartsim_ensemble.add_ml_model()`` code snippet, we offer the following arguments:

-  `name` ("cnn"): A name to reference the model in the Orchestrator.
-  `backend` ("TF): Indicating that the model is a TensorFlow model.
-  `model_path` (model_file): The path to the ML model script.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the ML model to a non-existent database.

When the ``Ensemble`` is started via ``Experiment.start()``, the ML model will be loaded to the
launched ``Orchestrator``. The ML model can then be executed on the ``Orchestrator`` via a SmartSim
client (:ref:`SmartRedis<dead_link>`) within the application code.

.. _TS_ensemble_doc:
------------
TorchScripts
------------
When configuring a ``Ensemble``, users can instruct SmartSim to load TorchScripts dynamically
to the ``Orchestrator``. TorchScripts added are loaded into the ``Orchestrator`` prior to
the execution of the ``Ensemble``. To load a TorchScript to the database, SmartSim users
can follow one of the processes:

- :ref:`Define a TorchScript function in-memory<in_mem_TF_doc_ensemble>`
   Use the ``Ensemble.add_function()`` to instruct SmartSim to load an in-memory TorchScript to the ``Orchestrator``.
- :ref:`Define a TorchScript function from file<_ensemble>`
   Provide file path to ``Ensemble.add_script()`` to instruct SmartSim to load the TorchScript from file to the ``Orchestrator``.
- :ref:`Define a TorchScript function as string<TS_raw_string_ensemble>`
   Provide function string to ``Ensemble.add_script()`` to instruct SmartSim to load a raw string as a TorchScript function to the ``Orchestrator``.

Continue or select the respective process link to learn more on how each function (``Ensemble.add_script()`` and ``Ensemble.add_function()``)
dynamically loads TorchScripts to the ``Orchestrator``.

.. _in_mem_TF_doc_ensemble:
Attach an in-memory TorchScript
-------------------------------
Users can define TorchScript functions within the Python driver script
to attach to a ``Ensemble``. This feature is supported by ``Ensemble.add_function()`` which provides flexible
device selection, allowing users to choose between which device the the TorchScript is executed on, `"GPU"` or `"CPU"`.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

.. warning::
    ``Ensemble.add_function()`` does **not** support loading in-memory TorchScript functions to a colocated ``Orchestrator``.
    If you would like to load a TorchScript function to a colocated ``Orchestrator``, define the function
    as a :ref:`raw string<TS_raw_string_ensemble>` or :ref:`load from file<TS_from_file_ensemble>`.

When specifying an in-memory TF function using ``Ensemble.add_function()``, the
following arguments are offered:

- `name` (str): reference name for the script inside of the ``Orchestrator``.
- `function` (t.Optional[str] = None): TorchScript function code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

.. _in_mem_TF_ex:
Example: Loading a in-memory TorchScript function
'''''''''''''''''''''''''''''''''''''''''''''''''
This example walks through the steps of instructing SmartSim to load an in-memory TorchScript function
to a standard ``Orchestrator``.

.. note::
    The example assumes:

    - a standard ``Orchestrator`` is launched prior to the ``Ensembles`` execution
    - an initialized ``Ensemble`` named `smartsim_ensemble` exists within the ``Experiment`` workflow

**Define an in-memory TF function**

To begin, define an in-memory TorchScript function within the Python driver script.
For the purpose of the example, we add a simple TorchScript function, `timestwo`:

.. code-block:: python

    def timestwo(x):
        return 2*x

**Attach the in-memory TorchScript function to a SmartSim Ensemble**

We use the ``Ensemble.add_function()`` function to instruct SmartSim to load the TorchScript function `timestwo`
onto the launched standard ``Orchestrator``. Specify the function `timestwo` to the `function`
parameter:

.. code-block:: python

    smartsim_ensemble.add_function(name="example_func", function=timestwo, device="GPU", devices_per_node=2, first_device=0)

In the above ``smartsim_ensemble.add_function()`` code snippet, we offer the following arguments:

-  `name` ("example_func"): A name to uniquely identify the model within the database.
-  `function` (timestwo): Name of the TorchScript function defined in the Python driver script.
-  `device` ("CPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the TorchScript to a non-existent database.

When the ``Ensemble`` is started via ``Experiment.start()``, the TF function will be loaded to the
standard ``Orchestrator``. The function can then be executed on the ``Orchestrator`` via a SmartSim
client (:ref:`SmartRedis<dead_link>`) within the application code.

.. _TS_from_file_ensemble:
Attach a TorchScript from file
------------------------------
Users can attach TorchScript functions from a file to a ``Ensemble`` and upload them to a
colocated or standard ``Orchestrator``. This functionality is supported by the ``Ensemble.add_script()``
function's `script_path` parameter. The function supports
flexible device selection, allowing users to choose between `"GPU"` or `"CPU"` via the `device` parameter.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

When specifying a TorchScript using ``Ensemble.add_script()``, the
following arguments are offered:

- `name` (str): Reference name for the script inside of the ``Orchestrator``.
- `script` (t.Optional[str] = None): TorchScript code (only supported for non-colocated orchestrators).
- `script_path` (t.Optional[str] = None): path to TorchScript code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

.. _TS_from_file_ex:
Example: Loading a TorchScript from File
''''''''''''''''''''''''''''''''''''''''
This example walks through the steps of instructing SmartSim to load a TorchScript from file
to a ``Orchestrator``.

.. note::
    This example assumes:

    - a ``Orchestrator`` is launched prior to the ``Ensembles`` execution
    - an initialized ``Ensemble`` named `smartsim_ensemble` exists within the ``Experiment`` workflow

**Define a TorchScript script**

For the example, we create the Python script `torchscript.py`. The file contains a
simple torch function shown below:

.. code-block:: python

    def negate(x):
        return torch.neg(x)

**Attach the TorchScript script to a SmartSim Ensemble**

Assuming an initialized ``Ensemble`` named `smartsim_ensemble` exists, we add a TorchScript script using
the ``Ensemble.add_script()`` function and specify the script path to the parameter `script_path`:

.. code-block:: python

    smartsim_ensemble.add_script(name="example_script", script_path="path/to/torchscript.py", device="GPU", devices_per_node=2, first_device=0)

In the above ``smartsim_ensemble.add_script()`` code snippet, we offer the following arguments:

-  `name` ("example_script"): Reference name for the script inside of the ``Orchestrator``.
-  `script_path` ("path/to/torchscript.py"): Path to the script file.
-  `device` ("CPU"): device for script execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the ML model to a non-existent database.

When `smartsim_ensemble` is started via ``Experiment.start()``, the TorchScript will be loaded from file to the
orchestrator that is launched prior to the start of the `smartsim_ensemble`.

.. _TS_raw_string_ensemble:
Define TorchScripts as raw string
---------------------------------
Users can upload TorchScript functions from string to send to a colocated or
standard ``Orchestrator``. This feature is supported by the
``Ensemble.add_script()`` function's `script` parameter. The function supports
flexible device selection, allowing users to choose between `"GPU"` or `"CPU"` via the `device` parameter.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

When specifying a TorchScript using ``Ensemble.add_script()``, the
following arguments are offered:

- `name` (str): Reference name for the script inside of the ``Orchestrator``.
- `script` (t.Optional[str] = None): TorchScript code (only supported for non-colocated orchestrators).
- `script_path` (t.Optional[str] = None): path to TorchScript code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

.. _TS_from_file_ex:
Example: Loading a TorchScript from string
''''''''''''''''''''''''''''''''''''''''''
This example walks through the steps of instructing SmartSim to load a TorchScript function
from string to a ``Orchestrator`` before the execution of the associated ``Ensemble``.

.. note::
    This example assumes:

    - a ``Orchestrator`` is launched prior to the ``Ensembles`` execution
    - an initialized ``Ensemble`` named `smartsim_ensemble` exists within the ``Experiment`` workflow

**Define a string TorchScript**

Define the TorchScript code as a variable in the Python driver script:

.. code-block:: python

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

**Attach the TorchScript function to a SmartSim Ensemble**

Assuming an initialized ``Ensemble`` named `smartsim_ensemble` exists, we add a TensorFlow model using
the ``Ensemble.add_script()`` function and specify the variable `torch_script_str` to the parameter
`script`:

.. code-block:: python

    smartsim_ensemble.add_script(name="example_script", script=torch_script_str, device="GPU", devices_per_node=2, first_device=0)

In the above ``smartsim_ensemble.add_script()`` code snippet, we offer the following arguments:

-  `name` ("example_script"): key to store script under.
-  `script` (torch_script_str): TorchScript code.
-  `device` ("CPU"): device for script execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an orchestrator will result in
    a failed attempt to load the TorchScript to a non-existent database.

When the ``Ensemble`` is started via ``Experiment.start()``, the TorchScript will be loaded to the
orchestrator that is launched prior to the start of the ``Ensemble``.

.. _prefix_ensemble:
=========================
Data Collision Prevention
=========================
--------
Overview
--------
When multiple ``Ensemble`` members use the same code to access their respective data
in the ``Orchestrator``, key overlapping can occur, leading to inadvertent data access
between ``Ensemble`` members. To address this, the SmartSim ``Ensembles`` supports key prefixing
via the ``Ensemble.enable_key_prefixing()`` function,
which automatically adds the ``Model`` `name` as a prefix to the keys sent to the ``Orchestrator``.
Enabling key prefixing eliminates issues related to key overlapping, allowing ``Ensemble``
members to use the same code without issue.

-------------------------------
Example: Ensemble Key Prefixing
-------------------------------
In this example, we create an ``Ensemble`` of comprised of two ``Models`` that use identical code
to send data to a standard ``orchestrator``. To prevent key collisions and ensure data
integrity, we enable key prefixing in the ``Ensemble`` which automatically
appends the ``Model`` `name` when sending and accessing data on the ``Orchestrator``. After the
producer ``Ensemble`` completes, we launch a consumer ``Model`` within the Python driver script
that accesses the produced data on the ``Orchestrator``.

The Application Producer Script
-------------------------------
In the Python driver script, we instruct SmartSim to create an ``Ensemble`` comprised of
two ``Models`` that execute the same producer executable code.
In the producer application script, a SmartRedis ``Client`` sends a
tensor to the ``Orchestrator``. Since both ``Ensemble`` ``Models`` use the same producer script script,
two identical tensors are placed on the ``Orchestrator`` which could cause a key collision.
To prevent a key collision, we enable key prefixing on the ``Ensemble`` in the driver script.
This means that when a producer ``Model`` places a tensor on the ``Orchestrator``, it will prepend
the ``Model`` `name` to the tensor `name`. This migth look like: `"model_1.tensor_name"`.

Below is the simulation code for the two producer ``Models`` within the ``Ensemble``:

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

After the completion of the ``Ensemble``, the contents of the ``Orchestrator`` are:

.. code-block:: bash

    1) "producer_0.tensor"
    2) "producer_1.tensor"

The Application Consumer Script
-------------------------------
In the Python driver script, we initialize a consumer ``Model`` that requests
the tensors produced from the ``Ensemble``. To do so, we use SmartRedis
key prefixing functionality to instruct the SmartRedis ``Client`` to append
the name of a ``Model`` to the searched `name`.

First specify the imports and initialize a SmartRedis ``Client``:

.. code-block:: python

    from smartredis import Client, log_data
    from smartredis import *

    # Initialize a Client
    client = Client(cluster=False)

Retrieve the tensor from the first producer ``Model`` in the ``Ensemble``. Use the
``Client.set_data_source()`` function to append the ``Ensemble`` member `name`, `producer_0`, to the
key being searched. When ``Client.poll_tensor()`` is executed, the `client` will poll for key, `producer_0.tensor`:

.. code-block:: python

    client.set_data_source("producer_0")
    val1 = client.poll_tensor("tensor", 100, 100)

Follow the same instructions above, however, change the prefixed `name` to the `name`
of the second producer ``Model`` (`producer_1`):

.. code-block:: python

    client.set_data_source("producer_1")
    val2 = client.poll_tensor("tensor", 100, 100)

We print the boolean return to verify that the tensors were found:

.. code-block:: python

    client.log_data(LLInfo, f"producer_0.tensor was found: {val1}")
    client.log_data(LLInfo, f"producer_1.tensor was found: {val2}")

When the ``Experiment`` driver script is executed, the following output will appear in `consumer.out`::
    Default@11-46-05:producer_0.tensor was found: True
    Default@11-46-05:producer_1.tensor was found: True

The Experiment Script
---------------------
In the ``Experiment`` driver script we initialize a standalone ``Orchestrator``,
initialize an ``Ensemble`` using the replicas creation strategy and we initialize
a consumer ``Model`` that will access the data from the ``Ensemble``. It is essential
to launch the ``Orchestrator`` before any other SmartSim entity since each simulation
launch connects to a launched ``Orchestrator``. We then launch the ``Ensemble``, then
launch the consumer ``Model``.

To setup for the example in the Python driver script, we

-  initialize the Experiment `exp`
-  initialize the standard orchestrator `single_shard_db`

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

We are now setup to discuss key prefixing within the ``Experiment`` driver script.
Since we are create an ``Ensemble`` using the replicas strategy, initialize a ``RunSettings``
object to expand to all ``Ensemble`` members. Specify the path to the application
producer script:

.. code-block:: python

    # Initialize a RunSettings object
    ensemble_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/producer_script.py")

Next, initialize an ``Ensemble`` by specifying `ensemble_settings` and the number of replicas to create:

.. code-block:: python

    producer_ensemble = exp.create_ensemble("producer", run_settings=ensemble_settings, replicas=2)

Instruct SmartSim to prefix all tensors sent to the ``Orchestrator`` via ``Ensemble.enable_key_prefixing()``:

.. code-block:: python

    producer_ensemble.enable_key_prefixing()

Next, initialize the consumer ``Model`` that requests the prefixed tensors produced by the ``Ensemble``:

.. code-block:: python

    # Initialize a RunSettings object
    model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/consumer_script.py")
    consumer_model = exp.create_model("consumer", model_settings)

Next, organize the SmartSIm entity output files into a single ``Experiment`` folder:

.. code-block:: python

    exp.generate(single_shard_db, producer_ensemble, consumer_model, overwrite=True)

Launch the ``Orchestrator``:

.. code-block:: python

    exp.start(single_shard_db, summary=True)

Launch the ``Ensemble``:

.. code-block:: python

    exp.start(producer_ensemble, block=True, summary=True)

Set `block=True` so that ``Experiment.start()`` waits until the last ``Model`` has finished before continuing.

Register the ``Models`` that will be accessed within the consumer script:

.. code-block:: python

    for model in producer_ensemble:
        consumer_model.register_incoming_entity(model)

Launch the consumer ``Model``:

.. code-block:: python

    exp.start(consumer_model, block=True, summary=True)

Tear down the ``Orchestrator``:

.. code-block:: python

    exp.stop(single_shard_db)
    logger.info(exp.summary())