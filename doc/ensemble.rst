.. _ensemble_doc:
********
Ensemble
********
========
Overview
========
A SmartSim ``Ensemble`` enables users to run a **group** of computational tasks together in an
``Experiment`` workflow. An ``Ensemble`` is comprised of multiple ``Model`` objects,
where each ``Ensemble`` member (SmartSim ``Model``) represents an individual application.
An ``Ensemble`` can be managed as a single entity and
launched with other :ref:`Model<model_object_doc>` and :ref:`Orchestrators<orch_docs:>` to construct AI-enabled workflows.

The :ref:`Ensemble API<ensemble_api>` offers key features, including methods to:

- :ref:`Attach configuration files<attach_files_ensemble>` for use at ``Ensemble`` runtime.
- :ref:`Load AI models<ai_model_ensemble_doc>` (TF, TF-lite, PT, or ONNX) into the ``Orchestrator`` at ``Ensemble`` runtime.
- :ref:`Load TorchScripts<TS_ensemble_doc>` into the ``Orchestrator`` at ``Ensemble`` runtime.
- :ref:`Prevent data collisions<prefix_ensemble>` within the ``Ensemble``, which allows for reuse of application code.

To create a SmartSim ``Ensemble``, use the ``Experiment.create_ensemble()`` API function.
When creating an ``Ensemble``, it is important to consider one of the **three** ``Ensemble`` creation strategies:

1. :ref:`Parameter expansion<param_expansion_init>`: Generate a variable-sized set of unique simulation instances
   configured with user-defined input parameters.
2. :ref:`Replica creation<replicas_init>`: Generate a specified number of copies or instances of a simulation.
3. :ref:`Manually<append_init>`: Attach pre-configured ``Models`` to an ``Ensemble`` to manage as a single unit.

SmartSim manages ``Ensemble`` instances through the :ref:`Experiment API<experiment_api>` by providing functions to
launch, monitor, and stop applications.

==============
Initialization
==============
Overview
========
The :ref:`Experiment API<experiment_api>` is responsible for initializing all workflow entities.
An ``Ensemble`` is created using the ``Experiment.create_ensemble()`` factory method, and users can customize the
``Ensemble`` creation via the factory method parameters.

The factory method arguments of ``Experiment.create_ensemble()`` are:

-  `name` (str): Specify the name of the ``Ensemble``, aiding in its unique identification.
-  `params` (dict[str, Any]): Provides a dictionary of {parameters:values} for expanding into the ``Model`` members within the ``Ensemble``. Enables parameter expansion for diverse scenario exploration.
-  `params_as_args` (list[str]): Specify which parameters from the `params` dictionary should be treated as command line arguments when executing the ``Models``.
-  `batch_settings` (BatchSettings, optional): Describes settings for batch workload treatment.
-  `run_settings` (RunSettings, optional): Describes execution settings for individual ``Model`` members.
-  `replicas` (int, optional): Declare the number of ``Model`` clones within the ``Ensemble``, crucial for the creation of simulation replicas.
-  `perm_strategy` (str): Specifies a strategy for parameter expansion into ``Model`` instances, influencing the method of ``Ensemble`` creation and number of ``Ensemble`` members. The options are `"all_perm"`, `"step"`, and `"random"`.

By using specific combinations of the factory method arguments mentioned above, users can tailor
the creation of an ``Ensemble`` to align with one of the following creation strategies:

- :ref:`Parameter expansion<param_expansion_init>` allows for diverse scenario exploration by expanding a dictionary of parameters into the ``Ensemble`` ``Model`` members.
- :ref:`Manually Append<append_init>` allows users to attach pre-configured ``Models`` to an ``Ensemble`` to manage as a single unit.
- :ref:`Replicas<replicas_init>` enables the creation of a specified number of ``Model`` clones within the ``Ensemble``.

.. _param_expansion_init:
Parameter Expansion
===================
In ``Ensemble`` simulations, parameter expansion is a technique that
allows users to set parameter values per ``Ensemble`` member. This is done
by specifying input to the `params` and `perm_strategy` factory method arguments during ``Ensemble`` creation (``Experiment.create_ensemble()``).
User's may control how the `params` values are applied to the ``Ensemble`` through the `perm_strategy` argument.
The `perm_strategy` argument accepts three values listed below.

**Parameter Expansion Strategy Options:**

-  `"all_perm"`: Generate all possible parameter permutations for an exhaustive exploration. This
  means that every possible combination of parameters will be used in the ``Ensemble``.
-  `"step"`: Create sets for each element in n arrays, providing a systematic exploration. This
  means that the parameters will be changed in a step-by-step manner, allowing you to see the
  effect of each parameter individually.
-  `"random"`: Enable random selection from predefined parameter spaces, offering a stochastic approach.
  This means that the parameters will be chosen randomly for each ``Model``, which can be useful
  for exploring a wide range of possibilities.

--------
Examples
--------
We provide two parameter expansion examples by using the `params` and `perm_strategy`
factory method arguments to create an ``Ensemble``.

Example 1 : Parameter Expansion with ``RunSettings``, `params` and `perm_strategy`

    This example extends the same run settings with sampled grouped parameters to each ``Ensemble`` member.
    The ``Ensemble`` encompasses all grouped permutations of the specified `params` by creating a ``Model`` member for each
    permutation. To achieve this, we specify the parameter expansion strategy, `"all_perm"`, to the `perm_strategy`
    factory method argument.

    .. dropdown:: Example Driver Script source code

        .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/param_expansion_1.py

    Begin by initializing a ``RunSettings`` object to apply to
    all ``Ensemble`` members:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/param_expansion_1.py
        :language: python
        :linenos:
        :lines: 6-7

    Next, define the parameters that will be applied to the ``Ensemble``:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/param_expansion_1.py
        :language: python
        :linenos:
        :lines: 9-13

    Finally, initialize an ``Ensemble`` by specifying the ``RunSettings``, `params` and `perm_strategy="all_perm"`:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/param_expansion_1.py
        :language: python
        :linenos:
        :lines: 15-16

    By specifying `perm_strategy="all_perm"`, all permutations of the `params` will
    be calculated and distributed across ``Ensemble`` members. Here there are four permutations of the `params` values:

    .. code-block:: bash

        ensemble member 1: ["Ellie", 2]
        ensemble member 2: ["Ellie", 11]
        ensemble member 3: ["John", 2]
        ensemble member 4: ["John", 11]

    Therefore, SmartSim will create four ``Model`` ``Ensemble`` members and assign a permutation group to each
    ``Model``.

Example 2 : Parameter Expansion with ``RunSettings``, ``BatchSettings``, `params` and `perm_strategy`

    In this example, the ``Ensemble`` will be submitted as a batch job. An identical set of
    ``RunSettings`` will be applied to all ``Ensemble`` member and parameters will be
    applied in a `step` strategy to create the ``Ensemble`` members.

    .. dropdown:: Example Driver Script source code

        .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py

    Begin by initializing and configuring a ``BatchSettings`` object to
    run the ``Ensemble`` instance:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py
        :language: python
        :linenos:
        :lines: 6-8

    The above ``BatchSettings`` object will instruct SmartSim to run the ``Ensemble`` on two
    nodes with a timeout of `10 hours`.

    Next initialize a ``RunSettings`` object to apply to all ``Ensemble`` members:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py
        :language: python
        :linenos:
        :lines: 10-12

    Next, define the parameters to include in ``Ensemble``:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py
        :language: python
        :linenos:
        :lines: 14-18

    Finally, initialize an ``Ensemble`` by passing in the ``RunSettings``, `params` and `perm_strategy="step"`:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py
        :language: python
        :linenos:
        :lines: 20-21

    By specifying `perm_strategy="step"`, the values of the `params` key will be
    grouped into intervals and distributed across ``Ensemble`` members. Here, there are two groups:

    .. code-block:: bash

        ensemble member 1: ["Ellie", 2]
        ensemble member 2: ["John", 11]

    Therefore, the ``Ensemble`` will have two ``Model`` members each assigned a group.

.. _replicas_init:
Replicas
========
In ``Ensemble`` simulations, a replica strategy involves the creation of
identical ``Models`` within an ``Ensemble``. This strategy is particularly useful for
applications that have some inherent randomness. Users may use the `replicas` factory method argument
to create a specified number of identical ``Model`` members during ``Ensemble`` creation (``Experiment.create_ensemble()``).

--------
Examples
--------
We provide two examples for initializing an ``Ensemble`` using the replicas creation
strategy.

Example 1 : Replica Creation with ``RunSettings`` and `replicas`

    This example extends the same run settings to ``Ensemble`` member clones. To achieve this, we specify the number
    of clones to create via the `replicas` argument and pass a ``RunSettings`` object to the `run_settings`
    argument.

    .. dropdown:: Example Driver Script source code

        .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/replicas_1.py

    To create an ``Ensemble`` of identical ``Models``, begin by initializing a ``RunSettings``
    object:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/replicas_1.py
        :language: python
        :linenos:
        :lines: 6-7

    Initialize the ``Ensemble`` by specifying the ``RunSettings`` object and number of clones to `replicas`:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/replicas_1.py
        :language: python
        :linenos:
        :lines: 9-10

    By passing in `replicas=4`, four identical ``Ensemble`` members will be initialized.

Example 2 : Replica Creation with ``RunSettings``, ``BatchSettings`` and `replicas`

    This example extends the same run settings and batch settings to ``Ensemble`` member clones. To achieve this, we specify the number
    of clones to create via the `replicas` argument, passing a ``RunSettings`` object to the `run_settings`
    argument and passing a ``BatchSettings`` argument to `batch_settings`.

    .. dropdown:: Example Driver Script source code

        .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/replicas_2.py

    To launch the ``Ensemble`` of identical ``Models`` as a batch job, begin by initializing a ``BatchSettings``
    object:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/replicas_2.py
        :language: python
        :linenos:
        :lines: 6-9

    The above ``BatchSettings`` object will instruct SmartSim to run the ``Ensemble`` on four
    nodes with a timeout of `10 hours`.

    Next, create a ``RunSettings`` object to apply to all ``Model`` replicas:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/replicas_2.py
        :language: python
        :linenos:
        :lines: 10-12

    Initialize the ``Ensemble`` by specifying the ``RunSettings`` object, ``BatchSettings`` object
    and number of clones to `replicas`:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/replicas_2.py
        :language: python
        :linenos:
        :lines: 14-15

    By passing in `replicas=4`, four identical ``Ensemble`` members will be initialized.

.. _append_init:
Manually Append
===============
Manually appending ``Models`` to an ``Ensemble`` offers an in-depth level of customization in ``Ensemble`` design.
This approach is favorable when users have distinct requirements for individual ``Models``, such as variations
in parameters, run settings, or different types of simulations.

--------
Examples
--------
We provide an example for manually appending ``Models`` to an ``Ensemble``.

Example 1 : Append ``Models`` to launch as a batch job
    In this example, we append ``Models`` to an ``Ensemble`` for batch job execution. To do
    this, we first initialize an Ensemble with a ``BatchSettings`` object. Then, manually
    create ``Models`` and add each to the ``Ensemble`` using the ``Ensemble.add_model()`` function.

    .. dropdown:: Example Driver Script source code

        .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py

    To create an empty ``Ensemble`` to append ``Models``, initialize the ``Ensemble`` with
    a batch settings object:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
        :language: python
        :linenos:
        :lines: 6-11

    Next, create the ``Models`` to append to the ``Ensemble``:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
        :language: python
        :linenos:
        :lines: 13-20

    Finally, append the ``Model`` objects to the ``Ensemble``:

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
        :language: python
        :linenos:
        :lines: 22-25

    The new ``Ensemble`` is comprised of two appended ``Model`` members.

.. _attach_files_ensemble:
=====
Files
=====
Overview
========
``Ensemble`` members often depend on external files (e.g. training datasets, evaluation datasets, etc)
to operate as intended. Users can instruct SmartSim to copy, symlink, or manipulate external files
prior to an ``Ensemble`` launch via the ``Ensemble.attach_generator_files()`` function so that the
files are available for use by ``Models`` referenced by an ``Ensemble``.

.. note::
    Multiple calls to ``Ensemble.attach_generator_files()`` will overwrite previous file configurations
    on the ``Ensemble``.

To attach a file to an ``Ensemble`` for use at runtime, provide one of the following arguments to the
``Ensemble.attach_generator_files()`` function:

* `to_copy` (t.Optional[t.List[str]] = None): Files that are copied into the path of the ``Ensemble`` members.
* `to_symlink` (t.Optional[t.List[str]] = None): Files that are symlinked into the path of the ``Ensemble`` members.
  A symlink, or symbolic link, is a file that points to another file or directory, allowing you to access that file
  as if it were located in the same directory as the symlink.

To specify a template file in order to programmatically replace specified parameters during generation
of ``Ensemble`` member directories, pass the following value to the ``Ensemble.attach_generator_files()`` function:

* `to_configure` (t.Optional[t.List[str]] = None): This parameter is designed for text-based ``Ensemble``
  member input files. During directory generation for ``Ensemble`` members, the linked files are parsed and replaced with
  the `params` values applied to each ``Ensemble`` member. To further explain, the ``Ensemble``
  creation strategy is considered when replacing the tagged parameters in the input files.
  These tagged parameters are placeholders in the text that are replaced with the actual
  parameter values during the directory generation process. The default tag is a semicolon
  (e.g., THERMO = ;THERMO;).

In the :ref:`Example<files_example_doc_ensem>` subsection, we provide an example using the value `to_configure`
within ``Ensemble.attach_generator_files()``.

.. _files_example_doc_ensem:
Example
=======
This example demonstrates how to attach a text file to an ``Ensemble`` for parameter replacement.
This is accomplished using the `params` function parameter in
the ``Experiment.create_ensemble()`` factory function and the `to_configure` function parameter
in ``Ensemble.attach_generator_files()``.

.. dropdown:: Example Driver Script source code

    .. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/file_attach.py

In this example, we have a text file named `params_inputs.txt`. Within the text, is the parameter `THERMO`
that is required by each ``Ensemble`` member at runtime:

.. code-block:: txt

   THERMO = ;THERMO;

In order to have the tagged parameter `;THERMO;` replaced with a usable value at runtime, two steps are required:

1. The `THERMO` variable must be included in ``Experiment.create_ensemble()`` factory method as
   part of the `params` parameter.
2. The file containing the tagged parameter `;THERMO;`, `params_inputs.txt`, must be attached to the ``Ensemble``
   via the ``Ensemble.attach_generator_files()`` method as part of the `to_configure` parameter.

To encapsulate our application within an ``Ensemble``, we must create an ``Experiment`` instance
to gain access to the ``Experiment`` factory method that creates the ``Ensemble``.
Begin by importing the ``Experiment`` module and initializing
an ``Experiment``:

.. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
    :language: python
    :linenos:
    :lines: 1-4

To create our ``Ensemble``, we are using the `replicas` initialization strategy.
Begin by creating a simple ``RunSettings`` object to specify the path to
the executable simulation as an executable:

.. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
    :language: python
    :linenos:
    :lines: 6-7

Next, initialize an ``Ensemble`` object with ``Experiment.create_ensemble()``
and pass in the `ensemble_settings` instance and specify `replicas=2`:

.. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
    :language: python
    :linenos:
    :lines: 9-10

We now have an ``Ensemble`` instance named `example_ensemble`. Attach the above text file
to the ``Ensemble`` for use at entity runtime. To do so, we use the
``Ensemble.attach_generator_files()`` function and specify the `to_configure`
parameter with the path to the text file, `params_inputs.txt`:

.. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
    :language: python
    :linenos:
    :lines: 12-13

To created an isolated directory for the ``Ensemble`` member outputs and configuration files, invoke ``Experiment.generate()`` via the
``Experiment`` instance `exp` with `example_ensemble` as an input parameter:

.. literalinclude:: ../tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
    :language: python
    :linenos:
    :lines: 17-18

After invoking ``Experiment.generate()``, the attached generator files will be available for the
application when ``exp.start(example_ensemble)`` is called.

The contents of `params_inputs.txt` after ``Ensemble`` completion are:

.. code-block:: txt

   THERMO = 1

=====================
ML Models and Scripts
=====================
Overview
========
SmartSim users have the capability to utilize ML runtimes within an ``Ensemble``.
Functions accessible through an ``Ensemble`` object support loading ML models (TensorFlow, TensorFlow-lite,
PyTorch, and ONNX) and TorchScripts into standalone ``Orchestrators`` or colocated ``Orchestrators`` at
application runtime.

Depending on the storage method of the ML model, there are **two** distinct approaches to load it into the ``Orchestrator``:

- :ref:`from memory<in_mem_ML_model_ensemble_ex>`
- :ref:`from file<from_file_ML_model_ensemble_ex>`

Depending on the storage method of the TorchScript, there are **three** distinct approaches to load it into the ``Orchestrator``:

- :ref:`from memory<in_mem_TF_ensemble_doc>`
- :ref:`from file<TS_from_file_ensemble>`
- :ref:`from string<TS_raw_string_ensemble>`

Once a ML model or TorchScript is loaded into the ``Orchestrator``, ``Ensemble`` ``Model`` members can
leverage ML capabilities by utilizing the SmartSim client (:ref:`SmartRedis<smartredis-api>`)
to execute the stored ML models or TorchScripts.

.. _ai_model_ensemble_doc:
AI Models
=========
When configuring an ``Ensemble``, users can instruct SmartSim to load
Machine Learning (ML) models dynamically to the ``Orchestrator`` (colocated or standalone). ML models added
are loaded into the ``Orchestrator`` prior to the execution of the ``Ensemble``. To load an ML model
to the ``Orchestrator``, SmartSim users can provide the ML model **in-memory** or specify the **file path**
when using the ``Ensemble.add_ml_model()`` function. The supported ML frameworks are TensorFlow,
TensorFlow-lite, PyTorch, and ONNX.

When attaching an ML model using ``Ensemble.add_ml_model()``, the
following arguments are offered to customize the storage and execution of the ML model:

- `name` (str): name to reference the ML model in the ``Orchestrator``.
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

.. _in_mem_ML_model_ensemble_ex:
-------------------------------------
Example: Attach an in-memory ML Model
-------------------------------------
This example demonstrates how to attach an in-memory ML model to a SmartSim ``Ensemble``
to load into an ``Orchestrator`` at ``Ensemble`` runtime.

.. note::
    This example assumes:

    - an ``Orchestrator`` is launched prior to the ``Ensemble`` execution
    - an initialized ``Ensemble`` named `smartsim_ensemble` exists within the ``Experiment`` workflow

**Define an in-memory Keras CNN**

The ML model must be defined using one of the supported ML frameworks. For the purpose of the example,
we define a Keras CNN, which is based on Tensorflow, in the same script as the SmartSim ``Experiment``:

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
the ``Ensemble.add_ml_model()`` function and specify the in-memory ML model to the parameter `model`:

.. code-block:: python

    smartsim_ensemble.add_ml_model(name="cnn", backend="TF", model=model, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)

In the above ``smartsim_ensemble.add_ml_model()`` code snippet, we offer the following arguments:

-  `name` ("cnn"): A name to reference the ML model in the ``Orchestrator``.
-  `backend` ("TF"): Indicating that the ML model is a TensorFlow model.
-  `model` (model): The in-memory representation of the TensorFlow model.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When the ``Ensemble`` is started via ``Experiment.start()``, the ML model will be loaded to the
launched ``Orchestrator``. The ML model can then be executed on the ``Orchestrator`` via a SmartSim
client (:ref:`SmartRedis<smartredis-api>`) within the application code.

.. _from_file_ML_model_ensemble_ex:
----------------------------------------
Example: Attaching an ML Model from file
----------------------------------------
This example demonstrates how to attach a ML model from file to a SmartSim ``Ensemble``
to load into an ``Orchestrator`` at ``Ensemble`` runtime.

.. note::
    This example assumes:

    - a standalone ``Orchestrator`` is launched prior to ``Ensemble`` execution
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

**Attach the ML model to a SmartSim Ensemble**

Assuming an initialized ``Ensemble`` named `smartsim_ensemble` exists, we add a TensorFlow model using
the ``Ensemble.add_ml_model()`` function and specify the ML model path to the parameter `model_path`:

.. code-block:: python

    smartsim_model.add_ml_model(name="cnn", backend="TF", model_path=model_file, device="GPU", devices_per_node=2, first_device=0, inputs=inputs, outputs=outputs)

In the above ``smartsim_ensemble.add_ml_model()`` code snippet, we offer the following arguments:

-  `name` ("cnn"): A name to reference the ML model in the ``Orchestrator``.
-  `backend` ("TF"): Indicating that the ML model is a TensorFlow model.
-  `model_path` (model_file): The path to the ML model script.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When the ``Ensemble`` is started via ``Experiment.start()``, the ML model will be loaded to the
launched ``Orchestrator``. The ML model can then be executed on the ``Orchestrator`` via a SmartSim
client (:ref:`SmartRedis<smartredis-api>`) within the application code.

.. _TS_ensemble_doc:
TorchScripts
============
When configuring an ``Ensemble``, users can instruct SmartSim to load TorchScripts dynamically
to the ``Orchestrator``. TorchScripts added are loaded into the ``Orchestrator`` prior to
the execution of the ``Ensemble``. To load a TorchScript to the ``Orchestrator``, SmartSim users
can follow one of the processes:

- :ref:`Define a TorchScript function in-memory<in_mem_TF_doc>`
   Use the ``Ensemble.add_function()`` to instruct SmartSim to load an in-memory TorchScript to the ``Orchestrator``.
- :ref:`Define a TorchScript function from file<TS_from_file>`
   Provide file path to ``Ensemble.add_script()`` to instruct SmartSim to load the TorchScript from file to the ``Orchestrator``.
- :ref:`Define a TorchScript function as string<TS_raw_string>`
   Provide function string to ``Ensemble.add_script()`` to instruct SmartSim to load a raw string as a TorchScript function to the ``Orchestrator``.

Continue or select the respective process link to learn more on how each function (``Ensemble.add_script()`` and ``Ensemble.add_function()``)
dynamically loads TorchScripts to the ``Orchestrator``.

.. _in_mem_TF_ensemble_doc:
-------------------------------
Attach an in-memory TorchScript
-------------------------------
Users can define TorchScript functions within the Python driver script
to attach to an ``Ensemble``. This feature is supported by ``Ensemble.add_function()`` which provides flexible
device selection, allowing users to choose between which device the the TorchScript is executed on, `"GPU"` or `"CPU"`.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

.. warning::
    ``Ensemble.add_function()`` does **not** support loading in-memory TorchScript functions to a colocated ``Orchestrator``.
    If you would like to load a TorchScript function to a colocated ``Orchestrator``, define the function
    as a :ref:`raw string<TS_raw_string>` or :ref:`load from file<TS_from_file>`.

When specifying an in-memory TF function using ``Ensemble.add_function()``, the
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

.. note::
    The example assumes:

    - a standalone ``Orchestrator`` is launched prior to ``Ensemble`` execution
    - an initialized ``Ensemble`` named `smartsim_ensemble` exists within the ``Experiment`` workflow

**Define an in-memory TF function**

To begin, define an in-memory TorchScript function within the Python driver script.
For the purpose of the example, we add a simple TorchScript function, `timestwo`:

.. code-block:: python

    def timestwo(x):
        return 2*x

**Attach the in-memory TorchScript function to a SmartSim Ensemble**

We use the ``Ensemble.add_function()`` function to instruct SmartSim to load the TorchScript function `timestwo`
onto the launched standalone ``Orchestrator``. Specify the function `timestwo` to the `function`
parameter:

.. code-block:: python

    smartsim_ensemble.add_function(name="example_func", function=timestwo, device="GPU", devices_per_node=2, first_device=0)

In the above ``smartsim_ensemble.add_function()`` code snippet, we offer the following arguments:

-  `name` ("example_func"): A name to uniquely identify the ML model within the ``Orchestrator``.
-  `function` (timestwo): Name of the TorchScript function defined in the Python driver script.
-  `device` ("CPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When the ``Ensemble`` is started via ``Experiment.start()``, the TF function will be loaded to the
standalone ``Orchestrator``. The function can then be executed on the ``Orchestrator`` via a SmartSim
client (:ref:`SmartRedis<smartredis-api>`) within the application code.

.. _TS_from_file_ensemble:
------------------------------
Attach a TorchScript from file
------------------------------
Users can attach TorchScript functions from a file to an ``Ensemble`` and upload them to a
colocated or standalone ``Orchestrator``. This functionality is supported by the ``Ensemble.add_script()``
function's `script_path` parameter. The function supports
flexible device selection, allowing users to choose between `"GPU"` or `"CPU"` via the `device` parameter.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

When specifying a TorchScript using ``Ensemble.add_script()``, the
following arguments are offered:

- `name` (str): Reference name for the script inside of the ``Orchestrator``.
- `script` (t.Optional[str] = None): TorchScript code (only supported for non-colocated ``Orchestrators``).
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

    - a ``Orchestrator`` is launched prior to ``Ensemble`` execution
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

In the above ``smartsim_model.add_script()`` code snippet, we offer the following arguments:

-  `name` ("example_script"): Reference name for the script inside of the ``Orchestrator``.
-  `script_path` ("path/to/torchscript.py"): Path to the script file.
-  `device` ("CPU"): device for script execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When `smartsim_ensemble` is started via ``Experiment.start()``, the TorchScript will be loaded from file to the
``Orchestrator`` that is launched prior to the start of the `smartsim_ensemble`.

.. _TS_raw_string_ensemble:
---------------------------------
Define TorchScripts as raw string
---------------------------------
Users can upload TorchScript functions from string to send to a colocated or
standalone ``Orchestrator``. This feature is supported by the
``Ensemble.add_script()`` function's `script` parameter. The function supports
flexible device selection, allowing users to choose between `"GPU"` or `"CPU"` via the `device` parameter.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

When specifying a TorchScript using ``Ensemble.add_script()``, the
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
This example walks through the steps of instructing SmartSim to load a TorchScript function
from string to a ``Orchestrator`` before the execution of the associated ``Model``.

.. note::
    This example assumes:

    - a ``Orchestrator`` is launched prior to ``Ensemble`` execution
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
    Calling `exp.start(smartsim_ensemble)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When the ``Ensemble`` is started via ``Experiment.start()``, the TorchScript will be loaded to the
``Orchestrator`` that is launched prior to the start of the ``Ensemble``.


.. _prefix_ensemble:
=========================
Data Collision Prevention
=========================
Overview
========
When multiple ``Ensemble`` members use the same code to access their respective data
in the ``Orchestrator``, key overlapping can occur, leading to inadvertent data access
between ``Ensemble`` members. To address this, a SmartSim ``Ensemble`` supports key prefixing
via the ``Ensemble.enable_key_prefixing()`` function,
which automatically adds the ``Ensemble`` member `name` as a prefix to the keys sent to the ``Orchestrator``.
Enabling key prefixing eliminates issues related to key overlapping, allowing ``Ensemble``
members to use the same code without issue.

Example: Ensemble Key Prefixing
===============================
In this example, we create an ``Ensemble`` comprised of two ``Models`` that use identical code
to send data to a standalone ``Orchestrator``. To prevent key collisions and ensure data
integrity, we enable key prefixing on the ``Ensemble`` which automatically
appends the ``Ensemble`` member `name` to the data sent to the ``Orchestrator``. After the
``Ensemble`` completes, we launch a consumer ``Model`` within the Python driver script
to demonstrate accessing prefixed data sent to the ``Orchestrator`` by ``Ensemble`` members.

-------------------------------
The Application Producer Script
-------------------------------
In the Python driver script, we instruct SmartSim to create an ``Ensemble`` comprised of
two ``Models`` that execute the same application script.
In the application script, a SmartRedis ``Client`` sends a
tensor to the ``Orchestrator``. Since both ``Ensemble`` ``Models`` use the same application script,
two identical tensors are placed on the ``Orchestrator`` which could cause a key collision.
To prevent this case, we enable key prefixing on the ``Ensemble`` in the driver script
via ``Ensemble.enable_key_prefixing()``.
This means that when an ``Ensemble`` member places a tensor on the ``Orchestrator``, SmartSim will prepend
the ``Ensemble`` member `name` to the tensor `name`.

Here we provide the application script for the ``Ensemble``:

.. code-block:: python

    from smartredis import Client, log_data
    import numpy as np

    # Initialize a Client
    client = Client(cluster=False)

    # Create NumPy array
    array = np.array([1, 2, 3, 4])
    # Use SmartRedis client to place tensor in standalone Orchestrator
    client.put_tensor("tensor", array)

After the completion of the ``Ensemble`` using the application script, the contents of the ``Orchestrator`` are:

.. code-block:: bash

    1) "producer_0.tensor"
    2) "producer_1.tensor"

-------------------------------
The Application Consumer Script
-------------------------------
In the Python driver script, we initialize a consumer ``Model`` that requests
the tensors produced from the ``Ensemble``. To do so, we use SmartRedis
key prefixing functionality to instruct the SmartRedis ``Client`` to append
the name of an ``Ensemble`` member to the key `name`.

First specify the imports and initialize a SmartRedis ``Client``:

.. code-block:: python

    from smartredis import Client, log_data, LLInfo

    # Initialize a Client
    client = Client(cluster=False)

To retrieve the tensor from the first ``Ensemble`` member, use
``Client.set_data_source()``. Specify the name of the first ``Ensemble`` member, `producer_0`,
to the function to instruct SmartSim to append the ``Ensemble`` member name when searching
for data in the ``Orchestrator``. When ``Client.poll_tensor()`` is executed in the consumer ``Model``,
the `client` will poll for key, `producer_0.tensor`:

.. code-block:: python

    # Set the data source
    client.set_data_source("producer_0")
    # Check if the tensor exists
    val1 = client.poll_tensor("tensor", 100, 100)

Follow the same instructions above, however, change the data source `name` to the `name`
of the second ``Ensemble`` member (`producer_1`):

.. code-block:: python

    # Set the data source
    client.set_data_source("producer_1")
    # Check if the tensor exists
    val2 = client.poll_tensor("tensor", 100, 100)

We print the boolean return to verify that the tensors were found:

.. code-block:: python

    client.log_data(LLInfo, f"producer_0.tensor was found: {val1}")
    client.log_data(LLInfo, f"producer_1.tensor was found: {val2}")

When the ``Experiment`` driver script is executed, the following output will appear in `consumer.out`:

.. code-block:: bash

    Default@11-46-05:producer_0.tensor was found: True
    Default@11-46-05:producer_1.tensor was found: True

.. note::
    For SmartSim to recognize the ``Ensemble`` member names as a valid data source
    to ``Client.set_data_source()``, you must register the ``Ensemble`` member
    on the consumer ``Model`` in the driver script via ``Model.register_incoming_entity()``.
    We demonstrate this in the ``Experiment`` driver script section of the example.

---------------------
The Experiment Script
---------------------
In the ``Experiment`` driver script we initialize:

- a standalone ``Orchestrator``
- an ``Ensemble`` via the replicas initialization strategy
- a consumer ``Model``

In the example it is essential to launch the ``Orchestrator`` before any other SmartSim entity since each simulation
connects to a launched ``Orchestrator``.

To setup for the example in the Python driver script, we begin by

-  initializing the ``Experiment`` instance, `exp`
-  initializing the standalone ``Orchestrator``, `standalone_orch`

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

    # Initialize a single sharded Orchestrator
    standalone_orch = exp.create_database(port=6379, db_nodes=1, interface="ib0")

We are now setup to discuss key prefixing within the ``Experiment`` driver script.
To create an ``Ensemble`` using the replicas strategy, begin by initializing a ``RunSettings``
object to apply to all ``Ensemble`` members. Specify the path to the application
producer script:

.. code-block:: python

    # Initialize a RunSettings object
    ensemble_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/producer_script.py")

Next, initialize an ``Ensemble`` by specifying `ensemble_settings` and the number of ``Model`` `replicas` to create:

.. code-block:: python

    producer_ensemble = exp.create_ensemble("producer", run_settings=ensemble_settings, replicas=2)

Instruct SmartSim to prefix all tensors sent to the ``Orchestrator`` from the ``Ensemble`` via ``Ensemble.enable_key_prefixing()``:

.. code-block:: python

    producer_ensemble.enable_key_prefixing()

Next, initialize the consumer ``Model`` that requests the prefixed tensors produced by the ``Ensemble``:

.. code-block:: python

    # Initialize a RunSettings object
    model_settings = exp.create_run_settings(exe=exe_ex, exe_args="/path/to/consumer_script.py")
    # Create the Model
    consumer_model = exp.create_model("consumer", model_settings)

Next, organize the SmartSim entity output files into a single ``Experiment`` folder:

.. code-block:: python

    exp.generate(standalone_orch, producer_ensemble, consumer_model, overwrite=True)

Launch the ``Orchestrator``:

.. code-block:: python

    exp.start(standalone_orch, summary=True)

Launch the ``Ensemble``:

.. code-block:: python

    exp.start(producer_ensemble, block=True, summary=True)

Set `block=True` so that ``Experiment.start()`` waits until the last ``Ensemble`` member has finished before continuing.

The consumer ``Model`` application script uses ``Client.set_data_source()`` which
accepts the ``Ensemble`` member names when searching for prefixed
keys in the ``Orchestrator``. In order for SmartSim to recognize the ``Ensemble``
member names as a valid data source in the consumer ``Model``, we must register
the entity interaction:

.. code-block:: python

    for model in producer_ensemble:
        consumer_model.register_incoming_entity(model)

Launch the consumer ``Model``:

.. code-block:: python

    exp.start(consumer_model, block=True, summary=True)

To finish, tear down the standalone ``Orchestrator``:

.. code-block:: python

    exp.stop(single_shard_db)