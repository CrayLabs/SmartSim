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
launched with other :ref:`Model's<model_object_doc>` and :ref:`Orchestrators<orch_docs>` to construct AI-enabled workflows.

The :ref:`Ensemble API<ensemble_api>` offers key features, including methods to:

- :ref:`Attach Configuration Files<attach_files_ensemble>` for use at ``Ensemble`` runtime.
- :ref:`Load AI Models<ai_model_ensemble_doc>` (TF, TF-lite, PT, or ONNX) into the ``Orchestrator`` at ``Ensemble`` runtime.
- :ref:`Load TorchScripts<TS_ensemble_doc>` into the ``Orchestrator`` at ``Ensemble`` runtime.
- :ref:`Prevent Data Collisions<prefix_ensemble>` within the ``Ensemble``, which allows for reuse of application code.

To create a SmartSim ``Ensemble``, use the ``Experiment.create_ensemble`` API function. When
initializing an ``Ensemble``, consider one of the **three** creation strategies explained
in the :ref:`Initialization<init_ensemble_strategies>` section.

SmartSim manages ``Ensemble`` instances through the :ref:`Experiment API<experiment_api>` by providing functions to
launch, monitor, and stop applications.

.. _init_ensemble_strategies:

==============
Initialization
==============
Overview
========
The :ref:`Experiment API<experiment_api>` is responsible for initializing all workflow entities.
An ``Ensemble`` is created using the ``Experiment.create_ensemble`` factory method, and users can customize the
``Ensemble`` creation via the factory method parameters.

The factory method arguments for ``Ensemble`` creation can be found in the :ref:`Experiment API<exp_init>`
under the ``create_ensemble`` docstring.

By using specific combinations of the factory method arguments, users can tailor
the creation of an ``Ensemble`` to align with one of the following creation strategies:

1. :ref:`Parameter Expansion<param_expansion_init>`: Generate a variable-sized set of unique simulation instances
   configured with user-defined input parameters.
2. :ref:`Replica Creation<replicas_init>`: Generate a specified number of ``Model`` replicas.
3. :ref:`Manually<append_init>`: Attach pre-configured ``Model``'s to an ``Ensemble`` to manage as a single unit.

.. _param_expansion_init:

Parameter Expansion
===================
Parameter expansion is a technique that allows users to set parameter values per ``Ensemble`` member.
This is done by specifying input to the `params` and `perm_strategy` factory method arguments during
``Ensemble`` creation (``Experiment.create_ensemble``). Users may control how the `params` values
are applied to the ``Ensemble`` through the `perm_strategy` argument. The `perm_strategy` argument
accepts three values listed below.

**Parameter Expansion Strategy Options:**

-  `"all_perm"`: Generate all possible parameter permutations for an exhaustive exploration. This
   means that every possible combination of parameters will be used in the ``Ensemble``.
-  `"step"`: Create parameter sets by collecting identically indexed values across parameter lists.
   This allows for discrete combinations of parameters for ``Model``'s.
-  `"random"`: Enable random selection from predefined parameter spaces, offering a stochastic approach.
   This means that the parameters will be chosen randomly for each ``Model``, which can be useful
   for exploring a wide range of possibilities.

--------
Examples
--------
This subsection contains two examples of ``Ensemble`` parameter expansion. The
:ref:`first example<param_first_ex>` illustrates parameter expansion using two parameters
while the :ref:`second example<param_second_ex>` demonstrates parameter expansion with two
parameters along with the launch of the ``Ensemble`` as a batch workload.

.. _param_first_ex:

Example 1 : Parameter Expansion Using `all_perm` Strategy

    In this example an ``Ensemble`` of four ``Model`` entities is created by expanding two parameters
    using the `all_perm` strategy. All of the ``Model``'s in the ``Ensemble`` share the same ``RunSettings``
    and only differ in the value of the `params` assigned to each member. The source code example
    is available in the dropdown below for convenient execution and customization.

    .. dropdown:: Example Driver Script Source Code

        .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/param_expansion_1.py

    Begin by initializing a ``RunSettings`` object to apply to
    all ``Ensemble`` members:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/param_expansion_1.py
        :language: python
        :linenos:
        :lines: 6-7

    Next, define the parameters that will be applied to the ``Ensemble``:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/param_expansion_1.py
        :language: python
        :linenos:
        :lines: 9-13

    Finally, initialize an ``Ensemble`` by specifying the ``RunSettings``, `params` and `perm_strategy="all_perm"`:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/param_expansion_1.py
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

.. _param_second_ex:

Example 2 : Parameter Expansion Using `step` Strategy with the ``Ensemble`` Configured For Batch Launching

    In this example an ``Ensemble`` of two ``Model`` entities is created by expanding two parameters
    using the `step` strategy. All of the ``Model``'s in the ``Ensemble`` share the same ``RunSettings``
    and only differ in the value of the `params` assigned to each member. Lastly, the ``Ensemble`` is
    submitted as a batch workload. The source code example is available in the dropdown below for
    convenient execution and customization.

    .. dropdown:: Example Driver Script source code

        .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py

    Begin by initializing and configuring a ``BatchSettings`` object to
    run the ``Ensemble`` instance:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py
        :language: python
        :linenos:
        :lines: 6-8

    The above ``BatchSettings`` object will instruct SmartSim to run the ``Ensemble`` on two
    nodes with a timeout of `10 hours`.

    Next initialize a ``RunSettings`` object to apply to all ``Ensemble`` members:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py
        :language: python
        :linenos:
        :lines: 10-12

    Next, define the parameters to include in ``Ensemble``:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py
        :language: python
        :linenos:
        :lines: 14-18

    Finally, initialize an ``Ensemble`` by passing in the ``RunSettings``, `params` and `perm_strategy="step"`:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/param_expansion_2.py
        :language: python
        :linenos:
        :lines: 20-21

    When specifying `perm_strategy="step"`, the `params` sets are created by collecting identically
    indexed values across the `param` value lists.

    .. code-block:: bash

        ensemble member 1: ["Ellie", 2]
        ensemble member 2: ["John", 11]

.. _replicas_init:

Replicas
========
A replica strategy involves the creation of identical ``Model``'s within an ``Ensemble``.
This strategy is particularly useful for applications that have some inherent randomness.
Users may use the `replicas` factory method argument to create a specified number of identical
``Model`` members during ``Ensemble`` creation (``Experiment.create_ensemble``).

--------
Examples
--------
This subsection contains two examples of using the replicas creation strategy. The
:ref:`first example<replicas_first_ex>` illustrates creating four ``Ensemble`` member clones
while the :ref:`second example<replicas_second_ex>` demonstrates creating four ``Ensemble``
member clones along with the launch of the ``Ensemble`` as a batch workload.

.. _replicas_first_ex:

Example 1 : ``Ensemble`` creation with replicas strategy

    In this example an ``Ensemble`` of four identical ``Model`` members is created by
    specifying the number of clones to create via the `replicas` argument.
    All of the ``Model``'s in the ``Ensemble`` share the same ``RunSettings``.
    The source code example is available in the dropdown below for convenient execution
    and customization.

    .. dropdown:: Example Driver Script Source Code

        .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/replicas_1.py

    To create an ``Ensemble`` of identical ``Model``'s, begin by initializing a ``RunSettings``
    object:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/replicas_1.py
        :language: python
        :linenos:
        :lines: 6-7

    Initialize the ``Ensemble`` by specifying the ``RunSettings`` object and number of clones to `replicas`:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/replicas_1.py
        :language: python
        :linenos:
        :lines: 9-10

    By passing in `replicas=4`, four identical ``Ensemble`` members will be initialized.

.. _replicas_second_ex:

Example 2 : ``Ensemble`` Creation with Replicas Strategy and ``Ensemble`` Batch Launching

    In this example an ``Ensemble`` of four ``Model`` entities is created by specifying
    the number of clones to create via the `replicas` argument. All of the ``Model``'s in
    the ``Ensemble`` share the same ``RunSettings`` and the ``Ensemble`` is
    submitted as a batch workload. The source code example is available in the dropdown below for
    convenient execution and customization.

    .. dropdown:: Example Driver Script Source Code

        .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/replicas_2.py

    To launch the ``Ensemble`` of identical ``Model``'s as a batch job, begin by initializing a ``BatchSettings``
    object:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/replicas_2.py
        :language: python
        :linenos:
        :lines: 6-9

    The above ``BatchSettings`` object will instruct SmartSim to run the ``Ensemble`` on four
    nodes with a timeout of `10 hours`.

    Next, create a ``RunSettings`` object to apply to all ``Model`` replicas:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/replicas_2.py
        :language: python
        :linenos:
        :lines: 10-12

    Initialize the ``Ensemble`` by specifying the ``RunSettings`` object, ``BatchSettings`` object
    and number of clones to `replicas`:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/replicas_2.py
        :language: python
        :linenos:
        :lines: 14-15

    By passing in `replicas=4`, four identical ``Ensemble`` members will be initialized.

.. _append_init:

Manually Append
===============
Manually appending ``Model``'s to an ``Ensemble`` offers an in-depth level of customization in ``Ensemble`` design.
This approach is favorable when users have distinct requirements for individual ``Model``'s, such as variations
in parameters, run settings, or different types of simulations.

--------
Examples
--------
This subsection contains an example of creating an ``Ensemble`` by manually appending ``Model``'s.
The example illustrates attaching two SmartSim ``Model``'s to the ``Ensemble``.
The ``Ensemble`` is submitted as a batch workload.

Example 1 : Append ``Model``'s to an ``Ensemble`` and Launch as a Batch Job

    In this example, we append ``Model``'s to an ``Ensemble`` for batch job execution. To do
    this, we first initialize an Ensemble with a ``BatchSettings`` object. Then, manually
    create ``Model``'s and add each to the ``Ensemble`` using the ``Ensemble.add_model`` function.
    The source code example is available in the dropdown below for convenient execution and customization.

    .. dropdown:: Example Driver Script Source Code

        .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py

    To create an empty ``Ensemble`` to append ``Model``'s, initialize the ``Ensemble`` with
    a batch settings object:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
        :language: python
        :linenos:
        :lines: 6-11

    Next, create the ``Model``'s to append to the ``Ensemble``:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
        :language: python
        :linenos:
        :lines: 13-20

    Finally, append the ``Model`` objects to the ``Ensemble``:

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/manual_append_ensemble.py
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
prior to an ``Ensemble`` launch via the ``Ensemble.attach_generator_files`` function. Attached files
will be applied to all ``Ensemble`` members.

.. note::
    Multiple calls to ``Ensemble.attach_generator_files`` will overwrite previous file configurations
    on the ``Ensemble``.

To attach a file to an ``Ensemble`` for use at runtime, provide one of the following arguments to the
``Ensemble.attach_generator_files`` function:

* `to_copy` (t.Optional[t.List[str]] = None): Files that are copied into the path of the ``Ensemble`` members.
* `to_symlink` (t.Optional[t.List[str]] = None): Files that are symlinked into the path of the ``Ensemble`` members.
  A symlink, or symbolic link, is a file that points to another file or directory, allowing you to access that file
  as if it were located in the same directory as the symlink.

To specify a template file in order to programmatically replace specified parameters during generation
of ``Ensemble`` member directories, pass the following value to the ``Ensemble.attach_generator_files`` function:

* `to_configure` (t.Optional[t.List[str]] = None): This parameter is designed for text-based ``Ensemble``
  member input files. During directory generation for ``Ensemble`` members, the linked files are parsed and replaced with
  the `params` values applied to each ``Ensemble`` member. To further explain, the ``Ensemble``
  creation strategy is considered when replacing the tagged parameters in the input files.
  These tagged parameters are placeholders in the text that are replaced with the actual
  parameter values during the directory generation process. The default tag is a semicolon
  (e.g., THERMO = ;THERMO;).

In the :ref:`Example<files_example_doc_ensem>` subsection, we provide an example using the value `to_configure`
within ``Ensemble.attach_generator_files``.

.. seealso::
    To add a file to a single ``Model`` that will be appended to an ``Ensemble``, refer to the :ref:`Files<files_doc>`
    section of the ``Model`` documentation.

.. _files_example_doc_ensem:

Example
=======
This example demonstrates how to attach a text file to an ``Ensemble`` for parameter replacement.
This is accomplished using the `params` function parameter in
the ``Experiment.create_ensemble`` factory function and the `to_configure` function parameter
in ``Ensemble.attach_generator_files``. The source code example is available in the dropdown below for
convenient execution and customization.

.. dropdown:: Example Driver Script Source Code

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/file_attach.py

In this example, we have a text file named `params_inputs.txt`. Within the text, is the parameter `THERMO`
that is required by each ``Ensemble`` member at runtime:

.. code-block:: bash

   THERMO = ;THERMO;

In order to have the tagged parameter `;THERMO;` replaced with a usable value at runtime, two steps are required:

1. The `THERMO` variable must be included in ``Experiment.create_ensemble`` factory method as
   part of the `params` parameter.
2. The file containing the tagged parameter `;THERMO;`, `params_inputs.txt`, must be attached to the ``Ensemble``
   via the ``Ensemble.attach_generator_files`` method as part of the `to_configure` parameter.

To encapsulate our application within an ``Ensemble``, we must create an ``Experiment`` instance
to gain access to the ``Experiment`` factory method that creates the ``Ensemble``.
Begin by importing the ``Experiment`` module and initializing an ``Experiment``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/file_attach.py
    :language: python
    :linenos:
    :lines: 1-4

To create our ``Ensemble``, we are using the `replicas` initialization strategy.
Begin by creating a simple ``RunSettings`` object to specify the path to
the executable simulation as an executable:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/file_attach.py
    :language: python
    :linenos:
    :lines: 6-7

Next, initialize an ``Ensemble`` object with ``Experiment.create_ensemble``
by passing in `ensemble_settings`, `params={"THERMO":1}` and `replicas=2`:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/file_attach.py
    :language: python
    :linenos:
    :lines: 9-10

We now have an ``Ensemble`` instance named `example_ensemble`. Attach the above text file
to the ``Ensemble`` for use at entity runtime. To do so, we use the
``Ensemble.attach_generator_files`` function and specify the `to_configure`
parameter with the path to the text file, `params_inputs.txt`:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/file_attach.py
    :language: python
    :linenos:
    :lines: 12-13

To create an isolated directory for the ``Ensemble`` member outputs and configuration files, invoke ``Experiment.generate`` via the
``Experiment`` instance `exp` with `example_ensemble` as an input parameter:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/file_attach.py
    :language: python
    :linenos:
    :lines: 15-16

After invoking ``Experiment.generate``, the attached generator files will be available for the
application when ``exp.start(example_ensemble)`` is called.

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/file_attach.py
    :language: python
    :linenos:
    :lines: 18-19

The contents of `params_inputs.txt` after ``Ensemble`` completion are:

.. code-block:: bash

   THERMO = 1

.. _ensemble_ml_model_script:

=====================
ML Models and Scripts
=====================
Overview
========
SmartSim users have the capability to load ML models and TorchScripts into an ``Orchestrator``
within the ``Experiment`` script for use within ``Ensemble`` members. Functions
accessible through an ``Ensemble`` object support loading ML models (TensorFlow, TensorFlow-lite,
PyTorch, and ONNX) and TorchScripts into standalone or colocated ``Orchestrators`` before
application runtime.

.. seealso::
    To add an ML model or TorchScript to a single ``Model`` that will be appended to an
    ``Ensemble``, refer to the :ref:`ML Models and Scripts<ml_script_model_doc>`
    section of the ``Model`` documentation.

Depending on the planned storage method of the **ML model**, there are **two** distinct
approaches to load it into the ``Orchestrator``:

- :ref:`From Memory<in_mem_ML_model_ensemble_ex>`
- :ref:`From File<from_file_ML_model_ensemble_ex>`

.. warning::
    Uploading an ML model :ref:`from memory<in_mem_ML_model_ensemble_ex>` is solely supported for
    standalone ``Orchestrators``. To upload an ML model to a colocated ``Orchestrator``, users
    must save the ML model to disk and upload :ref:`from file<from_file_ML_model_ensemble_ex>`.

Depending on the planned storage method of the **TorchScript**, there are **three** distinct
approaches to load it into the ``Orchestrator``:

- :ref:`From Memory<in_mem_TF_ensemble_doc>`
- :ref:`From File<TS_from_file_ensemble>`
- :ref:`From String<TS_raw_string_ensemble>`

.. warning::
    Uploading a TorchScript :ref:`from memory<in_mem_TF_ensemble_doc>` is solely supported for
    standalone ``Orchestrators``. To upload a TorchScript to a colocated ``Orchestrator``, users
    upload :ref:`from file<TS_from_file_ensemble>` or :ref:`from string<TS_raw_string_ensemble>`.

Once a ML model or TorchScript is loaded into the ``Orchestrator``, ``Ensemble`` members can
leverage ML capabilities by utilizing the SmartSim client (:ref:`SmartRedis<smartredis-api>`)
to execute the stored ML models or TorchScripts.

.. _ai_model_ensemble_doc:

AI Models
=========
When configuring an ``Ensemble``, users can instruct SmartSim to load
Machine Learning (ML) models dynamically to the ``Orchestrator`` (colocated or standalone). ML models added
are loaded into the ``Orchestrator`` prior to the execution of the ``Ensemble``. To load an ML model
to the ``Orchestrator``, SmartSim users can serialize and provide the ML model **in-memory** or specify the **file path**
via the ``Ensemble.add_ml_model`` function. The supported ML frameworks are TensorFlow,
TensorFlow-lite, PyTorch, and ONNX.

Users must **serialize TensorFlow ML models** before sending to an ``Orchestrator`` from memory
or from file. To save a TensorFlow model to memory, SmartSim offers the ``serialize_model``
function. This function returns the TF model as a byte string with the names of the
input and output layers, which will be required upon uploading. To save a TF model to disk,
SmartSim offers the ``freeze_model`` function which returns the path to the serialized
TF model file with the names of the input and output layers. Additional TF model serialization
information and examples can be found in the :ref:`ML Features<ml_features_docs>` section of SmartSim.

.. note::
    Uploading an ML model from memory is only supported for standalone ``Orchestrators``.

When attaching an ML model using ``Ensemble.add_ml_model``, the
following arguments are offered to customize storage and execution:

- `name` (str): name to reference the ML model in the ``Orchestrator``.
- `backend` (str): name of the backend (TORCH, TF, TFLITE, ONNX).
- `model` (t.Optional[str] = None): An ML model in memory (only supported for non-colocated ``Orchestrators``).
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

.. seealso::
    To add an ML model to a single ``Model`` that will be appended to an
    ``Ensemble``, refer to the :ref:`AI Models<ai_model_doc>`
    section of the ``Model`` documentation.

.. _in_mem_ML_model_ensemble_ex:

-------------------------------------
Example: Attach an In-Memory ML Model
-------------------------------------
This example demonstrates how to attach an in-memory ML model to a SmartSim ``Ensemble``
to load into an ``Orchestrator`` at ``Ensemble`` runtime. The source code example is
available in the dropdown below for convenient execution and customization.

.. dropdown:: Experiment Driver Script Source Code

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_ml_model_mem.py

.. note::
    This example assumes:

    - an ``Orchestrator`` is launched prior to the ``Ensemble`` execution
    - an initialized ``Ensemble`` named `ensemble_instance` exists within the ``Experiment`` workflow
    - a Tensorflow-based ML model was serialized using ``serialize_model`` which returns the
      ML model as a byte string with the names of the input and output layers

**Attach the ML Model to a SmartSim Ensemble**

In this example, we have a serialized Tensorflow-based ML model that was saved to a byte string stored under `model`.
Additionally, the ``serialize_model`` function returned the names of the input and output layers stored under
`inputs` and `outputs`. Assuming an initialized ``Ensemble`` named `ensemble_instance` exists, we add the byte string TensorFlow model using
``Ensemble.add_ml_model``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_ml_model_mem.py
  :language: python
  :linenos:
  :lines: 39-40

In the above ``ensemble_instance.add_ml_model`` code snippet, we offer the following arguments:

-  `name` ("cnn"): A name to reference the ML model in the ``Orchestrator``.
-  `backend` ("TF"): Indicating that the ML model is a TensorFlow model.
-  `model` (model): The in-memory representation of the TensorFlow model.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(ensemble_instance)` prior to the launch of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent standalone ``Orchestrator``.

When the ``Ensemble`` is started via ``Experiment.start``, the ML model will be loaded to the
launched standalone ``Orchestrator``. The ML model can then be executed on the ``Orchestrator`` via a SmartSim
client (:ref:`SmartRedis<smartredis-api>`) within the application code.

.. _from_file_ML_model_ensemble_ex:

-------------------------------------
Example: Attach an ML Model From File
-------------------------------------
This example demonstrates how to attach a ML model from file to a SmartSim ``Ensemble``
to load into an ``Orchestrator`` at ``Ensemble`` runtime. The source code example is
available in the dropdown below for convenient execution and customization.

.. dropdown:: Experiment Driver Script Source Code

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_ml_model_file.py

.. note::
    This example assumes:

    - a standalone ``Orchestrator`` is launched prior to ``Ensemble`` execution
    - an initialized ``Ensemble`` named `ensemble_instance` exists within the ``Experiment`` workflow
    - a Tensorflow-based ML model was serialized using ``freeze_model`` which returns the
      the path to the serialized model file and the names of the input and output layers

**Attach the ML Model to a SmartSim Ensemble**

In this example, we have a serialized Tensorflow-based ML model that was saved to disk and stored under `model`.
Additionally, the ``freeze_model`` function returned the names of the input and output layers stored under
`inputs` and `outputs`. Assuming an initialized ``Ensemble`` named `ensemble_instance` exists, we add a TensorFlow model using
the ``Ensemble.add_ml_model`` function and specify the ML model path to the parameter `model_path`:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_ml_model_file.py
  :language: python
  :linenos:
  :lines: 39-40

In the above ``ensemble_instance.add_ml_model`` code snippet, we offer the following arguments:

-  `name` ("cnn"): A name to reference the ML model in the ``Orchestrator``.
-  `backend` ("TF"): Indicating that the ML model is a TensorFlow model.
-  `model_path` (model_file): The path to the ML model script.
-  `device` ("GPU"): Specifying the device for ML model execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.
-  `inputs` (inputs): The name of the ML model input nodes (TensorFlow only).
-  `outputs` (outputs): The name of the ML model output nodes (TensorFlow only).

.. warning::
    Calling `exp.start(ensemble_instance)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When the ``Ensemble`` is started via ``Experiment.start``, the ML model will be loaded to the
launched ``Orchestrator``. The ML model can then be executed on the ``Orchestrator`` via a SmartSim
client (:ref:`SmartRedis<smartredis-api>`) within the application executable.

.. _TS_ensemble_doc:

TorchScripts
============
When configuring an ``Ensemble``, users can instruct SmartSim to load TorchScripts dynamically
to the ``Orchestrator``. The TorchScripts become available for each ``Ensemble`` member upon being loaded
into the ``Orchestrator`` prior to the execution of the ``Ensemble``. SmartSim users may upload
a single TorchScript function via ``Ensemble.add_function`` or alternatively upload a script
containing multiple functions via ``Ensemble.add_script``. To load a TorchScript to the
``Orchestrator``, SmartSim users can follow one of the following processes:

- :ref:`Define a TorchScript Function In-Memory<in_mem_TF_doc>`
   Use the ``Ensemble.add_function`` to instruct SmartSim to load an in-memory TorchScript to the ``Orchestrator``.
- :ref:`Define Multiple TorchScript Functions From File<TS_from_file>`
   Provide file path to ``Ensemble.add_script`` to instruct SmartSim to load the TorchScript from file to the ``Orchestrator``.
- :ref:`Define a TorchScript Function as String<TS_raw_string>`
   Provide function string to ``Ensemble.add_script`` to instruct SmartSim to load a raw string as a TorchScript function to the ``Orchestrator``.

.. note::
    Uploading a TorchScript :ref:`from memory<in_mem_TF_doc>` using ``Ensemble.add_function``
    is only supported for standalone ``Orchestrators``. Users uploading
    TorchScripts to colocated ``Orchestrators`` should instead use the function ``Ensemble.add_script``
    to upload :ref:`from file<TS_from_file>` or as a :ref:`string<TS_raw_string>`.

Each function also provides flexible device selection, allowing users to choose between which device the TorchScript is executed on, `"GPU"` or `"CPU"`.
In environments with multiple devices, specific device numbers can be specified using the
`devices_per_node` parameter.

.. note::
    If `device=GPU` is specified when attaching a TorchScript function to an ``Ensemble``, this instructs
    SmartSim to execute the TorchScript on GPU nodes. However, TorchScripts loaded to an ``Orchestrator`` are
    executed on the ``Orchestrator`` compute resources. Therefore, users must make sure that the device
    specified is included in the ``Orchestrator`` compute resources. To further explain, if a user
    specifies `device=GPU`, however, initializes ``Orchestrator`` on only CPU nodes,
    the TorchScript will not run on GPU nodes as advised.

Continue or select the respective process link to learn more on how each function (``Ensemble.add_script`` and ``Ensemble.add_function``)
dynamically loads TorchScripts to the ``Orchestrator``.

.. seealso::
    To add a TorchScript to a single ``Model`` that will be appended to an
    ``Ensemble``, refer to the :ref:`TorchScripts<TS_doc>`
    section of the ``Model`` documentation.

.. _in_mem_TF_ensemble_doc:

-------------------------------
Attach an In-Memory TorchScript
-------------------------------
Users can define TorchScript functions within the ``Experiment`` driver script
to attach to an ``Ensemble``. This feature is supported by ``Ensemble.add_function``.

.. warning::
    ``Ensemble.add_function`` does **not** support loading in-memory TorchScript functions to a colocated ``Orchestrator``.
    If you would like to load a TorchScript function to a colocated ``Orchestrator``, define the function
    as a :ref:`raw string<TS_raw_string_ensemble>` or :ref:`load from file<TS_from_file_ensemble>`.

When specifying an in-memory TF function using ``Ensemble.add_function``, the
following arguments are offered:

- `name` (str): reference name for the script inside of the ``Orchestrator``.
- `function` (t.Optional[str] = None): TorchScript function code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

.. _in_mem_TF_ex:

Example: Load a In-Memory TorchScript Function
----------------------------------------------
This example walks through the steps of instructing SmartSim to load an in-memory TorchScript function
to a standalone ``Orchestrator``. The source code example is available in the dropdown below for
convenient execution and customization.

.. dropdown:: Experiment Driver Script Source Code

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_torchscript_mem.py

.. note::
    The example assumes:

    - a standalone ``Orchestrator`` is launched prior to ``Ensemble`` execution
    - an initialized ``Ensemble`` named `ensemble_instance` exists within the ``Experiment`` workflow

**Define an In-Memory TF Function**

To begin, define an in-memory TorchScript function within the Python driver script.
For the purpose of the example, we add a simple TorchScript function, `timestwo`:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_torchscript_mem.py
  :language: python
  :linenos:
  :lines: 3-4

**Attach the In-Memory TorchScript Function to a SmartSim Ensemble**

We use the ``Ensemble.add_function`` function to instruct SmartSim to load the TorchScript function `timestwo`
onto the launched standalone ``Orchestrator``. Specify the function `timestwo` to the `function`
parameter:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_torchscript_mem.py
  :language: python
  :linenos:
  :lines: 15-16

In the above ``ensemble_instance.add_function`` code snippet, we offer the following arguments:

-  `name` ("example_func"): A name to uniquely identify the TorchScript within the ``Orchestrator``.
-  `function` (timestwo): Name of the TorchScript function defined in the Python driver script.
-  `device` ("GPU"): Specifying the device for TorchScript execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(ensemble_instance)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the TorchScript to a non-existent ``Orchestrator``.

When the ``Ensemble`` is started via ``Experiment.start``, the TF function will be loaded to the
standalone ``Orchestrator``. The function can then be executed on the ``Orchestrator`` via a SmartSim
client (:ref:`SmartRedis<smartredis-api>`) within the application code.

.. _TS_from_file_ensemble:

------------------------------
Attach a TorchScript From File
------------------------------
Users can attach TorchScript functions from a file to an ``Ensemble`` and upload them to a
colocated or standalone ``Orchestrator``. This functionality is supported by the ``Ensemble.add_script``
function's `script_path` parameter.

When specifying a TorchScript using ``Ensemble.add_script``, the
following arguments are offered:

- `name` (str): Reference name for the script inside of the ``Orchestrator``.
- `script` (t.Optional[str] = None): TorchScript code (only supported for non-colocated ``Orchestrators``).
- `script_path` (t.Optional[str] = None): path to TorchScript code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

Example: Loading a TorchScript From File
----------------------------------------
This example walks through the steps of instructing SmartSim to load a TorchScript from file
to an ``Orchestrator``. The source code example is available in the dropdown below for
convenient execution and customization.

.. dropdown:: Experiment Driver Script Source Code

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_torchscript_file.py

.. note::
    This example assumes:

    - an ``Orchestrator`` is launched prior to ``Ensemble`` execution
    - an initialized ``Ensemble`` named `ensemble_instance` exists within the ``Experiment`` workflow

**Define a TorchScript Script**

For the example, we create the Python script `torchscript.py`. The file contains multiple
simple torch function shown below:

.. code-block:: python

    def negate(x):
        return torch.neg(x)

    def random(x, y):
        return torch.randn(x, y)

    def pos(z):
        return torch.positive(z)

**Attach the TorchScript Script to a SmartSim Ensemble**

Assuming an initialized ``Ensemble`` named `ensemble_instance` exists, we add a TorchScript script using
the ``Ensemble.add_script`` function and specify the script path to the parameter `script_path`:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_torchscript_mem.py
  :language: python
  :linenos:
  :lines: 12-13

In the above ``smartsim_model.add_script`` code snippet, we offer the following arguments:

-  `name` ("example_script"): Reference name for the script inside of the ``Orchestrator``.
-  `script_path` ("path/to/torchscript.py"): Path to the script file.
-  `device` ("GPU"): device for script execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(ensemble_instance)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When `ensemble_instance` is started via ``Experiment.start``, the TorchScript will be loaded from file to the
``Orchestrator`` that is launched prior to the start of `ensemble_instance`.

.. _TS_raw_string_ensemble:

---------------------------------
Define TorchScripts as Raw String
---------------------------------
Users can upload TorchScript functions from string to send to a colocated or
standalone ``Orchestrator``. This feature is supported by the
``Ensemble.add_script`` function's `script` parameter.

When specifying a TorchScript using ``Ensemble.add_script``, the
following arguments are offered:

- `name` (str): Reference name for the script inside of the ``Orchestrator``.
- `script` (t.Optional[str] = None): String of function code (e.g. TorchScript code string).
- `script_path` (t.Optional[str] = None): path to TorchScript code.
- `device` (t.Literal["CPU", "GPU"] = "CPU"): device for script execution, defaults to “CPU”.
- `devices_per_node` (int = 1): The number of GPU devices available on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.
- `first_device` (int = 0): The first GPU device to use on the host. This parameter only applies to GPU devices and will be ignored if device is specified as CPU.

Example: Load a TorchScript From String
---------------------------------------
This example walks through the steps of instructing SmartSim to load a TorchScript function
from string to an ``Orchestrator`` before the execution of the associated ``Ensemble``.
The source code example is available in the dropdown below for convenient execution and customization.

.. dropdown:: Experiment Driver Script Source Code

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_torchscript_string.py

.. note::
    This example assumes:

    - an ``Orchestrator`` is launched prior to ``Ensemble`` execution
    - an initialized ``Ensemble`` named `ensemble_instance` exists within the ``Experiment`` workflow

**Define a String TorchScript**

Define the TorchScript code as a variable in the Python driver script:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_torchscript_string.py
  :language: python
  :linenos:
  :lines: 12-13

**Attach the TorchScript Function to a SmartSim Ensemble**

Assuming an initialized ``Ensemble`` named `ensemble_instance` exists, we add a TorchScript using
the ``Ensemble.add_script`` function and specify the variable `torch_script_str` to the parameter
`script`:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/ensemble_torchscript_string.py
  :language: python
  :linenos:
  :lines: 15-16

In the above ``ensemble_instance.add_script`` code snippet, we offer the following arguments:

-  `name` ("example_script"): key to store script under.
-  `script` (torch_script_str): TorchScript code.
-  `device` ("GPU"): device for script execution.
-  `devices_per_node` (2): Use two GPUs per node.
-  `first_device` (0): Start with 0 index GPU.

.. warning::
    Calling `exp.start(ensemble_instance)` prior to instantiation of an ``Orchestrator`` will result in
    a failed attempt to load the ML model to a non-existent ``Orchestrator``.

When the ``Ensemble`` is started via ``Experiment.start``, the TorchScript will be loaded to the
``Orchestrator`` that is launched prior to the start of the ``Ensemble``.

.. _prefix_ensemble:

=========================
Data Collision Prevention
=========================
Overview
========
When multiple ``Ensemble`` members use the same code to send and access their respective data
in the ``Orchestrator``, key overlapping can occur, leading to inadvertent data access
between ``Ensemble`` members. To address this, SmartSim supports key prefixing
through ``Ensemble.enable_key_prefixing`` which enables key prefixing for all
``Ensemble`` members. For example, during an ``Ensemble`` simulation with prefixing enabled, SmartSim will add
the ``Ensemble`` member `name` as a prefix to the keys sent to the ``Orchestrator``.
Enabling key prefixing eliminates issues related to key overlapping, allowing ``Ensemble``
members to use the same code without issue.

The key components of SmartSim ``Ensemble`` prefixing functionality include:

1. **Sending Data to the Orchestrator**: Users can send data to an ``Orchestrator``
   with the ``Ensemble`` member name prepended to the data name by utilizing SmartSim :ref:`Ensemble functions<model_prefix_func_ensemble>`.
2. **Retrieving Data From the Orchestrator**: Users can instruct a ``Client`` to prepend a
   ``Ensemble`` member name to a key during data retrieval, polling, or check for existence on the ``Orchestrator``
   through SmartRedis :ref:`Client functions<client_prefix_func>`. However, entity interaction
   must be registered using :ref:`Ensemble<model_prefix_func_ensemble>` or :ref:`Model<model_prefix_func>` functions.

.. seealso::
    For information on prefixing ``Client`` functions, visit the :ref:`Client functions<client_prefix_func>` page of the ``Model``
    documentation.

For example, assume you have an ``Ensemble`` that was initialized using the :ref:`replicas<replicas_init>` creation strategy.
Two identical ``Model`` were created named `ensemble_0` and `ensemble_1` that use the same executable application
within an ``Ensemble`` named `ensemble`. In the application code you use the function ``Client.put_tensor("tensor_0", data)``.
Without key prefixing enabled, the slower member will overwrite the data from the faster simulation.
With ``Ensemble`` key prefixing turned on, `ensemble_0` and `ensemble_1` can access
their tensor `"tensor_0"` by name without overwriting or accessing the other ``Model``'s `"tensor_0"` tensor.
In this scenario, the two tensors placed in the ``Orchestrator`` are named `ensemble_0.tensor_0` and `ensemble_1.tensor_0`.

.. _model_prefix_func_ensemble:

------------------
Ensemble Functions
------------------
An ``Ensemble`` object supports two prefixing functions: ``Ensemble.enable_key_prefixing`` and
``Ensemble.register_incoming_entity``. For more information on each function, reference the
:ref:`Ensemble API docs<ensemble_api>`.

To enable prefixing on a ``Ensemble``, users must use the ``Ensemble.enable_key_prefixing``
function in the ``Experiment`` driver script. This function activates prefixing for tensors,
``Datasets``, and lists sent to an ``Orchestrator`` for all ``Ensemble`` members. This function
also enables access to prefixing ``Client`` functions within the ``Ensemble`` members. This excludes
the ``Client.set_data_source`` function, where ``enable_key_prefixing`` is not require for access.

.. note::
    ML model and script prefixing is not automatically enabled through ``Ensemble.enable_key_prefixing``.
    Prefixing must be enabled within the ``Ensemble`` by calling the ``use_model_ensemble_prefix`` method
    on the ``Client`` embedded within the member application.

Users can enable the SmartRedis ``Client`` to interact with prefixed data, ML models and TorchScripts
using the ``Client.set_data_source``. However, for SmartSim to recognize the producer entity name
passed to the function within an application, the producer entity must be registered on the consumer
entity using ``Ensemble.register_incoming_entity``.

If a consumer ``Ensemble`` member requests data sent to the ``Orchestrator`` by other ``Ensemble`` members, the producer members must be
registered on consumer member. To access ``Ensemble`` members, SmartSim offers the attribute ``Ensemble.models`` that returns
a list of ``Ensemble`` members. Below we demonstrate registering producer members on a consumer member:

.. code-block:: python

    # list of producer Ensemble members
    list_of_ensemble_names = ["producer_0", "producer_1", "producer_2"]

    # Grab the consumer Ensemble member
    ensemble_member = ensemble.models.get("producer_3")
    # Register the producer members on the consumer member
    for name in list_of_ensemble_names:
        ensemble_member.register_incoming_entity(ensemble.models.get(name))

For examples demonstrating how to retrieve data within the entity application that produced
the data, visit the ``Model`` :ref:`Copy/Rename/Delete Operations<copy_rename_del_prefix>` subsection.

Example: Ensemble Key Prefixing
===============================
In this example, we create an ``Ensemble`` comprised of two ``Model``'s that use identical code
to send data to a standalone ``Orchestrator``. To prevent key collisions and ensure data
integrity, we enable key prefixing on the ``Ensemble`` which automatically
appends the ``Ensemble`` member `name` to the data sent to the ``Orchestrator``. After the
``Ensemble`` completes, we launch a consumer ``Model`` within the ``Experiment`` driver script
to demonstrate accessing prefixed data sent to the ``Orchestrator`` by ``Ensemble`` members.

This example consists of **three** Python scripts:

1. :ref:`Application Producer Script<app_prod_prefix_ensemble>`: This script is encapsulated
   in a SmartSim ``Ensemble`` within the ``Experiment`` driver script. Prefixing is enabled
   on the ``Ensemble``. The producer script puts NumPy tensors on an ``Orchestrator``
   launched in the ``Experiment`` driver script. The ``Ensemble`` creates two
   identical ``Ensemble`` members. The producer script is executed
   in both ``Ensemble`` members to send two prefixed tensors to the ``Orchestrator``.
   The source code example is available in the dropdown below for convenient customization.

.. dropdown:: Application Producer Script Source Code

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/application_producer_script.py

1. :ref:`Application Consumer Script<app_con_prefix_ensemble>`: This script is encapsulated
   within a SmartSim ``Model`` in the ``Experiment`` driver script. The script requests the
   prefixed tensors placed by the producer script. The source code example is available in
   the dropdown below for convenient customization.

.. dropdown:: Application Consumer Script Source Code

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/application_consumer_script.py

1. :ref:`Experiment Driver Script<exp_prefix_ensemble>`: The driver script launches the
   ``Orchestrator``, the ``Ensemble`` (which sends prefixed keys to the ``Orchestrator``),
   and the ``Model`` (which requests prefixed keys from the ``Orchestrator``). The
   ``Experiment`` driver script is the centralized spot that controls the workflow.
   The source code example is available in the dropdown below for convenient execution and
   customization.

.. dropdown:: Experiment Driver Script Source Code

    .. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py

.. _app_prod_prefix_ensemble:

-------------------------------
The Application Producer Script
-------------------------------
In the ``Experiment`` driver script, we instruct SmartSim to create an ``Ensemble`` comprised of
two duplicate members that execute this producer script. In the producer script, a SmartRedis ``Client`` sends a
tensor to the ``Orchestrator``. Since the ``Ensemble`` members are identical and therefore use the same
application code, two tensors are sent to the ``Orchestrator``. Without prefixing enabled on the ``Ensemble``
the keys can be overwritten. To prevent this, we enable key prefixing on the ``Ensemble`` in the driver script
via ``Ensemble.enable_key_prefixing``. When the producer script is executed by each ``Ensemble`` member, a
tensor is sent to the ``Orchestrator`` with the ``Ensemble`` member `name` prepended to the tensor `name`.

Here we provide the producer script that is applied to the ``Ensemble`` members:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/application_producer_script.py
  :language: python
  :linenos:

After the completion of ``Ensemble`` members `producer_0` and `producer_1`, the contents of the ``Orchestrator`` are:

.. code-block:: bash

    1) "producer_0.tensor"
    2) "producer_1.tensor"

.. _app_con_prefix_ensemble:

-------------------------------
The Application Consumer Script
-------------------------------
In the ``Experiment`` driver script, we initialize a consumer ``Model`` that encapsulates
the consumer application to request the tensors produced from the ``Ensemble``. To do
so, we use SmartRedis key prefixing functionality to instruct the SmartRedis ``Client``
to append the name of an ``Ensemble`` member to the key `name`.

.. seealso::
    For more information on ``Client`` prefixing functions, visit the :ref:`Client functions<client_prefix_func>`
    subsection of the ``Model`` documentation.

To begin, specify the imports and initialize a SmartRedis ``Client``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/application_consumer_script.py
  :language: python
  :linenos:
  :lines: 1-4

To retrieve the tensor from the first ``Ensemble`` member named `producer_0`, use
``Client.set_data_source``. Specify the name of the first ``Ensemble`` member
as an argument to the function. This instructs SmartSim to append the ``Ensemble`` member name to the data
search on the ``Orchestrator``. When ``Client.poll_tensor`` is executed,
the SmartRedis `client` will poll for key, `producer_0.tensor`:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/application_consumer_script.py
  :language: python
  :linenos:
  :lines: 6-9

Follow the same steps above, however, change the data source `name` to the `name`
of the second ``Ensemble`` member (`producer_1`):

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/application_consumer_script.py
  :language: python
  :linenos:
  :lines: 11-14

We print the boolean return to verify that the tensors were found:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/application_consumer_script.py
  :language: python
  :linenos:
  :lines: 16-17

When the ``Experiment`` driver script is executed, the following output will appear in `consumer.out`:

.. code-block:: bash

    Default@11-46-05:producer_0.tensor was found: True
    Default@11-46-05:producer_1.tensor was found: True

.. warning::
    For SmartSim to recognize the ``Ensemble`` member names as a valid data source
    to ``Client.set_data_source``, you must register each ``Ensemble`` member
    on the consumer ``Model`` in the driver script via ``Model.register_incoming_entity``.
    We demonstrate this in the ``Experiment`` driver script section of the example.

.. _exp_prefix_ensemble:

---------------------
The Experiment Script
---------------------
The ``Experiment`` driver script manages all workflow components and utilizes the producer and consumer
application scripts. In the example, the ``Experiment``:

- launches standalone ``Orchestrator``
- launches an ``Ensemble`` via the replicas initialization strategy
- launches a consumer ``Model``
- clobbers the ``Orchestrator``

To begin, add the necessary imports, initialize an ``Experiment`` instance and initialize the
standalone ``Orchestrator``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 1-9

We are now setup to discuss key prefixing within the ``Experiment`` driver script.
To create an ``Ensemble`` using the replicas strategy, begin by initializing a ``RunSettings``
object to apply to all ``Ensemble`` members. Specify the path to the application
producer script:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 11-12

Next, initialize an ``Ensemble`` by specifying `ensemble_settings` and the number of ``Model`` `replicas` to create:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 14-15

Instruct SmartSim to prefix all tensors sent to the ``Orchestrator`` from the ``Ensemble`` via ``Ensemble.enable_key_prefixing``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 17-18

Next, initialize the consumer ``Model``. The consumer ``Model`` application requests
the prefixed tensors produced by the ``Ensemble``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 20-23

Next, organize the SmartSim entity output files into a single ``Experiment`` folder:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 25-26

Launch the ``Orchestrator``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 28-29

Launch the ``Ensemble``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 31-32

Set `block=True` so that ``Experiment.start`` waits until the last ``Ensemble`` member has finished before continuing.

The consumer ``Model`` application script uses ``Client.set_data_source`` which
accepts the ``Ensemble`` member names when searching for prefixed
keys in the ``Orchestrator``. In order for SmartSim to recognize the ``Ensemble``
member names as a valid data source in the consumer ``Model``, we must register
the entity interaction:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 34-36

Launch the consumer ``Model``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 38-39

To finish, tear down the standalone ``Orchestrator``:

.. literalinclude:: tutorials/doc_examples/ensemble_doc_examples/experiment_driver.py
  :language: python
  :linenos:
  :lines: 41-42