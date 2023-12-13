********
Ensemble
********

========
Overview
========
Ensemble is a group of Model instances that can be treated as a reference to a single instance.
An ``Ensemble`` represents a group of simulations and can be initialized in a number of ways
demonstrated in the initialize section. By initializing an Ensemble, a user has access to the
Ensemble API which has a number of helper functions that:

1. add models to the Ensemble
2. add ml models to the Ensemble that are loaded to the db at runtime
3. add script/functions to the ensemble that are launched in memory
4. attach generator files to the model
5. enable key prefixing within the Ensemble
6. register communication between entities

In the following Ensemble sections we discuss:

1. Initializing an Ensemble <link>
2. Appending Models to an Ensemble <link>
3. Loading AI Models and Scripts to an Ensemble <link>
4. Data Collision prevention <link>
5. Starting an Ensemble <link>
6. Stopping an Ensemble <link>

==========
Initialize
==========

.. list-table:: title
   :widths: auto
   :header-rows: 1

   * - input
     - Description
   * - `params`
     - Parameters to expand into ``Model`` members. For example, if you
       wanted to assign a parameter used within all Ensemble members,
       you must first specify the argument to `params` like
       `params="THERMO"`. You may then assign the value of this via the
       ``Ensemble.attach_generator_file(to_config)`` function or set the
       value in `params_as_args`.
   * - `params_as_args`
     - A list of params that should be used as command line arguments
       to the ``Model`` member executables and not written to generator files.
       For example, you have specified to `params` that the variable "THERMO"
       is used across all ``Models``. To set the variable, assign like so:
       `params_as_args={"THERMO": [95,100,105]}`.
   * - `batch_settings`
     - Describes settings for Ensemble as batch workload. This encompasses the
       entire ``Ensemble``. For example, if an Ensemble needed to support
       two ``Models`` that both required 5 nodes. Then we would specify that
       the batch settings needs 10 nodes for the Ensemble.
   * - `run_settings`
     - Describes how each Model should be executed. The run settings will be applied to
       all ``Models``.
   * - `replicas`
     - Represents the number of ``Model`` clones to create within an Ensemble. For example,
       `replicas=4` specifies to the Ensemble API to create four of the same Models.
   * - `perm_strategy`
     - This parameter is used to determine how the `params_as_args`
       will be distributed. There are three options for this argument: `all_perm`, `step`
       and `random`. If you would like to use all possible combinations of the pass in `params_as_args`,
       then `all_perm` is used. If you would like to step through each parameter passed
       into `params_as_args`, then specify `step`. If you would like to randomly select from
       all possible combinations, then use `random`. When using `random`, you can specify the
       number of random times to sample by specifying the `replicas` argument.

Ensembles require one of the following combinations of arguments:

Case 1 : ``RunSettings`` and `params`
    This case is for initializing an Ensemble of Models with all the
    same run settings and parameters. You may want to use this case
    if you are running a multiple of the same simulation with the same values.
    The `params` specified are either set via a text file through the
    ``Ensemble.attach_generator_file(to_configure)`` function or the
    `params_as_args` argument. You may specify a string to `params_as_args` or
    a dictionary depending on the number of arguments.
    The number of Models created here is determined by the
    `perm_strategy` function, as the number of Models will correspond with
    the number of value groups to input.

.. code-block:: python

    test

Case 2 : ``RunSettings`` and `replicas`
    This case is for creating an Ensemble of Model objects that
    use the same run settings and do not contain parameter values.
    Specify the number of Models in the Ensemble through the argument `replicas`. This
    case is used when parameters do not need to be set within an Ensemble.
    Therefore, we need to specify how many Models should be created within
    an Ensemble which is done via `replicas`.

.. code-block:: python

    test

Case 3 : ``BatchSettings``
    In this init strategy, you are able to manually add Models to the Ensemble
    and submit as a batch job. Each Model here may have different `params`
    and `param_as_args` values. This case is used in Ensemble workloads that require
    `Models` that work together. Learn to append a model here <link>.

.. code-block:: python

    test

Case 4 : ``RunSettings``, ``BatchSettings`` and `params`
    This init strategy is used when you would like to submit the Ensemble as a batch job,
    but would also like all `Models` to have the same run settings and `params`. You may
    use the `param_as_args` argument when initializing the Ensemble to determine the `params`
    values. The number of Models created is determined by the `perm_strategy` argument, as
    different permutations of the `params` values will be fed to each model.

.. code-block:: python

    test

Case 5 : ``RunSettings``, ``BatchSettings`` and `replicas`
    This init strategy is used when you would like to submit the Ensemble as a batch job,
    but would also like all `Models` to have the same run settings. You may determine
    the number of `Models` within the `Ensemble` through the `replicas` argument. This
    case is used during an Ensemble that runs the same simulations, however, the simulations
    produce different outputs.

.. code-block:: python

    test

=========
Appending
=========
SmartSim allows users to manually append Models to an Ensemble.
This functionality is useful when an Ensemble workload requires
the diversity of models. For example, an ensemble workload might
require an *Ensemble of Experts*. In this case, different models specialize
in different subtasks or aspects of the problem. The ensemble then
combines their predictions to achieve a more robust and accurate overall prediction.

In the following section, we walk through adding Model objects
to the Ensemble. Init **Case 3** above mentions that an Ensemble
initialized solely with a ``BatchSettings`` object requires
that Models be manually appended. To demonstrate this, we
follow case 3 to create the Ensemble in the example.

.. note::
    This example assumes that you have created an Experiment and
    are adding this code to the Experiment driver script.
    Remember that you only have access to the Ensemble, Model
    and BatchSettings API (used in this example)
    through the Experiment factory class. Our experiment object
    will be named ``exp``.

Later, we will create 2 Models that both utilize 5 nodes.
We are submitting the Ensemble as a batch job, therefore,
when initializing a ``BatchSettings`` object, specify that the batch
job will require 10 nodes:

.. code-block:: python

    sbatch_settings = exp.create_batch_settings(nodes=10)

Now initialize the Ensemble using the ``Experiment.create_ensemble()``
factory method and specify the `sbatch_settings` object:

.. code-block:: python

    ensemble = exp.create_ensemble(sbatch_settings)

Now that the empty Ensemble is initialized, begin taking steps to
create the two Models to append to the Ensemble. Start by creating
the model run settings. A Model object requires a ``RunSettings`` object,
or instructions on how to execute the Model. Below, we create two run settings
objects for `model_1` and `model_2`:

.. code-block:: python

    srun_settings_1 = exp.create_run_settings(exe=exe, exe_args="path/to/script_1")
    srun_settings_2 = exp.create_run_settings(exe=exe, exe_args="path/to/script_2")

Initialize the first Model using ``Experiment.create_model()``:

.. code-block:: python

    model_1 = exp.create_model(name="model_1", run_settings=srun_settings_1, params={"THERMO":[95,100]})

Above, we specify Model parameters that are used within the application script via the `params`
argument. In the application script, we set the parameter "THERMO_1" to a list of integers.

We specify the `params` argument to `model_2`, again passing in a list of integers. The idea is
that `model_1` and `model_2` are both different scripts that have the same end goal. They both use
the same "THERMO" argument, however, we would like to compare the outputs of both Models.
Initialize `model_2`:

.. code-block:: python

    model_2 = exp.create_model(name="model_2", run_settings=srun_settings_2, params={"THERMO":[95,100]})

The Ensemble API has a helper function named ``Ensemble.add_model()`` that accepts model
entities to add to an Ensemble:

.. code-block:: python

    ensemble.add_model(model_1)
    ensemble.add_model(model_2)

Now that we have added the models to the Ensemble, we can start the Ensemble via
``Experiment.start()``:

.. code-block:: python

    exp.start(ensemble)

=====================
ML Models and Scripts
=====================
--------
Overview
--------
Smartsim enables users to build diverse ensembles that leverage
the strengths of different model types.
Users may integrate TorchScript functions, scripts, and
TF, TF-lite, PT, or ONNX models within an ensemble workload.
TorchScript provides a set of tools and functionalities that enhance
the utility of PyTorch models within ensembles,
offering benefits in terms of performance, deployment, interoperability, and
composition of diverse model architectures.

The Ensemble API provides a subset of helper functions that support
adding TorchScript functions, scripts and Machine Learning models to
an Ensemble:

* ``Ensemble.add_ml_model()`` : Load a TF, TF-lite, PT, or ONNX model into the DB at runtime.
* ``Ensemble.add_function()`` : Launch a TorchScript function with each ensemble member.
* ``Ensemble.add_script()`` : Launch a TorchScript with each ensemble member.

In this following subsections, we discuss each helper function as well as provide examples for
each.

AI Models
---------
The ``Ensemble.add_ml_model()`` helper function adds
TF, TF-lite, PT, or ONNX models to the ensemble. Each model added
will be loaded into the (colocated or standard) database at runtime
prior to the execution of each entity belonging to the ensemble.

This function offers the following arguments:

1. `name` (str) : key to store model under
2. `model` (str | bytes | None) : model name in memory
3. `model_path` (str) : file path to the serialized model
4. `backend` (str) : name of the model backend (TORCH, TF, TF-LITE, ONNX)
5. `device` (str, optional) : name of device for execution, defaults to “CPU”
6. `batch_size` (int, optional) : batch size for execution, defaults to 0
7. `min_batch_size` (int, optional) : minimum batch size for model execution, defaults to 0
8. `tag` (str, optional) : additional tag for model information, defaults to “”
9.  `inputs` (list[str], optional) : names of model inputs (TF only), defaults to None
10. `outputs` (list[str], optional) : names model outputs (TF only), defaults to None

.. code-block:: python

    def create_tf_cnn():
        """Create a Keras CNN for testing purposes

        """
        from smartsim.ml.tf import serialize_model
        n = Net()
        input_shape = (3,3,1)
        inputs = Input(input_shape)
        outputs = n(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs, name=n.name)

        return serialize_model(model)

.. code-block:: python

    run_settings = exp.create_run_settings(
        exe=sys.executable,
        exe_args=sr_test_script
    )

.. code-block:: python

    smartsim_ensemble = exp.create_ensemble("smartsim_model", run_settings=run_settings, replicas=2)

.. code-block:: python

    smartsim_ensemble.add_ml_model("cnn", "TF", model=model, device="CPU", inputs=inputs, outputs=outputs)

.. code-block:: python

    db = exp.create_database(port=6780, interface="lo")

.. code-block:: python

    exp.start(db, smartsim_ensemble,  block=True)

TorchScript functions
---------------------
The ``Ensemble.add_function()`` helper function adds a
TorchScript function to launch with every Model entity
belonging to the ensemble. Each function added
is loaded into a colocated orchestrator prior to the execution of any
of the ensemble members. For standard orchestrators,
the ``add_script()``<link> method should be used.

This function offers the following arguments:

1. `name`  (str) : key to store function under
2. `function` (str, optional) : TorchScript code
3. `device`  (str, optional) : device for script execution, defaults to “CPU”
4. `devices_per_node` (int) : assign the number of CPU's or GPU's to use on the node

.. code-block:: python

    def timestwo(x):
        return 2*x

.. code-block:: python

    ensemble.add_function("test_func", function=timestwo, device="CPU")

TorchScript Scripts
-------------------
The ``Ensemble.add_script()`` helper function adds a TorchScript script to
launch with every Model within an Ensemble. Each script added
is loaded into an orchestrator (colocated or standard) prior to the execution of any
of the ensemble members.

When using the ``add_script()`` function, you may specify params:

1. `name`  (str) : key to store script under
2. `script` (str, optional) : TorchScript code
3. `script_path` (str, optional) : file path to TorchScript code
4. `device`  (str, optional) : device for script execution, defaults to “CPU”
5. `devices_per_node` (int) : assign the number of CPU's or GPU's to use on the node

You might use TorchScript scripts to represent individual models within the ensemble:

.. code-block:: python

    torch_script_str = "def negate(x):\n\treturn torch.neg(x)\n"

.. code-block:: python

    ensemble.add_script("test_script1", script_path=torch_script, device="CPU")

=========================
Data Collision Prevention
=========================