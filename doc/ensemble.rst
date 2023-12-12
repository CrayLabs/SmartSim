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
There are five different ways to initialize an ``Ensemble`` object using the
given parameter options.

.. autosummary::

   Ensemble.__init__

To prepare to discuss each case, lets first walk through each of the
above arguments.

1. `params`: Use the `params` argument to specify values used within the
   application. This argument will expand all specified parameters to
   all models. For example, if you wanted to replicate the `THERMO` for
   use across all Models within an Ensemble, you could set `params="THERMO"`.
   To set this argument, you could either use the function
   Ensemble.attach_generator_files(`to_configure`) or set the function
   `params_as_args` with the values you would like.
2. `params_as_args`: You can use this parameter to set the `params` used
   within the Models. For example, if I wanted to specify setting `THERMO`
   to `1,2,3,4` then I would pass in a dictionary to `params_as_args` as
   `params_as_args={"THERMO": [1,2,3,4]}`. You can then use the argument
   `perm_strategy` to determine how to set the `THERMO` value.
3. `batch_settings`: This parameter encompasses the deployment of the whole
   Ensemble. For example, if an Ensemble was composed of 100 Models, each
   requiring 10 nodes per simulation. Then the batch settings would specify
   that 100 nodes are required for the entire ensemble. Through a run settings
   object would you specify that the model requires 10 nodes.
4. `run_settings`: This parameter determines the run settings for each
   Model. This object is required per Model but can be added in different ways
   as will be demonstrated in the cases.
5. `replicas`: This argument is used when you would like to create clones of
   a Model. For example, if I would like to create eight clones of the same
   Model, I would specify this parameter equal to eight. This parameter is used
   in certain cases which will be demonstrated further below.
6. `perm_strategy`: This parameter is used to determine how the `params_as_args`
   will be distributed. There are three options for this argument: `all_perm`, `step`
   and `random`. If you would like to use all possible combinations of the pass in `params_as_args`,
   then `all_perm` is used. If you would like to step through each parameter passed
   into `params_as_args`, then specify `step`. If you would like to randomly select from
   all possible combinations, then use `random`. When using `random`, you can specify the
   number of random times to sample by specifying the `replicas` argument.

Now lets walk through the possible cases:
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

Case 2 : ``RunSettings`` and `replicas`
    This case is for creating an Ensemble of Model objects that
    use the same run settings and do not contain parameter values.
    Specify the number of Models in the Ensemble through the argument `replicas`. This
    case is used when parameters do not need to be set within an Ensemble.
    Therefore, we need to specify how many Models should be created within
    an Ensemble which is done via `replicas`.

Case 3 : ``BatchSettings``
    In this init strategy, you are able to manually add Models to the Ensemble
    and submit as a batch job. Each Model here may have different `params`
    and `param_as_args` values. This case is used in Ensemble workloads that require
    `Models` that work together. Learn to append a model here <link>.

Case 4 : ``RunSettings``, ``BatchSettings`` and `params`
    This init strategy is used when you would like to submit the Ensemble as a batch job,
    but would also like all `Models` to have the same run settings and `params`. You may
    use the `param_as_args` argument when initializing the Ensemble to determine the `params`
    values. The number of Models created is determined by the `perm_strategy` argument, as
    different permutations of the `params` values will be fed to each model.

Case 5 : ``RunSettings``, ``BatchSettings`` and `replicas`
    This init strategy is used when you would like to submit the Ensemble as a batch job,
    but would also like all `Models` to have the same run settings. You may determine
    the number of `Models` within the `Ensemble` through the `replicas` argument. This
    case is used during an Ensemble that runs the same simulations, however, the simulations
    produce different outputs.

=========
Appending
=========
ensemble.add_model()

=====================
ML Models and Scripts
=====================
--------
Overview
--------
An Ensemble object provides a subset of helper functions
that enable AI and Machine Learning within the Ensemble. In this section
we will demonstrate how to load a TF, TF-lite, PT, or ONNX model into the DB at runtime,
add a TorchScript function to launch with ensemble entities,
and load a TorchScript to launch with the ensembles Model entities.

AI Models
---------
The ``Ensemble.add_ml_model()`` helper function adds a
TF, TF-lite, PT, or ONNX model to load into the database at runtime.
Each ML Model added will be loaded into an orchestrator (converged or not)
prior to the execution of every entity belonging to the ensemble.

When using the ``add_script()`` function, you may specify params:

1. `name` (str) : key to store model under
2. `model` (str | bytes | None) : model name in memory
3. `model_path` (str) : file path to the serialized model
4. `backend` (str) : name of the model backend (TORCH, TF, TFLITE, ONNX)
5. `device` (str, optional) : name of device for execution, defaults to “CPU”
6. `batch_size` (int, optional) : batch size for execution, defaults to 0
7. `min_batch_size` (int, optional) : minimum batch size for model execution, defaults to 0
8. `tag` (str, optional) : additional tag for model information, defaults to “”
9.  `inputs` (list[str], optional) : names of model inputs (TF only), defaults to None
10. `outputs` (list[str], optional) : names model outputs (TF only), defaults to None

TorchScript functions
---------------------
The ``Ensemble.add_function()`` helper function adds a
TorchScript function to launch with every entity
belonging to the ensemble. Each function added
is loaded into a non-converged orchestrator prior to the execution of any
of the ensemble members. For converged orchestrators,
the ``add_script()``<link> method should be used.

When using the ``add_script()`` function, you may specify params:

1. `name`  (str) : key to store function under
2. `function` (str, optional) : TorchScript code
3. `device`  (str, optional) : device for script execution, defaults to “CPU”
4. `devices_per_node` (int) : assign the number of CPU's or GPU's to use on the node

TorchScript Scripts
-------------------
The ``Ensemble.add_script()`` helper function adds a TorchScript scripts to
launch with every Model within an Ensemble. Each script added
is loaded into an  (converged or not) orchestrator prior to the execution of any
of the ensemble members.

When using the ``add_script()`` function, you may specify params:

1. `name`  (str) : key to store script under
2. `script` (str, optional) : TorchScript code
3. `script_path` (str, optional) : file path to TorchScript code
4. `device`  (str, optional) : device for script execution, defaults to “CPU”
5. `devices_per_node` (int) : assign the number of CPU's or GPU's to use on the node