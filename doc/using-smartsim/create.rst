**********************
Creating an Experiment
**********************


Creating an Experiment in SmartSim is as simple as instaniating one Python
object. There are only three arguments to the Experiment.

.. automethod:: smartsim.Experiment.__init__

The simplest initialization of an ``Experiment`` is as follows:

.. code-block:: python
    :linenos:

    from smartsim import Experiment
    exp = Experiment("name-of-experiment", launcher="local") # local backend

In the code above, the Experiment is intialized to launch all entities locally
and if generated, will place all files neatly in a folder named ``name-of-experiment``.

.. note::
    Do not include spaces, or special characters in the names of either experiments, or entities
    created by the experiment.


Initializing Entities
=====================

``SmartSimEntity`` instances are created through the ``Experiment`` interface.
Entities record how a script, or application are executed on a specific launcher
(e.g. *Slurm*)

SmartSim can create entities as well as groups of entities in an ``Ensemble``.
``Ensemble`` instances are groups of entities that can be executed, monitored,
queried, and terminated together.

At a bare minimum, entities are required to have ``run_settings`` that define
execution and should have at least one argument: ``executable``.

Run Settings
------------

The run settings for any entity define how the entity should be run on
the launcher being used for the experiment. In the case of Ensembles,
run settings determine how each ``Model`` in the ensemble will be executed.

In the example below, run settings to a ``Model`` are given that describe
how to run a simple python script that sleeps for a certain amount of time.
These run settings are for the *local* launcher.

.. code-block:: python
    :linenos:

    run_settings = {
        "executable": "python",
        "exe_args": "sleep.py --time 10"
    }

For workload managers like *Slurm*, arguments normally supplied to the
run command, like ``srun`` can be provided as key-value pairs in the
run settings. For example, a model with the following run settings
will execute the ``lmp`` binary with the arguments
``-i in.crack`` on ``1`` node with ``48`` processors per node inside of the
allocation with the id ``123456``.

.. code-block:: python
    :linenos:

    run_settings = {
        "executable": "lmp",
        "exe_args": "-i in.crack",
        "exclusive": None,
        "nodes": 1,
        "ntasks-per-node": 48,
        "env_vars": {
            "OMP_NUM_THREADS": 1
        },
        "alloc": 123456
    }

SmartSim reserves a few run settings for specific purposes.

 - ``executable``: sets the executable to be run (str, required)
 - ``exe_args``: set arguments to the executable (str, optional)
 - ``env_vars``: set environment of the entity at launch (dict, optional)
 - ``alloc``: id of allocation (str, required for workload managers like Slurm)

Anything that is not recognized as a SmartSim argument will be passed
to the command created for the entity. This can be useful for passing
extra arguments to workload managers like *Slurm* or *PBS*. In the
example above, the ``ntasks-per-node`` argument for slurm gets translated
into ``--ntasks-per-node=48`` when the command for the entity is created.

For arguments without a value, such as ``exclusive`` in the example above,
``None`` can be used as a value to signify that the argument has no
value and the argument will be added to the command as ``--exclusive``.


Model
-----

To initialize a ``Model`` object, use the ``Experiment.create_model`` method.

.. code-block:: python
    :linenos:

    exp = Experiment("LAMMPS-crack-prop-2d-solid", launcher="local")
    run_settings = {
        "executable": "mpirun",
        "exe_args": "-np 2 lmp_mpi -i in.crack"
    }
    model = exp.create_model("crack-propagation-model", run_settings)

.. automethod:: smartsim.Experiment.create_model

Above we created an experiment for our LAMMPS model and defined how our model
should be run in the ``run_settings``. The ``Experiment.create_model`` call returns
the model object that can be started, monitored, and restarted.

Ensemble
--------

Instead of creating just a single ``Model``, we can also create a group
of ``Model`` instances by instantiating an ``Ensemble``.

To initialize a ``Ensemble``, use the ``Experiment.create_ensemble`` method.

.. code-block:: python
    :linenos:

    exp = Experiment("LAMMPS-crack-prop-2d-solid", launcher="local")
    # create an empty ensemble
    ensemble = exp.create_ensemble("crack-propagation-ensemble")

The code above initializes an empty ensemble that can be filled with ``Model``
objects.

.. automethod:: smartsim.Experiment.create_ensemble

There are two ways to populate the models within an ensemble. As mentioned,
the first way is to manually construct an ensemble through calls to
``Ensemble.add_model``.

.. code-block:: python
    :linenos:

    # create an empty ensemble
    ensemble = exp.create_ensemble("crack-propagation-ensemble")
    # create model to put into the ensemble
    run_settings = {
        "executable": "mpirun",
        "exe_args": "-np 2 lmp_mpi -i in.crack"
    }
    model = exp.create_model("crack-propagation-model", run_settings)
    # add the model to the ensemble
    ensemble.add_model(model)

Generating Ensembles
====================

The second way to create an ``Ensemble`` is to provide model parameters.
Model parameters are specific to the models to be run.
When passed to an ``Ensemble``, the ``params`` argument is expanded
at initialization into created ``Model`` instances. The number of
created ``Model`` instances depends on the length of the values
in the ``params`` dictionaries and the *permutation strategy*.

Permutation Strategies
----------------------

``Ensemble`` instances can generate their ``Model`` instances via
three built-in permutation strategies, or through a user defined
callable function.

There are three built in permutation strategies: ``all_perm``, ``random``, and ``step``.

  1) ``all_perm`` returns all possible combinations of the input parameters
  2) ``random`` returns ``n_models`` models. This can be seen as a random subset of all possible combinations.
     The argument ``n_models`` must be passed as a keyword argument to the ``Ensemble`` initialization.
  3) ``step`` returns every pair of equal length arrays. Like ``zip`` in python.

In this case, we want to construct a ensemble of 10 LAMMPS models
that will run a 2D crack propagation simulation. To do this, we
supply the ``Ensemble`` initialization with a dictionary of a single
key with 10 values. The default permutation strategy of creating
all possible permutations (``all_perm``) is used.

.. code-block:: python
    :linenos:

    exp = Experiment("LAMMPS-crack-prop-2d-solid", launcher="local")
    model_parameters = {
        "STEPS": [x for x in range(1000, 11000, 1000)]
    }
    run_settings = {
        "executable": "mpirun",
        "exe_args": "-np 2 lmp_mpi -i in.crack"
    }
    ensemble = exp.create_ensemble(
        "crack-propagation-ensemble",
        params=model_parameters,
        run_settings=run_settings,
        perm_strategy="all_perm"
    )

The above code will generate 10 ``Model`` instances each with a different
value of the *STEPS* model parameter. The ``run_settings`` are propagated to
each ``Model`` instance created, hence, this is also a quick way to
create replicates of models to run in parallel.


User-defined Ensemble Generation
--------------------------------

User supplied functions must accept at least ``param_names`` and ``param_values``,
where ``param_names`` is a list of the supplied parameter names, and ``param_values`` is a
list of the corresponding parameter names.

The functions must return a list of dictionaries, where each element in the list
is the dictionary of ``run_settings`` for a ``Model``.  For example:

.. code-block:: python
    :linenos:

    def my_function(param_names, param_values):
        # only return the single parameter/value
        return [{ param_names[0] : param_values[0] }]

    exp = Experiment("LAMMPS-crack-prop-2d-solid", launcher="local")
    model_parameters = {
        "STEPS": [x for x in range(1000, 11000, 1000)]
    }
    run_settings = {
        "executable": "mpirun",
        "exe_args": "-np 2 lmp_mpi -i in.crack"
    }
    ensemble = exp.create_ensemble(
        "crack-propagation-ensemble",
        params=model_parameters,
        run_settings=run_settings,
        perm_strategy=my_function
    )

The above code will only create one ``Model`` instance in our ``Ensemble``
as we only return a single dictionary of model parameters.

User written functions are not limited to only receiving the above arguments.
Extra arguments may be added to the function as necessary and passed to
the generation strategy through the keyword (``kwargs``) argument of
``Experiment.create_ensemble``


Input files and Datasets
========================

Very commonly, entities will need files such as input datasets and
configuration files. Once an entity has been created, these files
can be linked to an entity though a method call to ``attach_generator_files``.

Three groups of files can be supplied to ``attach_generator_files``. If a
directory is supplied, that directory will be included recursively.

``to_configure`` is used for input files that a user would like to write their model parameters into. These
files must be tagged ahead of time, and only include tagged variables
that are also included in the model parameters of the ``Ensemble`` or
``Model`` object. These files are provided as a list of file paths.

Tagging is the process of identifying model configurations you would like to create a
parameter space for and surrounding those values with a specific
character which SmartSim refers to as a ``tag``. The default tag
is a semi-colon (e.g. ``;``) but can be changed either through
the ``Experiment.generate`` method as an argument. A tutorial on tagging input files is
provided in the `ensemble generation tutorial <../../examples/LAMMPS/crack/README.html>`_

``to_copy`` is used for files that a user wants to have copied into
the filepath of the entity, but not have read or written. Both
files and directories can be included as paths in a list.

``to_symlink`` is used for files that a user wants to have in the
path of the entity at runtime, but not want to have copied such as large
input datasets.

To have the files attached to each entity present at runtime (and written
if supplied in the ``to_configure`` argument), users must call
``Experiment.generate()`` which generates a file structure for the
experiment as well as copies, symlinks, and writes model files
and directories.

.. automethod:: smartsim.Experiment.generate

The following generates the experiment file structure for the ``Ensemble``
we defined above

.. code-block:: python
    :linenos:

    exp = Experiment("LAMMPS-crack-prop-2d-solid", launcher="local")
    model_parameters = {
        "STEPS": [x for x in range(1000, 11000, 1000)]
    }
    run_settings = {
        "executable": "mpirun",
        "exe_args": "-np 2 lmp_mpi -i in.crack"
    }
    ensemble = exp.create_ensemble(
        "crack-propagation-ensemble",
        params=model_parameters,
        run_settings=run_settings
    )
    exp.generate(ensemble)

The above code will generate a file structure such as the one below

.. code-block:: text

    LAMMPS-crack-prop-2d-solid
    └── crack-propagation-ensemble
        ├── crack-propagation-ensemble_0
        ├── crack-propagation-ensemble_1
        ├── crack-propagation-ensemble_2
        ├── crack-propagation-ensemble_3
        ├── crack-propagation-ensemble_4
        ├── crack-propagation-ensemble_5
        ├── crack-propagation-ensemble_6
        ├── crack-propagation-ensemble_7
        ├── crack-propagation-ensemble_8
        └── crack-propagation-ensemble_9

    11 directories, 0 files


To extend the example we started above, we can attach an input file to
our ``Ensemble`` so that the value for *STEPS* is written into each
input file.

To attach the input file for the model, we call ``ensemble.attach_generator_files``
with the input file listed in the ``to_configure`` argument.


.. code-block:: python
    :linenos:

    exp = Experiment("LAMMPS-crack-prop-2d-solid", launcher="local")
    model_parameters = {
        "STEPS": [x for x in range(1000, 11000, 1000)]
    }
    run_settings = {
        "executable": "mpirun",
        "exe_args": "-np 2 lmp_mpi -i in.crack"
    }
    ensemble = exp.create_ensemble(
        "crack-propagation-ensemble",
        params=model_parameters,
        run_settings=run_settings
    )
    ensemble.attach_generator_files(to_configure="/path/to/in.crack")
    exp.generate(ensemble)

Which will generate a file structure for the ``Ensemble`` with a directory
for each ``Model`` object created. Each ``in.crack`` file will have the
tagged value of *STEPS* written into with the value assigned to the
``Model`` instance during the initialization of the ensemble.

.. code-block:: text

    LAMMPS-crack-prop-2d-solid
    └── crack-propagation-ensemble
        ├── crack-propagation-ensemble_0
        │   └── in.crack
        ├── crack-propagation-ensemble_1
        │   └── in.crack
        ├── crack-propagation-ensemble_2
        │   └── in.crack
        ├── crack-propagation-ensemble_3
        │   └── in.crack
        ├── crack-propagation-ensemble_4
        │   └── in.crack
        ├── crack-propagation-ensemble_5
        │   └── in.crack
        ├── crack-propagation-ensemble_6
        │   └── in.crack
        ├── crack-propagation-ensemble_7
        │   └── in.crack
        ├── crack-propagation-ensemble_8
        │   └── in.crack
        └── crack-propagation-ensemble_9
            └── in.crack

    11 directories, 10 files