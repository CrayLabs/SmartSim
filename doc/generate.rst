
*********
Ensembles
*********

Ensembles are a user-facing object within SmartSim. Ensembles are used
to quickly create a number of models that span a model parameter space.
The primary reason for exposing an ``Ensemble`` object is that manually
creating and executing many models can be very tedious. By utilizing
``Ensemble`` in SmartSim, creating an ensemble is as easy as defining
a few dictionarys and a single method call.

The ``Generator`` is the primary object responsible for creating and
configuring ensembles within SmartSim. There are two ways to use
a ``Generator``.

The most common method is through the ``Experiment.generate()``
method. This method will generate all of the previously defined
objects within an ``Experiment``. Calling this method will generate
the file structure for an experiment as well as create and configure
the model objects for each ``Ensemble`` created by that ``Experiment``.

The ``Experiment.generate`` method makes generating a number of models
quick and easy, but if greater control over the generation process is
needed, a ``Generator`` object can be initialized seperately from the
``Experiment``. Users can then user the ``Generator`` interface to
generate specific Ensembles and SmartSim entities along with their file
structure. An example of this is provided in the
`Ensemble suite tutorial <../examples/MOM6/double-gyre-ensembles/README.html>`_


Configuring Ensembles
=====================

Ensembles are created through the method ``Experiment.create_ensemble()`` or
manually initialized for advanced users. Before calling the creation method,
however, the configuration of the Ensemble must be defined.

Model Parameters
----------------

The first step in configuring an ensemble is tagging the input files
where the simulation configurations are held. Tagging is the process
of identifying model configurations you would like to create a
parameter space for and surronding those values with a specific
character which SmartSim refers to as a ``tag``. The default tag
is a semi-colon (e.g. ``;``) but can be changed either through
the ``Experiment.generate`` method as an arugment, or if you are
manually initializing a ``Generator`` instance, through the method
``Generator.set_tag``. A tutorial on tagging input files is
provided in the `ensemble generation tutorial <../examples/LAMMPS/crack/README.html>`_


Run Settings
------------

The run settings for any entity define how the model should be run on
the SmartSim launcher backend being used for the experiment. In the
case of Ensembles, run settings determine how each model in the
ensemble will be executed. For example, because of the ``run_settings``
defined below, each model in the ensemble configured with these
run settings will run the ``lmp`` executeable with the arguments
``-i in.crack`` on 1 node with 48 processors per node inside of the
allocation with the id 123456.

.. code-block:: python

  run_settings = {
      "executable": "lmp",
      "exe_args": "-i in.crack",
      "nodes": 1,
      "ppn": 48,
      "env_vars": {
          "OMP_NUM_THREADS": 1
      },
      "alloc": 123456
  }


Input files and Datasets
-------------------------

Once input files have been tagged, the ``Generator`` needs to know
where to get them from such that they can put them in the same filepath
as the simulation. To do this, created entities have a method to
attach files for the generator to utilize. There are three arguments
to the ``Ensemble.attach_generator_files`` function:

``to_configure`` is used for input files that a user would like the
``Generator`` to read and write their model parameters into. These
files must be tagged ahead of time, and only include tagged variables
that are also included in the model parameters of the Ensemble or
Model object. These files are provided as a list of file paths.

``to_copy`` is used for files that a user wants to have copied into
the filepath of the simulation, but not have read or written. Both
files and directories can be included as paths in a list.

``to_symlink`` is used for files that a user wants to have in the
path of the simulation, but not want to have copied such as large
input datasets.


Ensemble Generation
===================


Generation Strategies
---------------------

The Generator utilizes multiple strategies of generating models
from the Ensemble parameters provided by the user.
There are three built in permutation strategies: ``all_perm``, ``random``, and ``step``.

  1) ``all_perm`` returns all possible combinations of the input parameters
  2) ``random`` returns ``n_models`` models. This can be seen as a random subset of all possible combinations.
     The argument ``n_models`` must be passed as a keyword argument to the function being used for generation.
  3) ``step`` returns every pair of equal length arrays. Like ``zip`` in python.

.. code-block:: python

  exp = Experiment("generation-example")
  # < create some entities ...>
  exp.generate(strategy="random", n_models=2)

User supplied functions must accept at least ``param_names`` and ``param_values``,
where ``param_names`` is a list of the supplied parameter names, and ``param_values`` is a
list of the corresponding parameter names.

The functions must return a list of dictionaries, where each element in the list
is the dictionary for a model.  For example:

.. code-block:: python

  def my_function(param_names, param_values):
    # only return the single parameter/value
    return [{ param_names[0] : param_values[0] }]

  exp = Experiment("generation-example")
  # < create some entities ...>
  exp.generate(strategy=my_function, n_models=2)

User written functions are not limited to only receiving the above arguments.
Extra arguments may be added to the function as necessary; at runtime, these are
passed through to the selection strategy via the ``Experiment.generate`` function (as above,
as in for "random" and ``n_models``).
