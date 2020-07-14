
Ensembles
---------

Talk through the motivation and reason behind the ensemble entity
and how its used.


Ensemble Parameters
===================

describe how ensembles use model parameters


Ensemble Generation
===================

Describe the generation strategies


Generation Strategies
=====================

To generate our models we need to create an instance of a generator, provide
the tagged configuration files and make a call to ``Generator.generate()``.  The
``generate`` function creates models according to the specified permutation strategy,
which, by default, is "all permutations": it creates every model possible, given the
input parameters.  In order to select the strategy, we may call the
``Generator.set_strategy()`` function with the following argument types: a string
corresponding to one of the internal strategies, a string formatted as "module.function"
that the Generator will then load, or an actual function.

There are three built in permutation strategies: "all_perm", "random", and "step".
"all_perm" returns all possible combinations of the input parameters; "random" returns
``n_models`` models; this can be seen as a random subset of all possible combinations.
The argument ``n_models`` must be passed to the ``generate`` function.

.. code-block:: python

  # Supply the generator with necessary files to run the simulation
  # and generate the specified models.
  base_config = "LAMMPS/in.atm"
  GEN = Generator(experiment, model_files=base_config)
  GEN.set_strategy("random")
  GEN.generate(n_models=2)

User supplied functions must accept at _least_ ``param_names`` and ``param_values``,
where param_names is a list of the supplied parameter names, and param_values is a
list of the corresponding parameter names.  In the following example, ``param_names``
is equal to ``[steps]``, and param_values is ``[20, 25, 30, 35]``.

The functions must return a list of dictionaries, where each element in the list
is the dictionary for a model.  For example:

.. code-block:: python

  def my_function(param_names, param_values):
    # only return the single parameter/value
    return [{ param_names[0] : param_values[0] }]

  base_config = "LAMMPS/in.atm"
  GEN = Generator(experiment, model_files=base_config)
  GEN.set_strategy(my_function)
  GEN.generate()

User written functions are not limited to only receiving the above arguments.
Extra arguments may be added to the function as necessary; at runtime, these are
passed through to the selection strategy via the ``generate`` function (as above,
as in for "random" and ``n_models``).

Strategy selection is optional; by default, "all_perm" is used, and the following
is also valid:

.. code-block:: python

  # Supply the generator with necessary files to run the simulation
  # and generate the specified models
  base_config = "LAMMPS/in.atm"
  GEN = Generator(experiment, model_files=base_config)
  GEN.generate()
