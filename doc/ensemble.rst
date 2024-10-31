********
Ensemble
********
========
Overview
========
An ``Ensemble`` is a builder class that parameterizes the creation of multiple ``Application(s)``. The object
transforms into a list of deployable ``Job(s)``, with identical ``LaunchSettings`` applied to each ``Job``.

==========
Initialize
==========
An ``Ensemble`` supports two main creation strategies: **Parameter Expansion** and **Replication**.

Parameter Expansion
===================
With parameter expansion, an ``Ensemble`` enables the creation of multiple ``Application(s)`` by assigning different parameter
values to each one. This involves providing inputs to ``Ensemble.file_parameters``, ``Ensemble.exe_arg_parameters``,
and ``Ensemble.permutation_strategy``.

Here's a breakdown of how it works:

1. **File Parameters**: These are parameters that are read from files and can vary between ``Application(s)``.
2. **Executable Argument Parameters**: These are parameters passed as arguments to the executable and can also vary.
3. **Permutation Strategy**: This determines how the different parameter values are combined to create unique sets
   for each ``Application``. The strategies include:

   - **all_perm**: Generates all possible combinations of the parameters.
   - **step**: Collects identically indexed values across parameter lists to create sets.
   - **random**: Selects randomly from predefined parameter spaces.

For example, if you have two sets of file parameters and two sets of executable arguments, using the
`"step"` strategy might look like this:

.. code-block:: python

    file_params = {"SPAM": ["a", "b"], "EGGS": ["c", "d"]}
    exe_arg_parameters = {"EXE": [["a"], ["b", "c"]], "ARGS": [["d"], ["e", "f"]]}
    ensemble = Ensemble(
        name="example_ensemble",
        exe="python",
        exe_arg_parameters=exe_arg_parameters,
        file_parameters=file_params,
        permutation_strategy="step"
    )

This configuration will create two ``Application(s)`` with the following parameter sets:

.. code-block:: python

    [
        ParamSet(params={'SPAM': 'a', 'EGGS': 'c'}, exe_args={'EXE': ['a'], 'ARGS': ['d']}),
        ParamSet(params={'SPAM': 'b', 'EGGS': 'd'}, exe_args={'EXE': ['b', 'c'], 'ARGS': ['e', 'f']})
    ]

Each ``ParamSet`` contains the parameters assigned from ``file_params`` and the corresponding executable
arguments from ``exe_arg_parameters``. This allows for flexible and efficient creation of multiple
``Application(s)`` with varying configurations.

Replication
===========
Replication involves creating identical ``Application(s)`` by specifying the ``Ensemble.replicas`` argument. Applying this
to a parameter expansion setup doubles the number of ``Application(s)`` by replicating each parameter set.

Consider the following example where we apply the ``replicas`` argument to create multiple identical ``Application(s)``:

.. code-block:: python

    file_params = {"SPAM": ["a", "b"], "EGGS": ["c", "d"]}
    exe_arg_parameters = {"EXE": [["a"], ["b", "c"]], "ARGS": [["d"], ["e", "f"]]}
    ensemble = Ensemble(
        name="example_ensemble",
        exe="python",
        exe_arg_parameters=exe_arg_parameters,
        file_parameters=file_params,
        permutation_strategy="step",
        replicas=2
    )

This configuration will result in each parameter set being replicated, effectively doubling the number
of ``Application(s)`` created:

.. code-block:: python

    [
        ParamSet(params={'SPAM': 'a', 'EGGS': 'c'}, exe_args={'EXE': ['a'], 'ARGS': ['d']}),
        ParamSet(params={'SPAM': 'a', 'EGGS': 'c'}, exe_args={'EXE': ['a'], 'ARGS': ['d']}),
        ParamSet(params={'SPAM': 'b', 'EGGS': 'd'}, exe_args={'EXE': ['b', 'c'], 'ARGS': ['e', 'f']}),
        ParamSet(params={'SPAM': 'b', 'EGGS': 'd'}, exe_args={'EXE': ['b', 'c'], 'ARGS': ['e', 'f']})
    ]

Each ``ParamSet`` is duplicated according to the number of replicas specified, allowing for multiple identical
``Application(s)`` to be created and deployed.

=========
Configure
=========
This section will cover general methods for configuring `exe`, `exe_args`, `exe_arg_parameters`, `files`,
`permutation_strategy`, `max_permutations` and `replicas` attributes of ``Ensemble``.

Executable
==========
To set the executable after initializing the ``Ensemble`` object, assign the desired executable path to
the `exe` attribute. If you only specify the executable name, SmartSim will attempt to locate the executable path
on the machine. For example:

.. code-block:: python

    # Set the executable for the Application instance
    my_ensemble.exe = "/path/to/new_executable"

    # or just the executable name
    my_ensemble.exe = "new_executable"

Executable Arguments
====================
To set the executable arguments after initializing the ``Ensemble`` object, assign a list of arguments
or a string argument to the `exe_args` attribute. This will overwrite any previously set
`exe_args`. For example:

  .. code-block:: python

      my_ensemble.exe_args = ["arg1", "arg2"]

      my_ensemble.exe_args = "arg1"

Executable Arguments Parameters
===============================
To set the executable arguments for an ``Ensemble``, assign the desired arguments to the ``exe_arg_parameters``
attribute. This attribute expects a mapping of strings to sequences of sequences of strings, allowing you to
specify different sets of arguments for each ``Application`` instance.

For example:

.. code-block:: python

    my_ensemble.exe_arg_parameters = {"EXE": [["arg1"], ["arg2", "arg3"]], "ARGS": [["arg4"], ["arg5", "arg6"]]}

Permutation Strategy
====================
To configure the permutation strategy for an ``Ensemble``, assign the desired strategy to the
``permutation_strategy`` attribute. The available strategies are:

- **all_perm**: Generates all possible parameter permutations for exhaustive exploration.
- **step**: Collects identically indexed values across parameter lists to create parameter sets.
- **random**: Enables random selection from predefined parameter spaces.

For example:

.. code-block:: python

    my_ensemble.permutation_strategy = "all_perm"

    my_ensemble.permutation_strategy = "step"

    my_ensemble.permutation_strategy = "random"

Maximum Permutations
====================
To limit the number of parameter permutations generated for an ``Ensemble``, assign the desired maximum
number to the ``max_permutations`` attribute. This is useful for ensuring that only a specified number
of permutations are created.

For example:

.. code-block:: python

    # Set the maximum number of parameter permutations to 10
    my_ensemble.max_permutations = 10

    # Set the maximum number of parameter permutations to 50
    my_ensemble.max_permutations = 50

Replicas
========
To create multiple identical ``Application(s)`` within an ``Ensemble``, assign the desired number of replicas to
the ``replicas`` attribute. This strategy is useful when you need to run the same ``Application`` multiple
times with the same configuration.

For example:

.. code-block:: python

    # Set the number of replicas to 3
    my_ensemble.replicas = 3

    # Set the number of replicas to 5
    my_ensemble.replicas = 5

Input Files
===========
In this section, we will explore how to attach files to an ensemble using the ``Ensemble.files``
attribute. This attribute allows users to add files through three different operations:

1. Copying
2. Creating symlinks
3. Configuring files

Each file operation can be added to the `files` attribute of the ``Ensemble`` instance.

----
Copy
----
Copying files involves creating a duplicate of the source file at the destination path. This is useful
when you need to ensure that the original file remains unchanged while providing a copy for the application
to use.

**Adding a Copy Operation:**
To add a copy operation, use the `add_copy` method on the `files` attribute of the ``Ensemble`` instance.
This method requires the source path (`src`) and the destination path (`dest`) to be of type ``pathlib.Path``.
For example:

.. code-block:: python

    my_ensemble.files.add_copy(
        src=pathlib.Path("/path/to/source"),
        dest=pathlib.Path("/path/to/destination")
    )

This will create a copy of the file located at `"/path/to/source"` and place it at `"/path/to/destination"`.

-------
Symlink
-------
Creating symlinks involves creating a symbolic link from the source file to the destination path. This
is useful when you want to reference the original file without duplicating it.

**Adding a Symlink Operation:**
To add a symlink operation, use the `add_symlink` method on the `files` attribute of the ``Ensemble`` instance.
This method requires the source path (`src`) and the destination path (`dest`) to be of type pathlib.Path.
For example:

.. code-block:: python

    my_ensemble.files.add_symlink(
        src=pathlib.Path("/path/to/source"),
        dest=pathlib.Path("/path/to/destination")
    )

This will create a symbolic link from the file located at `"/path/to/source"` to `"/path/to/destination"`.

---------
Configure
---------
Configuring files involves modifying the content of the source file based on specified parameters and
then placing the modified file at the destination path. This is useful for setting up configuration
files with dynamic content.

**Adding a Configure Operation:**
To add a configure operation, use the `add_configuration` method on the `files` attribute of the ``Ensemble``
instance. This method requires the source path (`src`), the destination path (`dest`), and the file parameters
(`file_parameters`) to be of type (TODO). Additionally, you can specify a tag (`tag`) to identify the
configuration. For example:

.. code-block:: python

    my_ensemble.files.add_configuration(
        src=pathlib.Path("/path/to/source"),
        file_parameters={"key": "value"},
        dest=pathlib.Path("/path/to/destination"),
        tag=";"
    )

This will modify the content of the file located at `"/path/to/source"` based on the specified parameters and
place the modified file at `"/path/to/destination"`.

================
Expand into Jobs
================
Expand an ``Ensemble`` into a list of deployable ``Job(s)`` and apply identical ``LaunchSettings`` to each ``Job``.

The number of ``Job(s)`` returned is controlled by the Ensemble attributes:

- ``Ensemble.exe_arg_parameters``
- ``Ensemble.file_parameters``
- ``Ensemble.permutation_strategy``
- ``Ensemble.max_permutations``
- ``Ensemble.replicas``

Consider the example below:

.. code-block:: python

    # Create LaunchSettings
    my_launch_settings = LaunchSettings(...)

    # Initialize the Ensemble
    ensemble = Ensemble("my_name", "echo", "hello world", replicas=3)
    # Expand Ensemble into Jobs
    ensemble_as_jobs = ensemble.build_jobs(my_launch_settings)

By calling ``build_jobs`` on ``ensemble``, three ``Job(s)`` are returned because three replicas
were specified. Each ``Job`` will have the provided ``LaunchSettings``.