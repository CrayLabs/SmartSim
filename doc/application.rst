***********
Application
***********
========
Overview
========
The ``Application`` class facilitates the execution of computational tasks within an ``Experiment`` workflow.
These tasks can include launching compiled applications, running scripts, or performing general
computational operations. ``Application(s)`` integrate into ``Job(s)``, where ``LaunchSettings`` provide
launcher-specific behavior for the ``Job``.

==========
Initialize
==========
This section details the steps to set up the parameters that control your application's initialization.

**Step 1: Import Application**

After installing Smartsim, ``Application`` may be imported in Python code like:

.. code-block:: python

    from smartsim import Application

**Step 2: Set the Application Name and Executable**

Set the application name and executable. The `name` is a string that identifies the application, and
`exe` is the string path to the executable. Optionally, you can provide `exe_args` as a string or sequence of
of strings to specify arguments for the executable.

* `name`: A string that identifies the application. Example:

  .. code-block:: python

    name="my_app"

* `exe`: A string that specifies the path to the executable. If only the executable name is provided, SmartSim
  will attempt to locate the path on the system. Examples:

  .. code-block:: python

    exe = "/path/to/new_executable"

  .. code-block:: python

    exe = "new_executable"

* `exe_args`: An optional argument that can be a string or a sequence of strings, representing the arguments
  for the executable. Examples:

  .. code-block:: python

    exe_args="--arg1 value1 --arg2 value2"

  .. code-block:: python

    exe_args=["--arg1", "value1", "--arg2", "value2"]

**Example initializing an Application:** Once you have imported ``Application`` using ``from smartsim
import Application``, initialize an ``Application``. For example:

.. code-block:: python

    app = Application(
        name="my_app",
        exe="/path/to/executable",
        exe_args="--arg1 value1 --arg2 value2"
    )

=========
Configure
=========
After initializing an ``Application`` object, configure the ``exe``, ``exe_args``, or ``files`` attribute. If the simulation
requires specific parameters, attach configuration files via the ``Application.files`` attribute. This attribute also
supports copying or symlinking files into the job's run directory to ensure accessibility during the simulation.
To reuse an application but alter the system state, update the ``exe_args`` attribute. To change the executable while keeping
the same arguments, modify the ``exe`` attribute.

Executable
==========
To reset the executable after initializing the ``Application`` object, assign the desired executable path to the ``exe``
attribute. If only the executable name is specified, SmartSim will attempt to locate the executable path on the
machine. For example:

.. code-block:: python

    # Set the executable path
    my_app.exe = "/path/to/new_executable"

    # or just the executable name
    my_app.exe = "new_executable"

Executable Arguments
====================
There are two methods to configure ``exe_args``:

1. Use ``Application.exe_args`` to overwrite the existing executable arguments.
2. Use ``Application.add_exe_args`` to add to the existing executable arguments.

**Option 1: Use Application.exe_args**

Set the executable arguments directly by assigning a list of arguments or a string argument
to the ``exe_args`` attribute. This will overwrite any previously set executable arguments.
For example:

  .. code-block:: python

      my_app.exe_args = ["arg1", "arg2"]

      my_app.exe_args = "arg1"

**Option 2: Use Application.add_exe_args**

To add additional executable arguments after initializing the ``Application`` object, use the
``add_exe_args`` method. This method appends the new arguments to the existing ``exe_args`` without
overwriting them. For example:

  .. code-block:: python

      my_app.add_exe_args(["new_arg1", "new_arg2"])

      my_app.add_exe_args("new_arg1")

Input Files
===========
In this section, we will explore how to attach files to an application using the ``Application.files``
attribute. This attribute allows users to add files through three different operations:

1. Copying
2. Creating symlinks
3. Configuring files

Each file operation can be added to the `files` attribute of the ``Application`` instance.

----
Copy
----
Copying files involves creating a duplicate of the source file at the destination path. This is useful
when you need to ensure that the original file remains unchanged while providing a copy for the application
to use.

**Adding a Copy Operation:**
To add a copy operation, use the `add_copy` method on the `files` attribute of the ``Application`` instance.
This method requires the absolute source path (`src`) and the relative destination path (`dest`) to be of
type ``pathlib.Path``. For example:

.. code-block:: python

    my_app.files.add_copy(
        src=pathlib.Path("/path/to/source"),
        dest=pathlib.Path("destination")
    )

This will create a copy of the file located at `"/path/to/source"` and place it at `"/path/to/destination"`.

-------
Symlink
-------
Creating symlinks involves creating a symbolic link from the source file to the destination path. This
is useful when you want to reference the original file without duplicating it.

**Adding a Symlink Operation:**
To add a symlink operation, use the `add_symlink` method on the `files` attribute of the ``Application`` instance.
This method requires the absolute source path (`src`) and the relative destination path (`dest`) to be of type
pathlib.Path. For example:

.. code-block:: python

    my_app.files.add_symlink(
        src=pathlib.Path("/path/to/source"),
        dest=pathlib.Path("destination")
    )

This will create a symbolic link from the file located at `"/path/to/source"` to `"/path/to/destination"`.

---------
Configure
---------
Configuring files involves modifying the content of the source file based on specified parameters and
then placing the modified file at the destination path. This is useful for setting up configuration
files with dynamic content.

**Adding a Configure Operation:**
To add a configure operation, use the ``add_configuration`` method on the ``files`` attribute of the ``Application``
instance. This method requires the absolute source path (``src``), the relative destination path (``dest``), and the file parameters
(``file_parameters``) to be of type mapping of string to strings. Additionally, you can specify a tag
(``tag``) to identify the configuration. For example:

.. code-block:: python

    my_app.files.add_configuration(
        src=pathlib.Path("/path/to/source"),
        file_parameters={"key": "value"},
        dest=pathlib.Path("destination"),
        tag=";"
    )

This will modify the content of the file located at `"/path/to/source"` based on the specified parameters and
place the modified file at `"/path/to/destination"`.