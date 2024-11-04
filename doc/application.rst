***********
Application
***********
========
Overview
========
The ``Application`` class facilitates the execution of computational tasks within an ``Experiment`` workflow.
These tasks include launching compiled applications, running scripts, or performing general
computational operations. ``Application(s)`` integrate into ``Job(s)``, where ``LaunchSettings`` provide
launcher-specific behavior for the ``Job``.

==========
Initialize
==========
A ``Application`` requires two input arguments: name and executable. Optionally, users may
provide executable arguments. This section details argument names, argument types
and how to use the arguments to initialize an ``Application``.

**Step 1: Import Application**

After installing Smartsim, ``Application`` may be imported in Python code like:

.. code-block:: python

    from smartsim import Application

**Step 2: Set the Application Name, Executable and Executable Arguments**

The `name` is a string that identifies the application, and `exe` is the string path to the executable.
Optionally, you can provide `exe_args` as a string or sequence of strings.

* `name`: A string that identifies the application. Example:

  .. code-block:: python

    name="my_app"

* `exe`: A string that specifies the path to the executable. If only the executable name is provided, SmartSim
  will attempt to locate the path on the system. Examples:

  .. code-block:: python

    exe = "/path/to/new_executable"

    exe = "new_executable"

* `exe_args`: An optional argument of type string or sequence of strings, representing the arguments
  for the executable. Examples:

  .. code-block:: python

    exe_args="--arg1 value1 --arg2 value2"

    exe_args=["--arg1", "value1", "--arg2", "value2"]

**Example initialize an Application:**

.. code-block:: python

    from smartsim import Application

    app = Application(
        name="my_app",
        exe="/path/to/executable",
        exe_args="--arg1 value1 --arg2 value2"
    )

======
Modify
======
This section explains how to modify the attached executable and executable arguments. Additionally,
learn how to attach copy, symlink and configuration files to an initialized ``Application``.

Executable
==========
To overwrite the executable, assign the desired executable path to the ``exe`` attribute.
If only the executable name is specified, SmartSim will attempt to locate the executable path on the
machine. For example:

.. code-block:: python

    my_app.exe = "/path/to/new_executable"

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
This section details how to attach files to an ``Application``. SmartSim supports three file operations:

1. :ref:`Copy<app_file_copy>`
2. :ref:`Symlink<app_file_symlink>`
3. :ref:`Configure<app_file_configure>`

The ``Application.files`` attribute enables adding file operations. Continue to the operation
sections for further details.

.. _app_file_copy:

----
Copy
----
The copy operation involves creating a duplicate of the source file or folder at the destination
path within the job's run directory.

**Add a Copy Operation:**
To add a copy operation, use the ``add_copy`` method on the ``files`` attribute of the ``Application`` instance.
This method requires the absolute source path (``src``) and an optional relative destination path (``dest``) to be of
type ``pathlib.Path``. For example:

.. code-block:: python

    my_app.files.add_copy(
        src=pathlib.Path("/path/to/source"),
        dest=pathlib.Path("destination")
    )

This will create a copy of the file located at `"/path/to/source"` and place it within the job's
run directory at `"/job/run/destination"`.

.. _app_file_symlink:

-------
Symlink
-------
Creating symlinks involves creating a symbolic link from the source file to the destination path. This
is useful when you want to reference the original file without duplicating it.

**Adding a Symlink Operation:**
To add a symlink operation, use the ``add_symlink`` method on the ``files`` attribute of the ``Application`` instance.
This method requires the absolute source path (``src``) and an optional relative destination path (``dest``) to be of type
pathlib.Path. For example:

.. code-block:: python

    my_app.files.add_symlink(
        src=pathlib.Path("/path/to/source"),
        dest=pathlib.Path("destination")
    )

This will create a symbolic link from the file located at `"/path/to/source"` to the job's
run directory at `"/job/run/destination"`.

.. _app_file_configure:

---------
Configure
---------
Configuring files involves modifying the content of the source file based on specified parameters and
then placing the modified file at the destination path. This is useful for setting up configuration
files with dynamic content.

**Adding a Configure Operation:**
To add a configure operation, use the ``add_configuration`` method on the ``files`` attribute of the ``Application``
instance. This method requires the absolute source path (``src``), an optional relative destination path (``dest``), and the file parameters
(``file_parameters``) to be of type mapping of string to strings. Additionally, you can specify an optional tag
(``tag``) to identify the configuration. For example:

.. code-block:: python

    my_app.files.add_configuration(
        src=pathlib.Path("/path/to/source"),
        file_parameters={"key": "value"},
        dest=pathlib.Path("destination"),
        tag=";"
    )

This will modify the content of the file located at `"/path/to/source"` based on the specified parameters and
place the modified file at `"/job/run/destination"`.