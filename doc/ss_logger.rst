****************
SmartSim Logging
****************
========
Overview
========
SmartSim supports experiment tracking with logging functionality
using the SmartSim log module. The logger, supported by Python logging, enables
monitoring the experiment at runtime by allowing users to print messages **to stdout**
and/or **to file** from within the Python driver script. The SmartSim logger allows users to categorize
messages by severity level. Users may instruct the SmartSim to print
specified severity levels and ignore others through the `SMARTSIM_LOG_LEVEL`
environment variable.

The `SMARTSIM_LOG_LEVEL` environment variable accepts **three** inputs,
ranked from lowest severity (1) to highest (3). Each severity level has
associated logging functions users may use within an experiment driver script:

    1. level: `quiet`
       - function: ``logger.error()`` & ``logger.warning()``
    2. level: `info`
       - function: ``logger.info()``
    3. level: `debug`
       - function: ``logger.debug()``

The messages printed to stdout or to file by the SmartSim logger depend on the specified value.
The logger prints the specified severity level and all levels below it. For instance,
if `SMARTSIM_LOG_LEVEL=info`, the logger will print messages from ``logger.error()``,
``logger.warning()``, and ``logger.info()``. With `SMARTSIM_LOG_LEVEL=debug`, messages from
``logger.error()``, ``logger.warning()``, ``logger.info()``, and ``logger.debug()`` will be printed.

==========================
Example: Logging to stdout
==========================
Below, we provide an implementation of the SmartSim logger.

To use the SmartSim logger, import the required module:

.. code-block:: python

      from smartsim.log import get_logger

Next, initialize an instance of the logger and provide a `name`:

.. code-block:: python

      logger = get_logger("example_logger")

To demonstrate the full functionality of of the SmartSim logger, we include all log
functions in the Python driver script with log messages:

.. code-block:: python

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

Before executing the Python script, we must set the environment variable, `SMARTSIM_LOG_LEVEL`,
to a log level. The default is `info`.
For the example, set `SMARTSIM_LOG_LEVEL` to `quiet` in the terminal::
    export SMARTSIM_LOG_LEVEL=quiet

When we execute the script with `SMARTSIM_LOG_LEVEL=quiet`,
the following messages from will print from above::
    21:07:40 osprey.us.cray.com SmartSim[10950] ERROR This is an error message
    21:07:40 osprey.us.cray.com SmartSim[10950] WARNING This is a warning message

Notice that the `info` and `debug` messages were ignored. This is because by setting
a lower severity level (`quiet`), we instruct SmartSim to ignore the higher severity levels (`debug` and `info`).

Next, set `SMARTSIM_LOG_LEVEL` to `info` in the terminal::
    export SMARTSIM_LOG_LEVEL=info

When we run the script with `SMARTSIM_LOG_LEVEL=info`,
the following output appears in stdout::
    19:52:05 osprey.us.cray.com SmartSim[130033] INFO This is a message
    19:52:05 osprey.us.cray.com SmartSim[130033] ERROR This is an error message
    19:52:05 osprey.us.cray.com SmartSim[130033] WARNING This is a warning message

Above, we have instructed SmartSim to print all `info` messages but ignore higher
verbose messages (`debug`).

Now set the `SMARTSIM_LOG_LEVEL` to `debug` and check the output of the program.
Set the environment variable in the terminal like so::
    export SMARTSIM_LOG_LEVEL=debug

When we run the program once again, the following output is printed to stdout::
    20:11:12 osprey.us.cray.com SmartSim[65385] INFO This is a message
    20:11:12 osprey.us.cray.com SmartSim[65385] DEBUG This is a debug message
    20:11:12 osprey.us.cray.com SmartSim[65385] ERROR This is an error message
    20:11:12 osprey.us.cray.com SmartSim[65385] WARNING This is a warning message

Notice that all log messages are visible since we set `SMARTSIM_LOG_LEVEL`
to the highest severity log level.

========================
Example: Logging to File
========================
The ``log_to_file()`` function in SmartSim allows users to log messages
to a specified file by providing a `name` to the function. The severity
level of messages printed to the file is determined by the
`SMARTSIM_LOG_LEVEL` variable.

Begin by importing the function `get_logger` and `log_to_file`:

.. code-block:: python

      from smartsim.log import get_logger, log_to_file

Initialize a logger for use within the Python driver script:

.. code-block:: python

      logger = get_logger("example_logger")

Using the ``log_to_file()`` function, instruct SmartSim to create a file named
`logger.out` to write log messages to:

.. code-block:: python

      log_to_file("logger.out")

For the example, we add all log message severities to the script:

.. code-block:: python

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

Set `SMARTSIM_LOG_LEVEL` to `debug` in the terminal to instruct SmartSim to print all
log messages::
    export SMARTSIM_LOG_LEVEL=debug

When we execute the Python program,
a file named `logger.out` is created in our working directory with the listed contents::
    21:07:40 osprey.us.cray.com SmartSim[10950] INFO This is a message
    21:07:40 osprey.us.cray.com SmartSim[10950] DEBUG This is a debug message
    21:07:40 osprey.us.cray.com SmartSim[10950] ERROR This is an error message
    21:07:40 osprey.us.cray.com SmartSim[10950] WARNING This is a warning message

If the program is reran with the same file name, the file contents will be overwritten.