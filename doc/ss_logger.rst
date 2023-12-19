***************
SmartSim Logger
***************
========
Overview
========
SmartSim supports experiment tracking through logging functionality
offered by the SmartSim `log` module. The logger, supported by Python logging, enables
monitoring the experiment at runtime by allowing users to print messages **to stdout**
and/or **to file** from within the Python driver script. The SmartSim logger permits users to categorize
messages by severity level. Users may instruct SmartSim to print to stdout
certain severity level log messages and omit others through the `SMARTSIM_LOG_LEVEL`
environment variable. The `SMARTSIM_LOG_LEVEL` environment variable may be overridden when logging to file
by specifying a log level to the ``log_to_file()`` function. Examples walking through
both logging :ref:`to stdout<log_to_stdout>` and :ref:`to file<log_to_file>` are provided below.

SmartSim offers **four** log functions to use within the Python driver script. The
below functions accept string messages:

- ``logger.warning()``
- ``logger.error()``
- ``logger.info()``
- ``logger.debug()``

The `SMARTSIM_LOG_LEVEL` environment variable accepts **three** log levels: `quiet`,
`info` and `debug`. By settings the environment to one of the log levels, will control
the log messages printed at runtime. The log levels are listed below from
highest severity to lowest severity:

- level: `quiet`
   - The `quiet` log level instructs SmartSim to print ``error()`` and ``warning()`` messages.
- level: `info`
   - The `info` log level instructs SmartSim to print ``info()``, ``error()`` and ``warning()`` messages.
- level: `debug`
   - The `debug` log level instructs SmartSim to print ``debug()``, ``info()``, ``error()`` and ``warning()`` messages.

`SMARTSIM_LOG_LEVEL` defaults to `info`. For SmartSim log sample usage examples, continue to the Configuration section.

=============
Configuration
=============
-------------
Log to stdout
-------------
.. _log_to_stdout:
The ``get_logger()`` function in SmartSim enables users to initialize
a logger instance. Once initialize, a user may use the instance
to specify a log level function and include a log message.
A user may control which messages are logged via the
`SMARTSIM_LOG_LEVEL` which defaults to `info`.

To use the SmartSim logger within the Python script, import the required module:

.. code-block:: python

      from smartsim.log import get_logger

Next, initialize an instance of the logger and provide a `name`:

.. code-block:: python

      logger = get_logger("SmartSim")

To demonstrate full functionality of the SmartSim logger, we include all log
functions in the Python driver script with log messages:

.. code-block:: python

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

Execute the script *without* setting the `SMARTSIM_LOG_LEVEL`.
Note that `SMARTSIM_LOG_LEVEL` defaults to `info`.
When we execute the script,
the following messages will print to stdout::
    11:15:00 system.host.com SmartSim[130033] INFO This is a message
    11:15:00 system.host.com SmartSim[130033] ERROR This is an error message
    11:15:00 system.host.com SmartSim[130033] WARNING This is a warning message

Notice that the `debug` messages were filtered. This is because by using
a higher severity level (`info`), we instruct SmartSim to omit the lower severity level (`debug`).

Next, set `SMARTSIM_LOG_LEVEL` to `debug` in terminal::
    export SMARTSIM_LOG_LEVEL=debug

When we execute the script,
the following messages will print to stdout::
    11:15:00 system.host.com SmartSim[65385] INFO This is a message
    11:15:00 system.host.com SmartSim[65385] DEBUG This is a debug message
    11:15:00 system.host.com SmartSim[65385] ERROR This is an error message
    11:15:00 system.host.com SmartSim[65385] WARNING This is a warning message

Notice that all log messages printed to stdout. By using
the lowest severity level (`debug`), we instruct SmartSim print all log levels.

Next, set `SMARTSIM_LOG_LEVEL` to `quiet` in terminal::
    export SMARTSIM_LOG_LEVEL=quiet

When we run the program once again,
the following output is printed to stdout::
    11:15:00 system.host.com SmartSim[65385] ERROR This is an error message
    11:15:00 system.host.com SmartSim[65385] WARNING This is a warning message

Notice that the `info` and `debug` messages were filtered. This is because by using
the highest severity level (`quiet`), we instruct SmartSim to omit the lower severity levels
(`info` and `debug`).

-----------
Log to File
-----------
.. _log_to_file:
The ``log_to_file()`` function in SmartSim allows users to log messages
to a specified file by providing a file `name`. If the file name
passed in does not exist, SmartSim will create the file.
If the program is reran with the same
file name, the file contents will be overwritten. The severity
level of messages outputted to the file can be set by the
`SMARTSIM_LOG_LEVEL` variable. The `SMARTSIM_LOG_LEVEL` may be overridden
by specifying a log level to the ``log_to_file()`` function
so file output is more/less verbose than stdout.

To demonstrate, begin by importing the function `get_logger` and `log_to_file`:

.. code-block:: python

      from smartsim.log import get_logger, log_to_file

Initialize a logger for use within the Python driver script:

.. code-block:: python

      logger = get_logger("SmartSim")

Add the ``log_to_file()`` function to instruct SmartSim to create a file named
`logger.out` to write log messages to:

.. code-block:: python

      log_to_file("logger.out")

For the example, we add all log message severities to the script:

.. code-block:: python

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

Note that the default value for the `SMARTSIM_LOG_LEVEL` variable is `info`.
Therefore, we will not set the environment variable and instead rely on the
default.

When we execute the Python script,
a file named `logger.out` is created in our working directory with the listed contents::
    11:15:00 system.host.com SmartSim[10950] INFO This is a message
    11:15:00 system.host.com SmartSim[10950] ERROR This is an error message
    11:15:00 system.host.com SmartSim[10950] WARNING This is a warning message

In the same Python script, add a log level to the ``log_to_file()`` as a input argument:

.. code-block:: python

      log_to_file("logger.out", "quiet")

When we execute the Python script once again,
SmartSim will override the `SMARTSIM_LOG_LEVEL` variable to output messages of log level `quiet`.
SmartSim will overwrite the contents of `logger.out` with::
    11:15:00 system.host.com SmartSim[10950] ERROR This is an error message
    11:15:00 system.host.com SmartSim[10950] WARNING This is a warning message