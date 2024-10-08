******
Logger
******

.. _ss_logger:

========
Overview
========
SmartSim supports logging experiment activity through a logging API accessible via
the SmartSim `log` module. The SmartSim logger, backed by Python logging, enables
real-time logging of experiment activity **to stdout** and/or **to file**, with
multiple verbosity levels for categorizing log messages.

Users may instruct SmartSim to log certain verbosity level log messages
and omit others through the `SMARTSIM_LOG_LEVEL` environment variable. The `SMARTSIM_LOG_LEVEL`
environment variable may be overridden when logging to file by specifying a log level to
the ``log_to_file`` function. Examples walking through logging :ref:`to stdout<log_to_stdout>`
and :ref:`to file<log_to_file>` are provided below.

SmartSim offers **four** log functions to use within the Python driver script. The
below functions accept string messages:

- ``logger.error``
- ``logger.warning``
- ``logger.info``
- ``logger.debug``

The `SMARTSIM_LOG_LEVEL` environment variable accepts **four** log levels: `quiet`,
`info`, `debug` and `developer`. Setting the log level in the environment (or via the override function)
controls the log messages that are output at runtime. The log levels are listed below from
least verbose to most verbose:

- level: `quiet`
   - The `quiet` log level instructs SmartSim to print ``error`` and ``warning`` messages.
- level: `info`
   - The `info` log level instructs SmartSim to print ``info``, ``error`` and ``warning`` messages.
- level: `debug`
   - The `debug` log level instructs SmartSim to print ``debug``, ``info``, ``error`` and ``warning`` messages.
- level: `developer`
   - The `developer` log level instructs SmartSim to print ``debug``, ``info``, ``error`` and ``warning`` messages.

.. note::
    Levels `developer` and `debug` print the same log messages. The `developer` log level is intended for use
    during code development and signifies highly detailed and verbose logging.

.. note::
    `SMARTSIM_LOG_LEVEL` defaults to log level `info`. For SmartSim log API examples, continue to the :ref:`Examples<log_ex>` section.

.. _log_ex:

========
Examples
========
.. _log_to_stdout:

-------------
Log to stdout
-------------
The ``get_logger`` function in SmartSim enables users to initialize a logger instance.
Once initialized, a user may use the instance to log a message using one of the four
logging functions.

To use the SmartSim logger within a Python script, import the required `get_logger`
function from the `log` module:

.. code-block:: python

      from smartsim.log import get_logger

Next, initialize an instance of the logger and provide a logger `name`:

.. code-block:: python

      logger = get_logger("SmartSim")

To demonstrate full functionality of the SmartSim logger, we include all log
functions in the Python driver script with log messages:

.. code-block:: python

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

Execute the script *without* setting the `SMARTSIM_LOG_LEVEL`. Remember that `SMARTSIM_LOG_LEVEL`
defaults to `info`. When we execute the script, the following messages will print to stdout:

.. code-block:: bash

    11:15:00 system.host.com SmartSim[130033] INFO This is a message
    11:15:00 system.host.com SmartSim[130033] ERROR This is an error message
    11:15:00 system.host.com SmartSim[130033] WARNING This is a warning message

Notice that the `debug` function message was filtered. This is because by using
a lower verbosity level (`info`), we instruct SmartSim to omit the higher verbosity level messages (`debug` and `developer`).

Next, set `SMARTSIM_LOG_LEVEL` to `debug`:

.. code-block:: bash

    export SMARTSIM_LOG_LEVEL=debug

When we execute the script again,
the following messages will print to stdout:

.. code-block:: bash

    11:15:00 system.host.com SmartSim[65385] INFO This is a message
    11:15:00 system.host.com SmartSim[65385] DEBUG This is a debug message
    11:15:00 system.host.com SmartSim[65385] ERROR This is an error message
    11:15:00 system.host.com SmartSim[65385] WARNING This is a warning message

Notice that all log messages print to stdout. By using a higher verbosity level (`debug`),
we instruct SmartSim to print all log functions at and above the level.

Next, set `SMARTSIM_LOG_LEVEL` to `quiet` in terminal:

.. code-block:: bash

    export SMARTSIM_LOG_LEVEL=quiet

When we run the program once again, the following output is printed
to stdout:

.. code-block:: bash

    11:15:00 system.host.com SmartSim[65385] ERROR This is an error message
    11:15:00 system.host.com SmartSim[65385] WARNING This is a warning message

Notice that the `info` and `debug` log functions were filtered. This is because by using
the least verbose level (`quiet`), we instruct SmartSim to omit messages at higher verbosity levels
(`info`, `debug` and `developer`).

To finish the example, set `SMARTSIM_LOG_LEVEL` to `info` in terminal:

.. code-block:: bash

    export SMARTSIM_LOG_LEVEL=info

When we execute the script, the following messages will print
to stdout:

.. code-block:: bash

    11:15:00 system.host.com SmartSim[130033] INFO This is a message
    11:15:00 system.host.com SmartSim[130033] ERROR This is an error message
    11:15:00 system.host.com SmartSim[130033] WARNING This is a warning message

Notice that the same messages were logged to stdout as when we ran the script with the default value `info`.
SmartSim omits messages at higher verbosity levels (`debug` and `developer`).

.. _log_to_file:

---------------
Logging to File
---------------
The ``log_to_file`` function in SmartSim allows users to log messages
to a specified file by providing a file name or relative file path. If the file name
passed in does not exist, SmartSim will create the file. If the program is re-executed with the same
file name, the file contents will be overwritten.

To demonstrate, begin by importing the functions `get_logger` and `log_to_file` from the `log` module:

.. code-block:: python

      from smartsim.log import get_logger, log_to_file

Initialize a logger for use within the Python driver script:

.. code-block:: python

      logger = get_logger("SmartSim")

Invoke the ``log_to_file`` function to instruct SmartSim to create a file named `logger.out`
to write log messages to:

.. code-block:: python

      log_to_file("logger.out")

For the example, we add all log functions to the script:

.. code-block:: python

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

Remember that the default value for the `SMARTSIM_LOG_LEVEL` variable is `info`.
Therefore, we will not set the environment variable and instead rely on the
default.

When we execute the Python script, a file named `logger.out` is created in our working
directory with the listed contents:

.. code-block:: bash

    11:15:00 system.host.com SmartSim[10950] INFO This is a message
    11:15:00 system.host.com SmartSim[10950] ERROR This is an error message
    11:15:00 system.host.com SmartSim[10950] WARNING This is a warning message

Notice that the `debug` function message was filtered. This is because by using
a lower verbosity level (`info`), we instruct SmartSim to omit higher verbosity messages (`debug` and `developer`).

In the same Python script, add a log level to the ``log_to_file`` as a input argument:

.. code-block:: python

      log_to_file("logger.out", "quiet")

When we execute the Python script once again, SmartSim will override the `SMARTSIM_LOG_LEVEL`
variable to output messages of log level `quiet`. SmartSim will overwrite the contents
of `logger.out` with:

.. code-block:: bash

    11:15:00 system.host.com SmartSim[10950] ERROR This is an error message
    11:15:00 system.host.com SmartSim[10950] WARNING This is a warning message