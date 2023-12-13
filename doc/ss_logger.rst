********
Overview
********
The SmartSim library provides logging functionality that users can quickly
integrate into an experiment. A SmartSim logger helps debug your experiment
and can help keep track of the experiment flow. On this page, we demonstrate
how to setup the a SmartSim logger within a python script.

First, import the `get_logger` module.
Initialize a logger instance by providing a `name` to the function ``get_logger()``:

.. code-block:: python

      from smartsim.log import get_logger

      logger = get_logger("example_logger")

SmartSim uses log levels to indicate the severity of log messages.
Each log level is associated with a logger helper function that is used to log events at
that level. The levels, in order of increasing verboseness, are as follows:
    1. `quiet`
    2. `info`
    3. `debug`

The levels are set through the `SMARTSIM_LOG_LEVEL` environment variable
which will be demonstrated later int he example.
The level `info` is the default log level.

SmartSim offers four logging helper functions. Let's add to the all four
to the Python script:

.. code-block:: python

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

Earlier we mentioned that log levels control the log helper functions.
This is done by setting the `SMARTSIM_LOG_LEVEL` environment variable.
For example, set `SMARTSIM_LOG_LEVEL` to `quiet`::
    export SMARTSIM_LOG_LEVEL=quiet

When we execute the Python script with `SMARTSIM_LOG_LEVEL=quiet`,
the following messages will print::
    21:07:40 osprey.us.cray.com SmartSim[10950] ERROR This is an error message
    21:07:40 osprey.us.cray.com SmartSim[10950] WARNING This is a warning message

Notice that the `info` and `debug` messages were ignored. This is because by setting
a lower log level (`quiet`), we instruct SmartSim to ignore the higher levels (`debug` and `info`).
SmartSim will always print `warning` and `error` messages to stdout.

Next, set `SMARTSIM_LOG_LEVEL` to `info`::
    export SMARTSIM_LOG_LEVEL=info

When we run the script with `SMARTSIM_LOG_LEVEL=info`,
the following output appears::
    19:52:05 osprey.us.cray.com SmartSim[130033] INFO This is a message
    19:52:05 osprey.us.cray.com SmartSim[130033] ERROR This is an error message
    19:52:05 osprey.us.cray.com SmartSim[130033] WARNING This is a warning message

Above, we have instructed SmartSim to print all `info` messages but ignore higher
verbose messages (`debug`).

Let's set the `SMARTSIM_LOG_LEVEL` to `debug` and check the output of the program.
Set the environment variable in the terminal like so::
    export SMARTSIM_LOG_LEVEL=debug

When we run the program once again, the following output is printed to stdout::
    20:11:12 osprey.us.cray.com SmartSim[65385] INFO This is a message
    20:11:12 osprey.us.cray.com SmartSim[65385] DEBUG This is a debug message
    20:11:12 osprey.us.cray.com SmartSim[65385] ERROR This is an error message
    20:11:12 osprey.us.cray.com SmartSim[65385] WARNING This is a warning message

Notice that all log messages are visible since we set `SMARTSIM_LOG_LEVEL`
to the highest log level.

You may also instruct SmartSim to write the log messages
to a file by using the `log_to_file()` function.
Below we show the same program from above, however, we implement the `log_to_file()`
by passing in the name of the file we would like SmartSim to create and
write to. In this case, the file name is `"logger.out"`

.. code-block:: python

      from smartsim.log import get_logger

      logger = get_logger("example_logger")

      log_to_file("logger.out")

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

A file named `logger.out` is created in our working directory with the listed contents::
    21:07:40 osprey.us.cray.com SmartSim[10950] INFO This is a message
    21:07:40 osprey.us.cray.com SmartSim[10950] DEBUG This is a debug message
    21:07:40 osprey.us.cray.com SmartSim[10950] ERROR This is an error message
    21:07:40 osprey.us.cray.com SmartSim[10950] WARNING This is a warning message

Note that the `SMARTSIM_LOG_LEVEL` is still set to `debug` and therefore all levels print.
If the program is reran with the same file name, the file contents will be overwritten.