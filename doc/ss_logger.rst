********
Overview
********
The SmartSim library provides logging functionality that users may quickly
integrate into an experiment. A logger helps debug your experiment
as well as keep track of the experiment flow. On this page, we demonstrate
how to setup the a SmartSim logger.

First, import the `get_logger` module.
Initialize a logger instance by providing a `name` to the function ``get_logger()``:

.. code-block:: python

      from smartsim.log import get_logger

      logger = get_logger("example_logger")

There are three logging levels indicating the severity of log messages.
Each log level is associated with a function that can be used to log events at
that level of severity. The defined levels, in order of increasing
severity, are the following:

1. quiet
2. info
3. debug

Next, we demonstrate using the logging level functions within the file:

.. code-block:: python

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

When we run the program, the following output appears::
    19:52:05 osprey.us.cray.com SmartSim[130033] INFO This is a message
    19:52:05 osprey.us.cray.com SmartSim[130033] ERROR This is an error message
    19:52:05 osprey.us.cray.com SmartSim[130033] WARNING This is a warning message

Notice that the `debug` function did not print to stdout. This is because
`SMARTSIM_LOG_LEVEL` is defaulted to `info`. The `SMARTSIM_LOG_LEVEL` environment
variable controls which messages are outputted with log levels. SmartSim will
ignore higher severity levels if present.

Let's set the `SMARTSIM_LOG_LEVEL` to `debug` and check the output of the program.
We set the environment variable like so::
    export SMARTSIM_LOG_LEVEL=debug

When we run the program again, the following output is printed to stdout::
    20:11:12 osprey.us.cray.com SmartSim[65385] INFO This is a message
    20:11:12 osprey.us.cray.com SmartSim[65385] DEBUG This is a debug message
    20:11:12 osprey.us.cray.com SmartSim[65385] ERROR This is an error message
    20:11:12 osprey.us.cray.com SmartSim[65385] WARNING This is a warning message

Notice that all messages are visible now.

Let's set the `SMARTSIM_LOG_LEVEL` to `quiet` and check the output of the program::
    export SMARTSIM_LOG_LEVEL=quiet

The output appears as follows::
    21:07:40 osprey.us.cray.com SmartSim[10950] ERROR This is an error message
    21:07:40 osprey.us.cray.com SmartSim[10950] WARNING This is a warning message

Lastly, you may also instruct SmartSim to write the log messages
to a file by using the `log_to_file()` function.
Below we show the same program from above, however, we add the `log_to_file()`
and pass in the name of the file we would like to write to. In this case,
the file name is `"logger.out"`

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

Note that the `SMARTSIM_LOG_LEVEL` is still set to `debug` and therefore all levels will prints.