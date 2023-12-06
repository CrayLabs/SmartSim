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

There are three logging levels indicating the severity of log messages.
Each log level is associated with a function that can be used to log events at
the associated level. The log levels, in order of increasing
severity, are as follows:

1. quiet
2. info
3. debug

Next, we place all four logging functions within the Python script:

.. code-block:: python

      logger.info("This is a message")
      logger.debug("This is a debug message")
      logger.error("This is an error message")
      logger.warning("This is a warning message")

When we run the script, the following output prints to stdout::
    19:52:05 osprey.us.cray.com SmartSim[130033] INFO This is a message
    19:52:05 osprey.us.cray.com SmartSim[130033] ERROR This is an error message
    19:52:05 osprey.us.cray.com SmartSim[130033] WARNING This is a warning message

Notice that the `debug` function did not print. This is because
`SMARTSIM_LOG_LEVEL` is defaulted to `info`. The `SMARTSIM_LOG_LEVEL` environment
variable controls what log level messages are printed. Given a log level, SmartSim will
ignore higher severity levels if present. Such as here, since `debug` is higher than `info`,
it is ignored.

Let's set the `SMARTSIM_LOG_LEVEL` to `debug` and check the output of the program.
Set the environment variable in the terminal like so::
    export SMARTSIM_LOG_LEVEL=debug

When we run the program once again, the following output is printed to stdout::
    20:11:12 osprey.us.cray.com SmartSim[65385] INFO This is a message
    20:11:12 osprey.us.cray.com SmartSim[65385] DEBUG This is a debug message
    20:11:12 osprey.us.cray.com SmartSim[65385] ERROR This is an error message
    20:11:12 osprey.us.cray.com SmartSim[65385] WARNING This is a warning message

Notice that all messages are visible now. This is because we specified the highest
severity level to `SMARTSIM_LOG_LEVEL`.

Let's set the `SMARTSIM_LOG_LEVEL` to `quiet` and check the output of the program::
    export SMARTSIM_LOG_LEVEL=quiet

The output appears as follows::
    21:07:40 osprey.us.cray.com SmartSim[10950] ERROR This is an error message
    21:07:40 osprey.us.cray.com SmartSim[10950] WARNING This is a warning message

SmartSim ignores `debug` and `info` logs, and will exclusively print `errors` and
`warnings`.

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