
**************
Command Server
**************

SmartSim can be used interactively which can be particularly useful when
developing new experiments or prototyping online analysis. The command server
enables this development to be done at scale even when scripts or Jupyter
notebooks are deployed on compute nodes which may not have access to the scheduler.

SmartSim uses ZMQ to send messages from the compute node to a head node
running the command server. All commands that managing allocations or
launching the experiment originate on the compute node, but are actually
executed on this login node. The Jupyter instance on the compute node has
full access to every aspect of the experiment including the orchestrator.

Starting the Command Server
===========================

Starting the command server is simple. Ensure that the SmartSim environment
has already been setup, navigate to the scripts directory where SmartSim was
installed and find the ``start_cmd_server.py`` script. The script requires
two arguments

 1) The IP address where you want commands to run (usually the head node)
 2) The port you want to open.

.. code-block:: bash

    python start_cmd_server.py --addr 127.0.0.1 --port 5555


Connecting to the Command Server
================================

To connect to the command server within a SmartSim script, import
the ``remote`` module and call the ``init_command_server`` function.
View the `API documentation for the remote module <api/remote.html>`_
for more information.

.. code-block:: python

   from smartsim import Experiment
   from smartsim.remote import init_command_server

   exp = Experiment("exp-on-compute-node")
   init_command_server(addr="127.0.0.1", port=5555)

   # .. continue with experiment code ...

Your experiment should now be connected to the already running command server
and all system commands, such as obtaining an allocation from Slurm, will now
emit from the server (in this case the head node) where the command server
was started.
