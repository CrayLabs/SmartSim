******
Dragon
******

========
Overview
========

Dragon is a composable distributed run-time targeting HPC workflows. In SmartSim,
Dragon can be used as a launcher, within a Slurm or PBS allocation or batch job.
The SmartSim team collaborates with the Dragon team to develop an efficient
launcher which will enable fast, interactive, and customized execution of
complex workflows on large HPC systems. As Dragon is scheduler-agnostic,
the same SmartSim script using Dragon as a launcher can be run indifferently
on a Slurm or PBS system. Support for additional schedulers is coming soon.

.. warning::
    The Dragon launcher is currently in its early development stage and should be treated as
    a prototype implementation. Your assistance is invaluable in identifying any issues
    encountered during usage and suggesting missing features for implementation. Please
    provide feedback in the form of a created issue on the
    `SmartSim issues GitHub page <https://github.com/CrayLabs/SmartSim/issues>`_.
    The :ref:`Known Issues section<dragon_known_issues>` is also a good starting
    point when troubleshooting workflows run using the Dragon launcher.

=====
Usage
=====
To use Dragon, you need to install it in your current Python environment. This can
be accomplished by providing the ``--dragon`` flag to the ``smart build`` command, as
detailed in the :ref:`Dragon Install <dragon_install>`. Note that specifying the device
configuration is also required for a proper build.

After installation, specify Dragon as the launcher when creating an ``Experiment``:

.. code-block:: python

    exp = Experiment(name="dragon-example", launcher="dragon")

Dragon introduces its own run settings class, ``DragonRunSettings``, which allows users to
specify nodes and tasks per node for a ``Model``. For instance, continuing from the previous
example:

.. code-block:: python

    # Because "dragon" was specified as the launcher during Experiment initialization,
    # create_run_settings will return a DragonRunSettings object
    rs = exp.create_run_settings(exe="mpi_app",
                                 exe_args=["--option", "value"],
                                 env_vars={"MYVAR": "VALUE"})
    # Above we specify the executable (exe), executable arguments (exe_args)
    # and environment variables (env_vars)

    # Sets the number of nodes for this job
    rs.set_nodes(4)
    # Set the tasks per node for this job
    rs.set_tasks_per_node(3)
    # Initialize the Model and pass in the DragonRunSettings object
    mpi_app = exp.create_model("MPI_APP", run_settings=rs)
    # Start the Model
    exp.start(mpi_app)

SmartSim supports ``DragonRunSettings`` with ``Model``, ``Ensemble`` and ``Orchestrator`` entities.
In the next sections, we detail how Dragon is integrated into SmartSim.

For more information on HPC launchers, visit the :ref:`Run Settings<run_settings_hpc_ex>` page.

=================
The Dragon Server
=================

Dragon can initiate processes on any available resource within an allocation. To facilitate
this, SmartSim initializes the Dragon infrastructure whenever a ``Model`` is launched and maintains
it until the parent ``Experiment`` concludes. To facilitate interaction with processes managed by
Dragon, SmartSim establishes a command server within the Dragon infrastructure. This server,
known as the `Dragon Server`, is responsible for executing commands to start or stop processes
and to query their status.

Sharing the Dragon Server across Experiments
============================================

Currently, SmartSim supports only one Dragon server per allocation. Consequently,
if multiple Experiments need to run within the same allocation, the Dragon server
must be shared among them. By default, the server starts from a subdirectory
of the ``Experiment`` path, where it creates a configuration file.
To enable server sharing, users can specify a custom path
from which the server should be launched. This can be achieved by setting the
environment variable ``SMARTSIM_DRAGON_SERVER_PATH`` to an existing absolute path.
Each ``Experiment`` will then search for the configuration file in the specified path
and initiate a new server instance only if the file is not found.

Dragon's High-Speed Transport Agents
====================================

On systems equipped with the HPE Slingshot interconnect, Dragon utilizes High-Speed
Transport Agents (HSTA) by default for internal messaging within the infrastructure
launched by SmartSim. On systems without the HPE Slingshot interconnect,
TCP agents are employed. To specify the use of TCP agents, users must set the environment
variable ``SMARTSIM_DRAGON_TRANSPORT`` to ``tcp`` prior to executing the Experiment.
To specify HSTA, ``SMARTSIM_DRAGON_TRANSPORT`` can be set to ``hsta`` or left unset.

=============
Communication
=============

SmartSim and the Dragon Server communicate using `ZeroMQ <https://zeromq.org/>`_.

Similar to other communication protocols, defining timeouts for send and receive operations
is crucial in SmartSim. SmartSim configures default timeouts that have been tested on various
systems, such as Polaris, Perlmutter, and other HPE Cray EX and Apollo systems.
However, if you encounter failed communication attempts, adjusting the timeouts may
be necessary. You can adjust these timeouts by setting the corresponding environment variables:

- **Server Start-up Timeout**: This timeout specifies the duration the SmartSim ``Experiment``
  waits when the server is initially started. It must accommodate the time required for
  Dragon to set up the infrastructure, which varies based on the system's workload manager
  response time. The default timeout is `"300000"` milliseconds (i.e., five minutes), and you can override
  it using the ``SMARTSIM_DRAGON_STARTUP_TIMEOUT`` environment variable.

- **Server Send and Receive Timeout**: This timeout dictates how long SmartSim and the Dragon
  server wait to send or receive a message. The default timeout is `"30000"` milliseconds (i.e., 30 seconds),
  and you can modify it using the ``SMARTSIM_DRAGON_TIMEOUT`` environment variable.

Setting any timeout to "-1" will result in an infinite waiting time, causing the execution to
block until the communication is completed, potentially hanging indefinitely if issues occur.

It's important to note that all communications are secured with `elliptic curve cryptography <http://curvezmq.org/>`_.
SmartSim generates the necessary key-pairs and stores them in the user's home directory by
default. However, you can specify an alternative absolute path using the ``SMARTSIM_KEY_PATH``
environment variable.

.. _dragon_known_issues:

============
Known issues
============

As previously noted, the integration of SmartSim with Dragon is still in its early
development stage, and there are known issues that may result in unexpected behavior
during runs:

- **Incomplete cleanup of Dragon resources**: When SmartSim exits, it attempts to properly
  shut down the Dragon infrastructure to clean up associated resources, such as shared memory
  segments, and terminate all processes. However, in rare cases, if the execution is
  abruptly interrupted (e.g., by terminating SmartSim with ``SIGKILL``), the cleanup process
  may be incomplete, leaving processes like the Dragon overlay network active on the node
  where SmartSim was executed (which could be a login node, particularly on Slurm systems).
  If this occurs, you can use the following command to address the issue:

  .. code-block::

    smart teardown --dragon

  This command will terminate all Dragon-related processes, release shared memory segments,
  but also terminate all Python processes associated with your username.

- **Dragon server not starting**: This issue may arise due to two main reasons:

  1. *HSTA not available on the system*: Try setting the environment variable
     ``SMARTSIM_DRAGON_TRANSPORT`` to ``tcp``.
  2. *System or Workload Manager too busy*: Attempt to mitigate this by setting the environment
     variable ``SMARTSIM_DRAGON_STARTUP_TIMEOUT`` to a larger value or ``"-1"``.

- **MPI-based applications hanging**: To run MPI-based applications on Dragon, Cray PMI or
  Cray PALS must be available on the system. This limitation is currently being addressed.


Interested users can learn more about the Dragon project at the external
`Dragon documentation page <https://dragonhpc.github.io/dragon/doc/_build/html/index.html>`_.