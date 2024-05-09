******
Dragon
******

============
Introduction
============

`Dragon <https://dragonhpc.github.io/dragon/doc/_build/html/index.html>`_ is a
composable distributed run-time targeting HPC workflows. In SmartSim,
Dragon can be used as a launcher, within a Slurm or PBS allocation or batch job.

.. note::
    The Dragon launcher is at an early development stage and should be considered
    a prototype implementation. Please report any issue you encounter while using
    it and provide feedback about missing features you would like to see
    implemented.

=====
Usage
=====
To be able to use Dragon, you will have to install it in your current Python
environment. This can be done as part of the ``smart build`` step, as explained
in `_dragon_install`.

Once installed, Dragon can be selected as launcher when creating an ``Experiment``:

.. code-block:: python

    exp = Experiment(name="dragon-example", launcher="dragon")


Dragon has its own run settings class, ``DragonRunSettings``,
which can be used to specify nodes and tasks per node for a ``Model``,
for example, continuing from the previous example:

.. code-block:: python

    rs = exp.create_run_settings(exe="mpi_app",
                                 exe_args=["--option", "value"],
                                 env_vars={"MYVAR": "VALUE"})
    rs.set_nodes(4)
    rs.set_tasks_per_node(3)
    mpi_app = exp.create_model("MPI_APP", run_settings=rs)
    exp.start(mpi_app)


All types of SmartSim entities are supported, including ``Ensemble``
and ``Orchestrator``, and the underlying Dragon launcher is completely
transparent to the user. In the next sections, we will explain
how Dragon is integrated into SmartSim.

=================
The Dragon Server
=================

Dragon can start processes on any resource available within an allocation.
To do this, the so-called Dragon infrastructure needs to be started. SmartSim
instantiates the Dragon infrastructure whenever a ``Model`` needs to be started
and will keep it up and running until the parent ``Experiment`` is active.
To be able to interact with processes started through Dragon,
SmartSim spins up a command server in the Dragon infrastructure and sends commands
to it every time a process needs to be started or stopped, and to query its status.
We call this server the `Dragon Server`, and its lifecycle is managed by SmartSim.


Sharing the Dragon Server across Experiments
============================================

Currently, SmartSim only supports one Dragon server per allocation. For this reason,
if multiple ``Experiment``s need to run in the same allocation, the Dragon server needs
to be shared among them. By default, the server is started from a subdirectory of the
``Experiment`` path. To make it possible to share the server, it is possible to
specify a path from which the Server should be started through the environment variable
``SMARTSIM_DRAGON_SERVER_PATH``: every ``Experiment`` will look for the running
server in the given path and only start a new server instance if there is none running.

Dragon's High-Speed Transport Agents
====================================

On systems where the HPE Slingshot interconnect is available, Dragon can use
Higs-Speed Transport Agents (HSTA) to send internal messages. This is the default
choice for messages sent in the Dragon infrastructure started by SmartSim. On
systems where the HPE Slingshot interconnect is not available, TCP agents must be
used. To specify TCP agents, the environment variable ``SMARTSIM_DRAGON_TRANSPORT``
must be set to ``tcp`` prior to the ``Experiment`` execution.

============
Communcation
============

SmartSim and the Dragon Server communicate using `ZeroMQ <https://zeromq.org/>`_.

As with any communication protocol, some timeouts for send and receive must be defined.
SmartSim sets some default timeouts that have been tested to work on most available systems,
but if you see failed communication attempts, you may want to try to adjust the
timeouts by setting the corresponding environment variable.
The timeouts are given in milliseconds and they are defined as follows:

- server start-up timeout: the time waited by the SmartSim ``Experiment`` when the server
  is first started. This timeout must account for the time it takes Dragon to set up the
  infrastructure, which depends on the system's workload manager response time.
  Defaults to ``"300000"`` (i.e. five minutes) and can be overridden with the environment variable
  ``SMARTSIM_DRAGON_STARTUP_TIMEOUT``.

- server send and receive timeout: the time waited by SmartSim and the Dragon server to send or
  receive a message. Defaults to ``"30000"`` (i.e. 30 seconds) and can be overridden with the
  environment variable ``SMARTSIM_DRAGON_TIMEOUT``.

Setting any timeout to ``"-1"`` will result in infinite waiting time, which means that the
execution will block until the communication is completed, and hang indefinitely if something went wrong.


All communications are secured with elliptic curve cryptography,
and the key-pairs needed by the protocol are created by SmartSim and stored in the
user's home directory, unless another path is specified through the environment variable
``SMARTSIM_KEY_PATH``.


..dragon_known_issues_:

============
Known issues
============

As previosuly remarked, the SmartSim-Dragon integration is at an early development stage
and there are some known issues that can lead to errors during runs.

- *Incomplete cleanup of Dragon resources*: when SmartSim exits, it ensures that the dragon
  infrastructure is correctly shut down, so that all the associated resources (such as
  shared memory segments) are cleaned up and all processes are terminated. Nevertheless,
  in some rare cases, when the execution is interrupted abruptly (for example by terminating
  SmartSim with ``SIGKILL``), the cleanup process can be incomplete and processes
  such as the Dragon overlay network will remain active on the node where SmartSim was
  executed (which could be a login node, especially on Slurm systems). If that happens
  you can run

  .. code-block::

    smart teardown --dragon

  which will kill all Dragon related processes, return shared memory segments, but also
  kill all Python processes (associated to your user name).

- *Dragon server not starting*: this can happen because of two main reasons

  1. HSTA not available on the system: try setting the environment variable
     ``SMARTSIM_DRAGON_TRANSPORT`` to ``tcp``
  2. System or Workload Manager too busy: try setting the environment variable
     ``SMARTSIM_DRAGON_STARTUP_TIMEOUT`` to a larger value or to ``"-1"``.


- *MPI-based applications hanging*: to run MPI-based applications on Dragon, Cray PMI or Cray PALS
  must be available on the system. This is a current limitation and is actively been worked on.