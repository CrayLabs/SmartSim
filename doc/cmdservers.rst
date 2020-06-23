

Command Server
--------------
SmartSim can be used interactively which can be particularly useful when
developing new experiments or prototyping online analysis. The command server
enables this development to be done at scale even when scripts or Jupyter
notebooks are deployed on compute nodes which may not have access to the scheduler.

SmartSim uses ZMQ to send messages from the compute node to head node running a
command server. All commands that managing allocations or launching the
experiment originate on the compute node, but are actually executed on this
login node. The Jupyter instance on the compute node has full access to every
aspect of the experiment including the orchestrator.

