

**********************
Starting an Experiment
**********************

SmartSim supports launching simulations, databases, and analysis packages on
heterogeneous, computational resources with users specifying hardware groups
on which SmartSim entities are launched. On execution, SmartSim will create
the orchestrator (database) and then execute the models and nodes.  The launching of the
SmartSim experiment is non-blocking, and as a result, the user is free to
execute other commands or launch additional experiments in the same Python script.
If the user would like to wait for the experiment to complete, the status of the
SmartSim models and nodes can be monitored with a blocking poll command through the SmartSim API.
