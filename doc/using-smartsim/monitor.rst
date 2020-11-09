
**********************
Monitoring Experiments
**********************

SmartSim allows users to monitor the status of SmartSim models, nodes, and
orchestrators that have been launched in an experiment.  The ``Experiment``
class provides a continuous status check with ``experiment.poll()`` that
reports entity status and blocks execution until all entities are no longer
running.  A non-blocking status check can be performed with
``experiment.get_status()`` which will return the status of the launched
entity.
