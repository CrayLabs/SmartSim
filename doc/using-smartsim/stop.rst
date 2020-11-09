

********************
Stopping Experiments
********************

Because the SmartSim experiment uses an in-memory database, the simulation data is
accessible for as long as the system allocation remains active.  However,
if the user would like to stop the experiment, the API includes the ability to stop
all or specified models, nodes, and database.  Similarly, the API allows the user
to release the system allocations(s) requested by SmartSim if that allocation is not
to be reused by follow-on experiments or for additional data analysis.
