**************
SmartDashboard
**************


``SmartDashboard`` is an add-on to SmartSim that provides a dashboard to help users understand
and monitor their SmartSim experiments in a visual way. Configuration, status, and logs
are available for all launched entities within an experiment for easy inspection.

A ``Telemetry Monitor`` is a new background process that is launched along with the experiment
that helps ``SmartDashboard`` properly display data. Experiment metadata is also stored in
the ``.smartsim`` directory, a hidden folder for internal api use and used by the dashboard.
Deletion of the experiment folder will remove all experiment metadata.



Installation
============

It's important to note that SmartDashboard only works while using SmartSim, so
SmartSim will need to be installed as well. SmartSim installation docs can be
found `here <https://www.craylabs.org/docs/installation_instructions/basic.html>`_.


User Install:

Run ``pip install git+https://github.com/CrayLabs/SmartDashboard.git`` to install
SmartDashboard without cloning the repository.

Developer Install:

Clone the ``SmartDashboard`` repository at https://github.com/CrayLabs/SmartDashboard.git.
Once cloned, ``cd`` into the repository and run:

``pip install -e .``


Running SmartDashboard
======================

After launching a SmartSim experiment, the dashboard can be launched using SmartSim's CLI.
  
``smart dashboard --port <port number> --directory <experiment directory path>``
  
The port can optionally be specified, otherwise the dashboard port will default to `8501`.
The directory must be specified and should be a relative or absolute path to the created experiment directory.

--port, -p        port number
--directory, -d   experiment directory



Example workflow:


.. code-block:: bash

  # directory before running experiment  
  ├── hello_world.py


.. code-block:: python

  # hello_world.py
  from smartsim import Experiment

  exp = Experiment("hello_world_exp", launcher="auto")
  run = exp.create_run_settings(exe="echo", exe_args="Hello World!")
  run.set_tasks(60)
  run.set_tasks_per_node(20)

  model = exp.create_model("hello_world", run)
  exp.start(model, block=True, summary=True)
  
 
.. code-block:: bash
    
  # in interactive terminal
  python hello_world.py
  

.. code-block:: bash

  # directory after running experiment
  ├── hello_world.py
  └── hello_world_exp


By default, ``hello_world_exp`` is created in the directory of the driver script.


.. code-block:: bash

  # in a different interactive terminal
  smart dashboard --port 8888 --directory hello_world_exp
 

The dashboard will automatically open in a browser at port 8888. The dashboard is also 
persistent, meaning that a user can still launch and use the dashboard even after the experiment has completed.


Using SmartDashboard
====================

Once the dashboard is launched, a browser will open to ``http://localhost:<port>``. 
SmartDashboard currently has two tabs on the left hand side.  
  
``Experiment Overview:`` This tab is where configuration information, statuses, and 
logs are located for each launched entity of the experiment. The ``Experiment`` 
section displays configuaration information for the overall experiment. In the ``Applications`` 
section, also known as SmartSim ``Models``, select a launched application to see its status, 
what it was configured with, and its logs. The ``Orchestrators`` section also provides 
configuration and status information, as well as logs per shard for a selected orchestrator. 
Finally, in the ``Ensembles`` section, select an ensemble to see its status and configuration. 
Then select any of its members to see its status, configuration, and logs.  
  
``Help:`` This tab links to SmartSim documentation and provides a SmartSim contact for support.
