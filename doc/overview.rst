


Using SmartSim
--------------

In order to use SmartSim, one of two interfaces can be leveraged.

TOML Interface
==============
First, the toml interface allows the user to keep all configurations within a
single file and run end to end experiments with under 10 lines of python. The
toml interace is recommended for users who are using the library to run multiple
configurations of their model, and other settings quickly and in a reproducible fashion.

.. code-block:: TOML

  [model]
  name = "LAMMPS"                  # name of simulation model
  targets = ["atm", "atm-2"]       # target names
  model_files = ["LAMMPS/in.atm"]  # model configuration file or directory
  experiment = "lammps_atm"        # experiment name

  [atm]                            # first target
     [atm.steps]                   # name of configuration change
     value = [20, 25]              # desired configuration values

  [atm-2]                          # second target
     [atm-2.steps]                 # name of configuration change
     value = [30, 35]              # desired configuration values


The toml file above is setting up an experiment. This particular experiment is running
a particle simulation model named ``LAMMPS``. The model configuration file, ``in.atm``
specify LAMMPS to run a Axilrod-Teller-Muto (ATM) potential calculation. Two targets
are defined, ``atm`` and ``atm-2``. Each target will have two models that will each
change a single configuration (the number of steps) twice.

To generate the models for this experiment (without running them) a short python snippet
is required that can easily be run inside of a ipython shell or a jupyter notebook.

.. code-block:: python

  from smartsim import Generator, State

  STATE = State(config="/LAMMPS/simulation.toml", log_level="DEBUG")

  # Data Generation Phase
  GEN = Generator(STATE)
  GEN.generate()



In the script above, we do three things
   - import smartsim.State and smartsim.Generator
   - Define the State instance and provide the ``simulation.toml`` written above,
     and set the log level
   - Instantate the generator module, and call Generator.generate()


Python Interface
================

The second interface that provides the same functionality is through Python. The
programmatic python interface is recommended for users who wish to use a piece(s)
of the SmartSim library within their own work.

The same experiment as the toml interface above can be achieved in a single script
as follows:

.. code-block:: python

  # import needed smartsim modules
  from smartsim import Controller, Generator, State

  # intialize state to conduct experiment
  state = State(experiment="lammps_atm")

  # Create targets
  param_dict_1 = {"steps": [20, 25]}
  param_dict_2 = {"steps": [30, 35]}
  state.create_target("atm", params=param_dict_1)
  state.create_target("atm-2", params=param_dict_2)

  # Supply the generator with necessary files to run the simulation
  # and generate the specified models
  base_config = "LAMMPS/in.atm"
  GEN = Generator(state, model_files=base_config)
  GEN.generate()










