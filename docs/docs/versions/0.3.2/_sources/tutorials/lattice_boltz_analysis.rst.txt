
Online Analysis
===============

Being able to visualize and interpret simulation data in real time is
invaluable for understanding the behavior of a physical system.

SmartSim can be used to stream data from Fortran, C, and C++ simulations
to Python where visualization is significantly easier and more interactive.

3.1 Lattice Boltzmann Simulation
--------------------------------

.. _implementation: https://github.com/pmocz/latticeboltzmann-python
.. _article: https://medium.com/swlh/create-your-own-lattice-boltzmann-simulation-with-python-8759e8b53b1c


This example was adapted from Philip Mocz's `implementation`_.
of the lattice Boltzmann method in Python. Since that example is licensed under GPL, so is this example.

Philip also wrote a great medium `article`_ explaining the simulation in detail.

.. |simulation| image:: ../images/latticeboltzmann.png
  :width: 700


|simulation|


3.2 Integrating SmartRedis
--------------------------

.. _Dataset: https://www.craylabs.org/docs/sr_data_structures.html#dataset

Typically HPC simulations are written in C, C++, Fortran or other high performance
languages. Embedding the SmartRedis client usually involves compiling in the
SmartRedis library into the simulation.

Because this simulation is written in Python, we can use the SmartRedis
Python client to stream data to the database.

To make the visualization easier, we use the SmartRedis `Dataset`_ object
to hold two 2D NumPy arrays. A convenience function is provided to convert
the fields into a dataset object.


.. code-block:: python

    from smartredis import Client, Dataset
    client = Client() # Addresses passed to job through SmartSim launch

    # send cylinder location only once
    client.put_tensor("cylinder", cylinder.astype(np.int8))

    for i in range(time_steps):
        # send every 5 time_step to reduce memory consumption
        if time_step % 5 == 0:
            dataset = create_dataset(time_step, ux, uy)
            client.put_dataset(dataset)

    def create_dataset(time_step, ux, uy):
        """Create SmartRedis Dataset containing multiple NumPy arrays
        to be stored at a single key within the database"""
        dataset = Dataset(f"data_{str(time_step)}")
        dataset.add_tensor("ux", ux)
        dataset.add_tensor("uy", uy)
        return dataset

This is all the SmartRedis code needed to stream the simulation data. Note that
the client does not need to have an address explicitly stated because we
are going to be launching the simulation through SmartSim.

The full simulation code can be found here # PUT IN LINK

3.2 Creating the Analysis Driver
--------------------------------


SmartSim, the infrastructure library, is used here to launch both the
database and the simulation locally, but in separate processes. The example
is designed to run on laptops, so the local launcher is used.


First the necessary libraries are imported and an `Experiment` instance is created.
Also defined is an `Orchestrator` which is the reference to the in-memory database
to be launched and used to stage data between the simulation and analysis code.

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from smartredis import Client
    from smartsim import Experiment
    from smartsim.database import Orchestrator
    from smartsim.settings import RunSettings

    exp = Experiment("finite_volume_simulation", launcher="local")
    db = Orchestrator(port=6780)

The reference to the simulation is created through a call to `Experiment.create_model()`.
The python script is "attached" to the model, such that when run directories are
created for it, the python script will be placed in that run directory.

Executable arguments are used to pass the simulation parameters to the simulation.

.. code-block:: python

    # simulation parameters and plot settings
    fig = plt.figure(figsize=(12,6), dpi=80)
    time_steps, seed = 3000, 42

    # define how simulation should be executed
    settings = RunSettings("python", exe_args=["fv_sim.py",
                                            f"--seed={seed}",
                                            f"--steps={time_steps}"])
    model = exp.create_model("fv_simulation", settings)

    # tell exp.generate to include this file in the created run directory
    model.attach_generator_files(to_copy="fv_sim.py")

    # generate directories for output, error and results
    exp.generate(db, model, overwrite=True)


The next portion starts the database and immediately connects
a client to it so that data can be retrieved by the analysis
code and plotted.

The simulation is started with `block=False`, so that the data
being streamed from the simulation can be analyzed in real time.

.. code-block:: python

    # start the database and connect client to get data
    exp.start(db)
    client = Client(address="127.0.0.1:6780", cluster=False)

    # start simulation without blocking so data can be analyzed in real time
    exp.start(model, block=False, summary=True)


SmartRedis is used to pull the Datasets created by
the simulation and use matplotlib to plot the results.

Another `Model` could have been created to plot the results and launched
in a similar manner to the simulation.

Doing so would enable the analysis application to be executed on different
resources such as GPU enabled nodes, or distributed across nodes.

This version, where the driver and analysis code coexist in the same
script, is shown for simplicity.

.. code-block:: python

    # poll until data is available
    client.poll_key("cylinder", 200, 100)
    cylinder = client.get_tensor("cylinder").astype(bool)

    for i in range(0, time_steps, 5): # plot every 5th timestep
        client.poll_key(f"data_{str(i)}", 10, 1000)
        dataset = client.get_dataset(f"data_{str(i)}")
        ux, uy = dataset.get_tensor("ux"), dataset.get_tensor("uy")

        plt.cla()
        ux[cylinder], uy[cylinder] = 0, 0
        vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        vorticity[cylinder] = np.nan
        cmap = plt.cm.get_cmap("bwr").copy()
        cmap.set_bad(color='black')
        plt.imshow(vorticity, cmap=cmap)
        plt.clim(-.1, .1)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        plt.pause(0.001)

    # Save figure
    plt.savefig('latticeboltzmann.png', dpi=240)
    plt.show()

    exp.stop(db)

The database is stopped when the simulation is done, but could persist if
the user would like to continue analyzing the data.

If the Python session dies and the user does not have access to
the `Experiment` object, the following can be called to stop any database instance.

.. code-block:: bash

    # be sure to be in Python environment where SmartSim is installed
    $(smart --dbcli) -h 127.0.0.1 -p 6780 shutdown

3.4 Running the Example
-----------------------


To run the example, be sure to have SmartSim and SmartRedis installed on your system.
In addition, Matplotlib and NumPy are required.

Before running, ensure your system has enough memory to hold the states of the simulation.
As it is setup right now, the database will consume just under 1Gb of memory.


.. code-block:: bash

    # (optional) activate python environment
    python driver.py

Matplotlib will interativly plot the state of the simulation while
the simulation is running. After the window is closed, SmartSim will
shutdown the database.

The following output files are created as a result of running the
online analysis example

.. code-block:: text

    finite_volume_simulation
    ├── database
    │   ├── orchestrator_0.err
    │   ├── orchestrator_0.out
    │   └── smartsim_db.dat
    └── fv_simulation
        ├── fv_sim.py
        ├── fv_simulation.err
        └── fv_simulation.out

    2 directories, 6 files
