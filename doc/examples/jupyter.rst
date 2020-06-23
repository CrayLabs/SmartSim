

Connect Jupyter to a Simulation
-------------------------------
Users can interact with a SmartSim experiment via Jupyter notebooks for online
analysis and visualization.

First, start a Jupyter notebook or Jupyter lab server on a compute node and
connect to it. This can easily be done with Cray Urika-XC for analysis
scale that needs to be done at scale. Alternatively, JupyterHub can be used or
a slurm job can be submitted manually port forwards from the compute node.

In most of those cases, a command server will need to be started within the
notebook to handle the parts of the SmartSim experiment that need to interact
with the scheduler or resource allocations. 

This example uses the command server and a notebook on a compute node to
setup and run a hypothetical Python-based model called `example_model.py`
that uses the Python SmartSim client to send a 2D field filled with random
numbers every timestep, t=0 to t=4.

The below code can be run in a Jupyter notebook cell to setup a SmartSim
experiment to visualize the output from this simple model with the command
server to be run on a node with a hostname `head_node001`. This experiment
also uses a 3-node clustered database (the orchestrator) to store the output
from the model. When `experiment.start()` is called, SmartSim will request an
allocation that satisfies the requested resources of both the orchestrator and
the model script itself.

.. code:: python

    import matplotlib.pyplot as plt
    from smartsim import Experiment, Client

    experiment = Experiment("random_numbers")
    experiment.init_remote_launcher(addr="head_node001")
    model_settings = {
        "ppn":1,
        "nodes":1,
        "duration":"1:00:00",
        "executable":"python example_model.py",
        }
    experiment.create_model("example_model", run_settings = model_settings)
    orc = experiment.create_orchestrator(cluster_size=3)
    experiment.start()

At this point the experiment will be running and will sending keys to the
database of the form :code:`array_{timestamp}` by calling
`client.put_array_nd_float64(array_{timestamp}, np.random.random(100))`.

In order to access the database, the jupyter notebook will also need to start
its own Python SmartSim client and setup all of the necessary connections:

.. code:: python

    client = Client(cluster=True)
 
If there is a known key in the database, the client within the notebook can
access it directly. By specifying `wait=True`, the retrieval will block further
execution until the key exists in the database. A new figure can be made and
plotted as soon as these data are available.

.. code:: python

    key_template = 'array_{timestamp}'

    for time in range(5):
        array = client.get_array_nd_float64(key_template.format(timestamp=time, wait = True)
        plt.figure()
        plt.pcolormesh(array)