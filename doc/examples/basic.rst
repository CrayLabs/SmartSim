
***********
Quick Start
***********

In this example, we will use the `CP2K <https://www.cp2k.org/>`_ to run
a quantum chemistry simulation through SmartSim. The example will focus
on the basics of how to get a new model running in SmartSim and the
``Experiment`` interface.

.. note::
   We will be using the Slurm launcher in this example. If your machine
   does not support Slurm, this example will not run.

To run this experiment, the CP2K model must be compiled and installed
on the machine where the SmartSim script will be executed.


Creating an Experiment
======================

Initializing an ``Experiment`` is simple as it only requires two arguments:
the name of the experiment, and the launcher backend for SmartSim.
In this example, we are launching our model using Slurm and we are going
to name the experiment ``h2o`` as we are going to be running the
h2o configuration of CP2K.

.. code-block:: python

    from smartsim import Experiment
    experiment = Experiment("h2o", launcher="slurm")

The options for the launcher backend are "slurm", which launches through
the Slurm workload manager; "local", which launches onto the local
architecture. For more information on launcher backend see
`the launchers documentation <../launchers.html>`_


Obtaining an Allocation
-----------------------

To run a model with the Slurm launcher, we first need an allocation for that model to run on.
Normally, in Slurm, this is done through direct interaction with the
Slurm daemon through the ``salloc`` command.

SmartSim provides an interface to obtain allocations programmatically
so that each script will contain the exact configuration upon which
it was launched, including the allocation information.

To obtain an allocation in SmartSim, we use the ``Experiment.get_allocation()``
method. In this case, we are requesting a single node on the ``gpu`` partition
of our cluster.

.. code-block:: python

    alloc = experiment.get_allocation(nodes=1, partition="gpu")

The id of the allocation is returned as a string to the user so that
they can specify what entities should run on which allocations
obtained by SmartSim. In later sections, we will show how to use the returned
allocation id to specify the allocation on which SmartSim entities are launched


The keyword arguments to the ``get_allocation`` mimic the exact names of command
line arguments that would be normally used with the workload manager, in this case Slurm.
This includes command line arguments that do not have a value associated
with them. In such cases, users can place a value of ``None`` for that argument.
An example of a more complicated allocation that is not used in this
example is given below:

.. code-block:: python

    experiment.get_allocation(nodes=5, constraint="haswell", partition="debug",
                              exclusive=None, time="10:00:00")

Once again, the above line is not used in this example, but it should help
clarify how specific types of allocations can be obtained through SmartSim.


Constructing the Model
----------------------

For each model we want to run, we need one corresponding model object within
SmartSim. To create a model object, we call ``Experiment.create_model()``.
There are several arguments to the ``create_model`` function that we will
need to define before creating it.

The ``run_settings`` of a model tell SmartSim how to run the model on the
available resources of the machine. Only two fields must be present within
the run settings: ``executable`` and ``alloc``(Slurm only). Run settings
follow the same initalization pattern as getting allocations within
SmartSim with a few additional arguments. SmartSim provides additional
fields to add in model configuration:

 1) ``exe_args``: for arguments to the executable
 2) ``ppn``: for processes per node (same as --ntasks-per-node)
 3) ``env_vars``: for additional environment variables to run the simulation with
 4) ``alloc``: for specifying the allocation to run the entity on.

Every other argument (that would normally be provided to srun) can be
specified as a key-value pair in the ``run_settings`` dictionary just
like getting allocations.

In this example we have the following ``run_settings`` for our CP2K model.

.. code-block:: python

    # Define how and where the CP2K models should be
    # executed. Allocation is provided with the "alloc" keyword
    run_settings = {"executable": "cp2k.psmp",
                    "partition": "gpu",
                    "exe_args": "-i h2o.inp",
                    "nodes": 1,
                    "alloc": alloc}

Notice that we include the variable ``alloc`` that we obtained earlier
in the ``get_allocation`` method so that our model will run on the
GPU allocation we already obtained.

We are now ready to put our model into the experiment:

.. code-block:: python

    # Define how and where the CP2K models should be
    # executed. Allocation is provided with the "alloc" keyword
    run_settings = {"executable": "cp2k.psmp",
                    "partition": "gpu",
                    "exe_args": "-i h2o.inp",
                    "nodes": 1,
                    "alloc": alloc}

    # Create the model with settings listed above
    model = experiment.create_model("h2o-1",
                                    run_settings=run_settings)

We have now told SmartSim to obtain an allocation and to configure
and run our CP2K model on that allocation.


Starting and Monitoring an Experiment
-------------------------------------

Now that our experiment is configured, we need to tell SmartSim to
run the experiment. The method for this is ``Experiment.start()``.
Any number of SmartSim entites can be specified in the call to
start as long as they have been created by that Experiment object.
If no entities are listed as arguments in the method call, all
entites that have been created (e.g. our model) will be executed.

.. code-block:: python

    experiment.start()

The ``Experiment.start()`` method is non-blocking so that as soon
as the model is started, we can use the ``Experiment`` interface
to monitor execution and even stop execution programmatically.

There are three methods to monitor the progress and status of a model
or entity launched through an Experiment.

 1) ``Experiment.poll()``
 2) ``Experiment.finished()``
 3) ``Experiment.get_status()``

``Experiment.get_status()`` retrieves the status of an entity, such as
our model, from the SmartSim launcher backend. This will include
various different statuses for why an entity launched through
SmartSim has succeeded or failed.

``Experiment.finished()`` is a non-blocking call that contacts the
SmartSim launcher backend and checks to see if the entity has completed
execution.

``Experiment.poll()`` is a blocking method that will continually
ping the SmartSim launched backend for status updates at a specific
interval provided as an argument. The status updates are logged
to standard output but can also be logged in a file.

In this example we will use ``Experiment.poll()`` to continually check
the status of our model every ten seconds as follows:

.. code-block:: python

    experiment.poll(interval=10)

Given that the call to ``Experiment.poll()`` is blocking, the script
will wait until our model is finished.

After our model is completed, we want to release the allocation
obtained by SmartSim. We can do this as follows:

.. code-block:: python

    experiment.release()


Lastly, to print a summary of what we ran, we can print the experiment
object after the completion of the experiment.

.. code-block:: python

    print(experiment)



Experiment Script
=================

Bringing all the peices above together we get the following script.

.. code-block:: python

    from smartsim import Experiment

    # Initialize the experiment using the default launcher "slurm"
    experiment = Experiment("h2o", launcher="slurm")

    # Get an allocation through Slurm to launch entities
    # extra arguments can be specified through kwargs
    # e.g. "qos"="interactive"
    alloc = experiment.get_allocation(nodes=1, partition="gpu")


    # Define how and where the CP2K model should be
    # executed. Allocation is provided with the "alloc" keyword
    run_settings = {"executable": "cp2k.psmp",
                    "partition": "gpu",
                    "exe_args": "-i h2o.inp",
                    "nodes": 1,
                    "alloc": alloc}

    # Create the model with settings listed above
    model = experiment.create_ensemble("h2o-1",
                                        run_settings=run_settings)


    # launch the model. Since we dont specify which
    # ensemble to run, launch all entities defined within
    # the experiment.
    experiment.start()

    # Since Experiment.start() is non-blocking when using
    # the Slurm launcher, poll slurm for status updates.
    experiment.poll()

    # release all allocations obtained by this experiment
    experiment.release()

    # print the experiment summary
    print(experiment)



