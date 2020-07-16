
*********
Launchers
*********

SmartSim interfaces with a number of "launchers", i.e workload managers like
Slurm that can obtain allocations(or equivalent), and launch jobs onto
various machine architectures.

Currently, SmartSim supports launching on Slurm, and locally. Support for
PBS, and Kubernetes is under development.

Largely, the launcher backend is opaque to the user. Once the launcher
has been specified in the ``Experiment``, no further action needs to
be taken on the part of the user to configure or call the workload
manager (e.g. ``salloc`` or ``srun`` for Slurm).

To specify a specific launcher, one argument needs to be provided
to the ``Experiment`` initialization.

.. code-block:: python

    from smartsim import Experiment

    exp = Experiment("name-of-experiment", launcher="slurm") # Slurm backend
    exp = Experiment("name-of-experiment", launcher="local") # local backend


Slurm
=====

Getting Allocations
-------------------

SmartSim provides an interface to obtain allocations programmatically
so that each script will contain the exact configuration upon which
it was launched including the allocation information.

To obtain an allocation in SmartSim, we use the ``Experiment.get_allocation()``
method.

.. code-block:: python

    alloc = experiment.get_allocation(nodes=1, partition="gpu")

The id of the allocation is returned as a string to the user so that
they can specify what entities should run on which allocations
obtained by SmartSim.

The keyword arguments to the ``get_allocation`` method mimic the exact naming
of Slurm arguments that one would normally include in the ``salloc`` command
as arguments. This includes command line arguments that do not have a value
associated with them. In such cases, users can place a value of ``None`` for
that argument. An example of a more complicated allocation is given below:

.. code-block:: python


    from smartsim import Experiment
    experiment = Experiment("Slurm-Experiment", launcher="slurm")
    experiment.get_allocation(nodes=5, constraint="haswell", partition="debug",
                              exclusive=None, time="10:00:00")


Adding Existing Allocations
---------------------------

Existing allocations can also be added to SmartSim. If you already obtained
the allocation to be used for your SmartSim experiment, it can be added to
the Experiment as follows:

.. code-block:: python

    from smartsim import Experiment
    experiment = Experiment("Slurm-Experiment", launcher="slurm")
    experiment.add_allocation(alloc_id)

Where ``alloc_id`` is the id of the allocation in Slurm.


Releasing Allocations
---------------------

SmartSim can release the allocations it has obtained through
``Experiment.release()``. If an id of the allocation is not provided as an
argument, all allocations in the experiment will be released. Below is an
example of obtaining and releasing an allocation.

.. code-block:: python

    from smartsim import Experiment
    experiment = Experiment("Slurm-Experiment", launcher="slurm")
    experiment.get_allocation(nodes=5, constraint="haswell", partition="debug",
                              exclusive=None, time="10:00:00")

    # < experiment code goes here>

    experiment.release()


Local
=====

The local launcher in SmartSim is mainly meant for prototyping and testing
workflows on a laptop. The following Experiment methods will raise exceptions
when called with the local launcher: ``release``, ``get_allocation``, ``add_allocation``
``stop``, ``stop_all``, ``get_status``, ``poll``, ``finished``.

In future releases, the local launcher will support more of the Experiment interface.


Capsules (experimental)
-----------------------

Documentation to come.