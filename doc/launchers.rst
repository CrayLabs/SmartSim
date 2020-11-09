
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


Local
=====

The Local launcher uses the ``Subprocess`` library in Python to
execute multiple entities in parallel. Each process is tracked
and the output of each entity is written to file.

As with all launchers in SmartSim, the local launcher supports
asynchronous execution meaning once entities have been launched
the main thread of execution is not blocked. Daemon threads
that manage currently running jobs will be created when active
jobs are present within SmartSim.


Slurm
=====

Getting Allocations
-------------------

SmartSim provides an interface to obtain allocations programmatically
so that each script will contain the exact configuration upon which
it was launched including the allocation information.

The slurm allocation interface is importable through the main module
and allocations can be obtained through a call to ``slurm.get_slurm_allocation``
as follows:

.. code-block:: python

    from smartsim import slurm
    alloc = slurm.get_slurm_allocation(nodes=1)

The id of the allocation is returned as a string to the user so that
they can specify what entities should run on which allocations
obtained by SmartSim.

Additional arguments that would have been passed to the ``salloc``
command can be passed through the ``add_opts`` argument in a dictionary.

.. code-block:: python

    from smartsim import slurm
    salloc_options = {
        "C": "haswell",
        "partition": "debug",
        "time": "10:00:00",
        "exclusive": None
    }
    alloc_id = slurm.get_slurm_allocation(nodes=5, add_opts=salloc_options)

The above code would generate a salloc command like:

.. code-block:: bash

    salloc -N 5 -C haswell --parition debug --time 10:00:00 --exclusive



Releasing Allocations
---------------------

The ``smartsim.slurm`` interface also supports releasing allocations
obtained in an experiment.

The example below releases a the allocation in the example above.

.. code-block:: python

    from smartsim import slurm
    salloc_options = {
        "C": "haswell",
        "partition": "debug",
        "time": "10:00:00",
        "exclusive": None
    }
    alloc_id = slurm.get_slurm_allocation(nodes=5, add_opts=salloc_options)

    # <experiment code goes here>

    slurm.release_slurm_allocation(alloc_id)

