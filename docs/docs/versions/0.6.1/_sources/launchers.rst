
*********
Launchers
*********

SmartSim interfaces with a number of backends called `launchers` that
are responsible for constructing jobs based on run parameters and
launching them onto a system.

The `launchers` allow SmartSim users to interact with their system
programmatically through a python interface.
Because of this, SmartSim users do not have to leave the Jupyter Notebook,
Python REPL, or Python script to launch, query, and interact with their jobs.

SmartSim currently supports 5 `launchers`:
  1. ``local``: for single-node, workstation, or laptop
  2. ``slurm``: for systems using the Slurm scheduler
  3. ``pbs``: for systems using the PBSpro scheduler
  4. ``lsf``: for systems using the LSF scheduler
  5. ``auto``: have SmartSim auto-detect the launcher to use.

To specify a specific launcher, one argument needs to be provided
to the ``Experiment`` initialization.

.. code-block:: python

    from smartsim import Experiment

    exp = Experiment("name-of-experiment", launcher="local")  # local launcher
    exp = Experiment("name-of-experiment", launcher="slurm")  # Slurm launcher
    exp = Experiment("name-of-experiment", launcher="pbs")    # PBSpro launcher
    exp = Experiment("name-of-experiment", launcher="lsf")    # LSF launcher
    exp = Experiment("name-of-experiment", launcher="auto")   # auto-detect launcher

-------------------------------------------------------------------------

Local
=====


The local launcher can be used on laptops, workstations and single
nodes of supercomputer and cluster systems. Through
launching locally, users can prototype workflows and quickly scale
them to larger systems with minimal changes.

As with all launchers in SmartSim, the local launcher supports
asynchronous execution meaning once entities have been launched
the main thread of execution is not blocked. Daemon threads
that manage currently running jobs will be created when active
jobs are present within SmartSim.

.. _psutil: https://github.com/giampaolo/psutil

The local launcher uses the `psutil`_ library to execute and monitor
user-created jobs.


Running Locally
---------------

The local launcher supports the base :ref:`RunSettings API <rs-api>`
which can be used to run executables as well as run executables
with arbitrary launch binaries like `mpiexec`.

The local launcher is the default launcher for all ``Experiment``
instances.

The local launcher does not support batch launching. Ensembles
are always executed in parallel but launched sequentially.

----------------------------------------------------------------------

Slurm
=====

The Slurm launcher works directly with the Slurm scheduler to launch, query,
monitor and stop applications. During the course of an ``Experiment``,
launched entities can be queried for status, completion, and errors.

The amount of communication between SmartSim and Slurm can be tuned
for specific guidelines of different sites by setting the
value for ``jm_interval`` in the SmartSim configuration file.

To use the Slurm launcher, specify at ``Experiment`` initialization:

.. code-block:: python

    from smartsim import Experiment

    exp = Experiment("NAMD-worklfow", launcher="slurm")


Running on Slurm
----------------

The Slurm launcher supports three types of ``RunSettings``:
  1. :ref:`SrunSettings <srun_api>`
  2. :ref:`MpirunSettings <openmpi_run_api>`
  3. :ref:`MpiexecSettings <openmpi_exec_api>`

As well as batch settings for ``sbatch`` through:
  1. :ref:`SbatchSettings <sbatch_api>`


Both supported ``RunSettings`` types above can be added
to a ``SbatchSettings`` batch workload through ``Ensemble``
creation.


Getting Allocations
-------------------

Slurm supports a number of user facing features that other schedulers
do not. For this reason, an extra module :ref:`smartsim.slurm <slurm_module_api>` can be
used to obtain allocations to launch on and release them after
``Experiment`` completion.

.. code-block:: python

    from smartsim.wlm import slurm
    alloc = slurm.get_allocation(nodes=1)

The ID of the allocation is returned as a string to the user so that
they can specify what entities should run on which allocations
obtained by SmartSim.

Additional arguments that would have been passed to the ``salloc``
command can be passed through the ``options`` argument in a dictionary.

Anything passed to the options will be processed as a Slurm
argument and appended to the salloc command with the appropriate
prefix (e.g. `-` or `--`).

For arguments without a value, pass None as the value:
    - `exclusive=None`

.. code-block:: python

    from smartsim.wlm import slurm
    salloc_options = {
        "C": "haswell",
        "partition": "debug",
        "exclusive": None
    }
    alloc_id = slurm.get_slurm_allocation(nodes=128,
                                          time="10:00:00",
                                          options=salloc_options)

The above code would generate a ``salloc`` command like:

.. code-block:: bash

    salloc -N 5 -C haswell --partition debug --time 10:00:00 --exclusive



Releasing Allocations
---------------------

The :ref:`smartsim.slurm <slurm_module_api>` interface
also supports releasing allocations obtained in an experiment.

The example below releases the allocation in the example above.

.. code-block:: python

    from smartsim.wlm import slurm
    salloc_options = {
        "C": "haswell",
        "partition": "debug",
        "exclusive": None
    }
    alloc_id = slurm.get_slurm_allocation(nodes=128,
                                        time="10:00:00",
                                        options=salloc_options)

    # <experiment code goes here>

    slurm.release_slurm_allocation(alloc_id)

-------------------------------------------------------------------

PBSPro
======

Like the Slurm launcher, the PBSPro launcher works directly with the PBSPro
scheduler to launch, query, monitor and stop applications.

The amount of communication between SmartSim and PBSPro can be tuned
for specific guidelines of different sites by setting the
value for ``jm_interval`` in the SmartSim configuration file.

To use the PBSpro launcher, specify at ``Experiment`` initialization:

.. code-block:: python

    from smartsim import Experiment

    exp = Experiment("LAMMPS-melt", launcher="pbs")



Running on PBSpro
-----------------

The PBSpro launcher supports three types of ``RunSettings``:
  1. :ref:`AprunSettings <aprun_api>`
  2. :ref:`MpirunSettings <openmpi_run_api>`
  3. :ref:`MpiexecSettings <openmpi_exec_api>`

As well as batch settings for ``qsub`` through:
  1. :ref:`QsubBatchSettings <qsub_api>`

Both supported ``RunSettings`` types above can be added
to a ``QsubBatchSettings`` batch workload through ``Ensemble``
creation.

---------------------------------------------------------------------

LSF
===

The LSF Launcher works like the PBSPro launcher and
is compatible with LSF and OpenMPI workloads.

To use the LSF launcher, specify at ``Experiment`` initialization:

.. code-block:: python

    from smartsim import Experiment

    exp = Experiment("MOM6-double-gyre", launcher="lsf")


Running on LSF
--------------

The LSF launcher supports three types of ``RunSettings``:
  1. :ref:`JsrunSettings <jsrun_api>`
  2. :ref:`MpirunSettings <openmpi_run_api>`
  3. :ref:`MpiexecSettings <openmpi_exec_api>`

As well as batch settings for ``bsub`` through:
  1. :ref:`BsubBatchSettings <bsub_api>`

Both supported ``RunSettings`` types above can be added
to a ``BsubBatchSettings`` batch workload through ``Ensemble``
creation.
