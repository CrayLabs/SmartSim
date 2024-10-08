
***********
Experiments
***********

The Experiment acts as both a factory function as well as an interface to interact
with the entities created by the experiment.

Users can initialize an :ref:`Experiment <experiment_api>` at the beginning of a Jupyter notebook,
interactive python session, or Python file and use the ``Experiment`` to
iteratively create, configure and launch computational kernels on the
system through the specified launcher.

.. |SmartSim Architecture| image:: images/SmartSim_Architecture.png
  :width: 700
  :alt: Alternative text

|SmartSim Architecture|


The interface was designed to be simple with as little complexity
as possible, and agnostic to the backend launching mechanism (local,
Slurm, PBSPro, etc).


Entities
========

The instances created by an ``Experiment`` fall into two classes:
  1. ``SmartSimEntity``
  2. ``EntityList``

``Model`` instances are ``SmartSimEntity`` objects. ``Ensemble`` instances
are ``EntityList`` objects. ``EntityList`` instances are containers of
``SmartSimEntity`` objects.

Model
=====

``Model(s)`` are created through the Experiment API. Models represent
any computational kernel. Models are flexible enough to support many
different applications, however, to be used with our clients (SmartRedis)
the application will have to be written in Python, C, C++, or Fortran.

Models are given :ref:`RunSettings <rs-api>` objects that specify how a kernel should
be executed with regard to the workload manager (e.g. Slurm) and the available
compute resources on the system.

Each launcher supports specific types of ``RunSettings``.

   - :ref:`SrunSettings <srun_api>` for Slurm
   - :ref:`AprunSettings <aprun_api>` for PBSPro and Cobalt
   - :ref:`MpirunSettings <openmpi_api>` for OpenMPI with `mpirun` on PBSPro, Cobalt, LSF, and Slurm
   - :ref:`JsrunSettings <jsrun_api>` for LSF

When on systems that support these launch binaries, ``Model`` objects can
be created to run applications in allocations obtained by the user, or in the
case of Slurm based systems, SmartSim can obtain allocations before launching
``Model`` instances.

Ensemble
========

In addition to a single model, SmartSim has the ability to launch an
``Ensemble`` of ``Model`` applications simultaneously.

An ``Ensemble`` can be constructed in three ways:
  1. Parameter expansion (by specifying ``params`` and ``perm_strat`` argument)
  2. Replica creation (by specifying ``replicas`` argument)
  3. Manually (by adding created ``Model`` objects) if launching as a batch job

Ensembles can be given parameters and permutation strategies that
define how the ``Ensemble`` will create the underlying model objects.

Three strategies are built in:
  1. ``all_perm`` for generating all permutations of model parameters
  2. ``step`` for creating one set of parameters for each element in `n` arrays
  3. ``random`` for random selection from predefined parameter spaces.

A callable function can also be supplied for custom permutation strategies.
The function should take in two lists: parameter names and parameter values.
The function should return a list of dictionaries that will be supplied as
model parameters. The length of the list returned will determine how many
``Model`` instances are created.

For example, the following the the built-in strategy ``all_perm``.

.. code-block:: python

    def create_all_permutations(param_names, param_values):
        perms = list(product(*param_values))
        all_permutations = []
        for p in perms:
            temp_model = dict(zip(param_names, p))
            all_permutations.append(temp_model)
        return all_permutations


After ``Ensemble`` initialization, ``Ensemble`` instances can be
passed as arguments to ``Experiment.generate()`` to write assigned
parameter values into attached and tagged configuration files.

Launching Ensembles
-------------------

Ensembles can be launched in previously obtained interactive allocations
and as a batch. Similar to ``RunSettings``, ``BatchSettings`` specify how
a application(s) in a batch job should be executed with regards to the system
workload manager and available compute resources.

  - :ref:`SbatchSettings <sbatch_api>` for Slurm
  - :ref:`QsubBatchSettings <qsub_api>` for PBSPro
  - :ref:`CobaltBatchSettings <cqsub_api>` for Cobalt
  - :ref:`BsubBatchSettings <bsub_api>` for LSF

If only passed ``RunSettings``, ``Ensemble`` objects will require either
a ``replicas`` argument or a ``params`` argument to expand parameters
into ``Model`` instances. At launch, the ``Ensemble`` will look for
interactive allocations to launch models in.

If passed ``BatchSettings`` without other arguments, an empty ``Ensemble``
will be created that ``Model`` objects can be added to manually. All ``Model``
objects added to the ``Ensemble`` will be launched in a single batch.

If passed ``BatchSettings`` and ``RunSettings``, the ``BatchSettings`` will
determine the allocation settings for the entire batch, and the ``RunSettings``
will determine how each individual ``Model`` instance is executed within
that batch.

