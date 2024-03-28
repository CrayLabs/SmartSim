
***********
Experiments
***********

The Experiment acts as both a factory class for constructing the stages of an
experiment (``Model``, ``Ensemble``, ``Orchestrator``, etc.) as well as an
interface to interact with the entities created by the experiment.

Users can initialize an :ref:`Experiment <experiment_api>` at the beginning of a
Jupyter notebook, interactive python session, or Python file and use the
``Experiment`` to iteratively create, configure and launch computational kernels
on the system through the specified launcher.

.. |SmartSim Architecture| image:: images/ss-arch-overview.png
  :width: 700
  :alt: Alternative text

|SmartSim Architecture|


The interface was designed to be simple, with as little complexity as possible,
and agnostic to the backend launching mechanism (local, Slurm, PBSPro, etc.).

Model
=====

``Model(s)`` are subclasses of ``SmartSimEntity(s)`` and are created through the
Experiment API. Models represent any computational kernel. Models are flexible
enough to support many different applications, however, to be used with our
clients (SmartRedis) the application will have to be written in Python, C, C++,
or Fortran.

Models are given :ref:`RunSettings <rs-api>` objects that specify how a kernel
should be executed with regard to the workload manager (e.g. Slurm) and the
available compute resources on the system.

Each launcher supports specific types of ``RunSettings``.

   - :ref:`SrunSettings <srun_api>` for Slurm
   - :ref:`AprunSettings <aprun_api>` for PBSPro
   - :ref:`MpirunSettings <openmpi_run_api>` for OpenMPI with `mpirun` on PBSPro, LSF, and Slurm
   - :ref:`JsrunSettings <jsrun_api>` for LSF

These settings can be manually specified by the user, or auto-detected by the
SmartSim Experiment through the ``Experiment.create_run_settings`` method.

A simple example of using the Experiment API to create a model and run it
locally:

.. code-block:: Python

  from smartsim import Experiment

  exp = Experiment("simple", launcher="local")

  settings = exp.create_run_settings("echo", exe_args="Hello World")
  model = exp.create_model("hello_world", settings)

  exp.start(model, block=True)
  print(exp.get_status(model))

If the launcher has been specified, or auto-detected through setting
``launcher=auto`` in the Experiment initialization, the ``create_run_settings``
method will automatically create the appropriate ``RunSettings`` object and
return it.

For example with Slurm

.. code-block:: Python

  from smartsim import Experiment

  exp = Experiment("hello_world_exp", launcher="slurm")
  srun = exp.create_run_settings(exe="echo", exe_args="Hello World!")

  # helper methods for configuring run settings are available in
  # each of the implementations of RunSettings
  srun.set_nodes(1)
  srun.set_tasks(32)

  model = exp.create_model("hello_world", srun)
  exp.start(model, block=True, summary=True)

  print(exp.get_status(model))

The above will run ``srun -n 32 -N 1 echo Hello World!``, monitor its
execution, and inform the user when it is completed. This driver script can be
executed in an interactive allocation, or placed into a batch script as follows:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --exclusive
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=32
    #SBATCH --time=00:10:00

    python /path/to/script.py

Ensemble
========

In addition to a single model, SmartSim has the ability to launch an
``Ensemble`` of ``Model`` applications simultaneously.

An ``Ensemble`` can be constructed in three ways:
  1. Parameter expansion (by specifying ``params`` and ``perm_strat`` argument)
  2. Replica creation (by specifying ``replicas`` argument)
  3. Manually (by adding created ``Model`` objects) if launching as a batch job

Ensembles can be given parameters and permutation strategies that define how the
``Ensemble`` will create the underlying model objects.

Three strategies are built in:
  1. ``all_perm``: for generating all permutations of model parameters
  2. ``step``: for creating one set of parameters for each element in `n` arrays
  3. ``random``: for random selection from predefined parameter spaces

Here is an example that uses the ``random`` strategy to intialize four models
with random parameters within a set range. We use the ``params_as_args`` field
to specify that the randomly selected learning rate parameter should be passed
to the created models as a executable argument.

.. code-block:: bash

  import numpy as np
  from smartsim import Experiment

  exp = Experiment("Training-Run", launcher="auto")

  # setup ensemble parameter space
  learning_rate = list(np.linspace(.01, .5))
  train_params = {"LR": learning_rate}

  # define how each member should run
  run = exp.create_run_settings(exe="python",
                                exe_args="./train-model.py")

  ensemble = exp.create_ensemble("Training-Ensemble",
                                params=train_params,
                                params_as_args=["LR"],
                                run_settings=run,
                                perm_strategy="random",
                                n_models=4)
  exp.start(ensemble, summary=True)


A callable function can also be supplied for custom permutation strategies.  The
function should take two arguments: a list of parameter names, and a list of
lists of potential parameter values. The function should return a list of
dictionaries that will be supplied as model parameters. The length of the list
returned will determine how many ``Model`` instances are created.

For example, the following is the built-in strategy ``all_perm``:

.. code-block:: python

    from itertools import product

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
an application(s) in a batch job should be executed with regards to the system
workload manager and available compute resources.

  - :ref:`SbatchSettings <sbatch_api>` for Slurm
  - :ref:`QsubBatchSettings <qsub_api>` for PBSPro
  - :ref:`BsubBatchSettings <bsub_api>` for LSF

If it only passed ``RunSettings``, ``Ensemble``, objects will require either
a ``replicas`` argument or a ``params`` argument to expand parameters
into ``Model`` instances. At launch, the ``Ensemble`` will look for
interactive allocations to launch models in.

If it passed ``BatchSettings`` without other arguments, an empty ``Ensemble``
will be created that ``Model`` objects can be added to manually. All ``Model``
objects added to the ``Ensemble`` will be launched in a single batch.

If it passed ``BatchSettings`` and ``RunSettings``, the ``BatchSettings`` will
determine the allocation settings for the entire batch, and the ``RunSettings``
will determine how each individual ``Model`` instance is executed within
that batch.

This is the same example as above, but tailored towards a running as a batch job
on a slurm system:

.. code-block:: bash

  import numpy as np
  from smartsim import Experiment

  exp = Experiment("Training-Run", launcher="slurm")

  # setup ensemble parameter space
  learning_rate = list(np.linspace(.01, .5))
  train_params = {"LR": learning_rate}

  # define resources for all ensemble members
  sbatch = exp.create_batch_settings(nodes=4,
                                    time="01:00:00",
                                    account="12345-Cray",
                                    queue="gpu")

  # define how each member should run
  srun = exp.create_run_settings(exe="python",
                                exe_args="./train-model.py")
  srun.set_nodes(1)
  srun.set_tasks(24)

  ensemble = exp.create_ensemble("Training-Ensemble",
                                params=train_params,
                                params_as_args=["LR"],
                                batch_settings=sbatch,
                                run_settings=srun,
                                perm_strategy="random",
                                n_models=4)
  exp.start(ensemble, summary=True)


This will generate and execute a batch script that looks something like
the following:

.. code-block:: bash

  # GENERATED

  #!/bin/bash

  #SBATCH --output=/lus/smartsim/Training-Ensemble.out
  #SBATCH --error=/lus/smartsim/Training-Ensemble.err
  #SBATCH --job-name=Training-Ensemble-CHTN0UI2DORX
  #SBATCH --nodes=4
  #SBATCH --time=01:00:00
  #SBATCH --partition=gpu
  #SBATCH --account=12345-Cray

  cd /scratch/smartsim/Training-Run ; /usr/bin/srun --output /scratch/smartsim/Training-Run/Training-Ensemble_0.out --error /scratch/smartsim/Training-Ensemble_0.err --job-name Training-Ensemble_0-CHTN0UI2E5DX --nodes=1 --ntasks=24 /scratch/pyenvs/smartsim/bin/python ./train-model.py --LR=0.17 &

  cd /scratch/smartsim/Training-Run ; /usr/bin/srun --output /scratch/smartsim/Training-Run/Training-Ensemble_1.out --error /scratch/smartsim/Training-Ensemble_1.err --job-name Training-Ensemble_1-CHTN0UI2JQR5 --nodes=1 --ntasks=24 /scratch/pyenvs/smartsim/bin/python ./train-model.py --LR=0.32 &

  cd /scratch/smartsim/Training-Run ; /usr/bin/srun --output /scratch/smartsim/Training-Run/Training-Ensemble_2.out --error /scratch/smartsim/Training-Ensemble_2.err --job-name Training-Ensemble_2-CHTN0UI2P2AR --nodes=1 --ntasks=24 /scratch/pyenvs/smartsim/bin/python ./train-model.py --LR=0.060000000000000005 &

  cd /scratch/smartsim/Training-Run ; /usr/bin/srun --output /scratch/smartsim/Training-Run/Training-Ensemble_3.out --error /scratch/smartsim/Training-Ensemble_3.err --job-name Training-Ensemble_3-CHTN0UI2TRE7 --nodes=1 --ntasks=24 /scratch/pyenvs/smartsim/bin/python ./train-model.py --LR=0.35000000000000003 &

  wait

Prefixing Keys in the Orchestrator
----------------------------------

If each of multiple ensemble members attempt to use the same code to access their respective models
in the Orchestrator, the keys by which they do this will overlap and they can end up accessing each
others' data inadvertently. To prevent this situation, the SmartSim Entity object supports key
prefixing, which automatically prepends the name of the model to the keys by which it is accessed.
With this enabled, key overlapping is no longer an issue and ensemble members can use the same code.

Under the hood, calling ensemble.enable_key_prefixing() causes the SSKEYOUT environment variable to
be set, which in turn causes all keys generated by an ensemble member to be prefixed with its model
name. Similarly, if the model for the ensemble member has incoming entities (such as those set via
model.register_incoming_entity() or ensemble.register_incoming_entity()), the SSKEYIN environment
variable will be set and the keys associated with those inputs will be automatically prefixed. Note
that entities must register themselves as this is not done by default.

Finally, please note that while prefixing is enabled by default for tensors, datasets, and aggregated
lists of datasets, a SmartRedis client must manually call Client.use_model_ensemble_prefix() to
ensure that prefixes are used with models and scripts.

We modify the example above to enable key prefixing as follows:

.. code-block:: bash

  import numpy as np
  from smartsim import Experiment

  exp = Experiment("Training-Run", launcher="slurm")

  # setup ensemble parameter space
  learning_rate = list(np.linspace(.01, .5))
  train_params = {"LR": learning_rate}

  # define resources for all ensemble members
  sbatch = exp.create_batch_settings(nodes=4,
                                    time="01:00:00",
                                    account="12345-Cray",
                                    queue="gpu")

  # define how each member should run
  srun = exp.create_run_settings(exe="python",
                                exe_args="./train-model.py")
  srun.set_nodes(1)
  srun.set_tasks(24)

  ensemble = exp.create_ensemble("Training-Ensemble",
                                params=train_params,
                                params_as_args=["LR"],
                                batch_settings=sbatch,
                                run_settings=srun,
                                perm_strategy="random",
                                n_models=4)

  # Enable key prefixing -- note that this should be done
  # before starting the experiment
  ensemble.enable_key_prefixing()

  exp.start(ensemble, summary=True)


Further Information
-------------------

For more informtion about Ensembles, please refer to the :ref:`Ensemble API documentation <ensemble_api>`.