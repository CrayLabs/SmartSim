***********
Run Settings
***********

=========
 Overview
=========
SmartSim run settings describe how a Model or Ensemble should be executed provided the system
and available computational resources. A ``RunSettings`` instance is created through the
``Experiment.create_run_settings()`` function which accepts a `run_command` argument.
If `run_command` is set to `auto`, SmartSim will attempt to match a run command on the
system with a RunSettings class in SmartSim. If found, the class corresponding to
that run_command will be created and returned:

When initializing an ``Experiment`` object and specifying `launcher="local"`, auto detection will be turned off.
If the `run_command` is passed a recognized run command (ex. 'Slurm') the ``RunSettings``
instance will be a child class such as ``SrunSettings``.

If not supported by smartsim, the base RunSettings class will be
created and returned with the specified run_command and run_args will be evaluated literally.

=====
Local
=====

Local
-----
The local `launcher` supports the base :ref:`RunSettings API <rs-api>`
which can be used to run executables as well as run executables
with arbitrary launch binaries like `mpiexec`.

The local launcher is the default launcher for all ``Experiment``
instances.

Ensembles are always executed in parallel but launched sequentially.

Initialize
----------
Case 1 : Setting when creating the Experiment
    When initializing an ``Experiment`` object, by specifying the `launcher` argument

    .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="local")  # local launcher

    .. code-block:: python

      settings = exp.create_run_settings()  # local launcher

    You can find the application and experiment source code in subsections below.

Case 2 : Specifying with `run_command`

    .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="local")  # local launcher

    .. code-block:: python

      settings = exp.create_run_settings(run_command="mpirun")  # local launcher


===
HPC
===
SmartSim offers support to run your experiment entity instances
with the following ``RunSettings`` child classes per WLM below:

1. Slurm WLM system
   - ``SrunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
2. PBSPro WLM system
   - ``AprunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
3. Cobalt WLM system
   - ``AprunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``
4. LSF WLM system
   - ``JsrunSettings``
   - ``MpirunSettings``
   - ``MpiexecSettings``

Initialize
----------

Case 1 : To use the an HPC launcher such as `Slurm`, specify at Experiment initialization:

    More specifically, specify through the `launcher` argument:

    .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="slurm")  # slurm launcher

    ``SrunSettings`` will be returned

    .. code-block:: python

      settings = exp.create_run_settings()

Case 2 : To use the `run_command` variable, specify at RunSettings initializations

    .. code-block:: python

      exp = Experiment("name-of-experiment", launcher="slurm")  # local launcher

    .. code-block:: python

      settings = exp.create_run_settings(run_command="mpiexec")  # local launcher

    The above will return a ``MpiexecSettings`` object.