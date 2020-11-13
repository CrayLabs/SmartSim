
***************************************
Tagging Input Files to Create Ensembles
***************************************

In this experiment we will demonstate how to use SmartSim
to generate and run ensembles of LAMMPS models. The model
is a simulation of crack propagation in a 2-dimensional solid.

We will show how to tag the input file, ``in.crack``, so that
multiple simulations with different configurations can be
run in the same python script.

.. note::

   There are two versions of this script, one for the
   local launcher and one for Slurm. The following
   will show the example for the local launcher.

   For the local example, LAMMPS must be installed as
   the ``lmp_mpi`` executable easily installed on OSX
   through brew ``brew install lammps``. If you are not
   working on OSX, then download and compile LAMMPS
   from source.

   To run the slurm example, you will need to have the
   LAMMPS executable, ``lmp`` compiled with MPI available
   on system that uses Slurm as the workload manager.

Tagging Input Files
===================

The first step creating a model parameter space in SmartSim is to
take the input file(s) that you wish to edit the configuration
of and "tag" the specific configuration values with a user-settable
character. The default tag is a semi-colon (e.g. ``;``).

The input file below is the ``in.crack`` file before we have
tagged any of the configuration values. This file was taken
directly from the LAMMPS examples directory.

.. code-block:: text

   # 2d LJ crack simulation

   dimension	2
   boundary	s s p

   atom_style	atomic
   neighbor	0.3 bin
   neigh_modify	delay 5

   # create geometry

   lattice		hex 0.93
   region		box block 0 100 0 40 -0.25 0.25
   create_box	5 box
   create_atoms	1 box

   mass		1 1.0
   mass		2 1.0
   mass		3 1.0
   mass		4 1.0
   mass		5 1.0

   # LJ potentials

   pair_style	lj/cut 2.5
   pair_coeff	* * 1.0 1.0 2.5

   # define groups

   region	        1 block INF INF INF 1.25 INF INF
   group		lower region 1
   region		2 block INF INF 38.75 INF INF INF
   group		upper region 2
   group		boundary union lower upper
   group		mobile subtract all boundary

   region		leftupper block INF 20 20 INF INF INF
   region		leftlower block INF 20 INF 20 INF INF
   group		leftupper region leftupper
   group		leftlower region leftlower

   set		group leftupper type 2
   set		group leftlower type 3
   set		group lower type 4
   set		group upper type 5

   # initial velocities

   compute	  	new mobile temp
   velocity	mobile create 0.01 887723 temp new
   velocity	upper set 0.0 0.3 0.0
   velocity	mobile ramp vy 0.0 0.3 y 1.25 38.75 sum yes

   # fixes

   fix		1 all nve
   fix		2 boundary setforce NULL 0.0 0.0

   # run

   timestep	0.003
   thermo		200
   thermo_modify	temp new

   neigh_modify	exclude type 2 3

   #dump		1 all atom 500 dump.crack

   #dump		2 all image 250 image.*.jpg type type &
   #		zoom 1.6 adiam 1.5
   #dump_modify	2 pad 4

   #dump		3 all movie 250 movie.mpg type type &
   #		zoom 1.6 adiam 1.5
   #dump_modify	3 pad 4

   run		5000


For our ensemble we wish to run the configuration with varying
tempurature for each model. Likewise, we want each tempurature
to run for two different timescales. In order to control this
from SmartSim, we will tag the ``thermo`` value of ``200`` and
the ``run`` value of ``5000``. We recommend putting placeholders
in place of the values for readability sake. We do this below:

.. code-block:: text

   # 2d LJ crack simulation

   dimension	2
   boundary	s s p

   atom_style	atomic
   neighbor	0.3 bin
   neigh_modify	delay 5

   # create geometry

   lattice		hex 0.93
   region		box block 0 100 0 40 -0.25 0.25
   create_box	5 box
   create_atoms	1 box

   mass		1 1.0
   mass		2 1.0
   mass		3 1.0
   mass		4 1.0
   mass		5 1.0

   # LJ potentials

   pair_style	lj/cut 2.5
   pair_coeff	* * 1.0 1.0 2.5

   # define groups

   region	        1 block INF INF INF 1.25 INF INF
   group		lower region 1
   region		2 block INF INF 38.75 INF INF INF
   group		upper region 2
   group		boundary union lower upper
   group		mobile subtract all boundary

   region		leftupper block INF 20 20 INF INF INF
   region		leftlower block INF 20 INF 20 INF INF
   group		leftupper region leftupper
   group		leftlower region leftlower

   set		group leftupper type 2
   set		group leftlower type 3
   set		group lower type 4
   set		group upper type 5

   # initial velocities

   compute	  	new mobile temp
   velocity	mobile create 0.01 887723 temp new
   velocity	upper set 0.0 0.3 0.0
   velocity	mobile ramp vy 0.0 0.3 y 1.25 38.75 sum yes

   # fixes

   fix		1 all nve
   fix		2 boundary setforce NULL 0.0 0.0

   # run

   timestep	0.003
   thermo		;THERMO;
   thermo_modify	temp new

   neigh_modify	exclude type 2 3

   #dump		1 all atom 500 dump.crack

   #dump		2 all image 250 image.*.jpg type type &
   #		zoom 1.6 adiam 1.5
   #dump_modify	2 pad 4

   #dump		3 all movie 250 movie.mpg type type &
   #		zoom 1.6 adiam 1.5
   #dump_modify	3 pad 4

   run		;STEPS;

Our input files are now ready to be included in a SmartSim
experiment.

Setting up the Experiment
-------------------------

Now that we have tagged our configuration file for the LAMMPS
model we can start to write the script that will run our
ensemble.

First, we need to initialize an ``Experiment``.

.. code-block:: python

   from smartsim import Experiment

   # Create the Experiment object
   experiment = Experiment("lammps_crack", launcher="local")

Each of our models will run with 2 MPI tasks per simulation.
We will specify exactly how we want each model in the ensemble to
run by creating a dictionary ``run_settings``.

When using a launch command like ``mpirun``, you can place
the model executable in the ``exe_args``.

.. code-block:: python

   # Set the run settings for each member of the
   # ensemble.
   run_settings = {
      "executable": "mpirun",
      "exe_args": "-np 2 lmp_mpi -i in.crack"
   }


Generating an Ensemble
----------------------

In SmartSim, we refer to Ensemble creation as ``generation``. This is because
SmartSim copies and modifies the input configurations of the simulation and "generates"
a file structure in which each of the models of the ensemble will be executed.

The files for generation can be specified by the user once an entity has been created
through the ``SmartSimEntity.attach_generator_files()`` method.

Since we only have one input file for this experiment, the only file we will
attach to our ensemble is ``in.crack``. We specify this file under the ``to_configure``
argument as we want SmartSim to read and edit the file based on which model
in the ensemble is being executed.

Lastly, before we generate the ensemble, we need to specify the input parameter
values we require each model within the ensemble to run with. The entire ensemble
generation process is provided below.

.. code-block:: python

   # Set the run settings for each member of the
   # ensemble.
   run_settings = {
      "executable": "mpirun",
      "exe_args": "-np 2 lmp_mpi -i in.crack"
   }

   # Set the parameter space for the ensemble
   # The default strategy is to generate all permutations
   # so all permutations of STEPS and THERMO will
   # be generated as a single model
   model_params = {
      "STEPS": [20000, 40000],
      "THERMO": [150, 250]
   }
   # Create ensemble with the model params and
   # run settings defined
   ensemble = experiment.create_ensemble("crack",
                                       params=model_params,
                                       run_settings=run_settings)

   # attach files to be generated at runtime
   # in each model directory where the executable
   # will be invoked
   ensemble.attach_generator_files(to_configure="./in.crack")
   experiment.generate(ensemble, overwrite=True)


As the above code snippet states, the default generation strategy of SmartSim
is to generate all permutations of the input parameter arrays given to the
``params`` argument of the ``Experiment.create_ensemble()`` method. Given that
there are 2 values for ``STEPS`` and 2 values of ``THERMO``, a total of 4
models will be generated.

SmartSim has multiple generation strategies, and supports custom generation
strategies as well. For more information on this, see the
`experiment creation tutorial <../../../doc/using-smartsim/create.html>`_

Starting and Monitoring the Experiment
--------------------------------------

Now that our ensemble has been generated and configured, we will
run the experiment and monitor the progress. We also print out a
summary of the experiment prior to launch to review our launch
parameters before execution.

``Experiment.summary()`` returns a Pandas DataFrame that details
the entities that have run and completed (or failed) in the current
experiment.

.. code-block:: python

   # Start the experiment
   experiment.start(ensemble, summary=True)

   # show what happened in the experiment
   print(experiment.summary())

After running the local launcher example, the generated file structure
should look as follows with all input, output, and data from each model
in the ensemble.

.. code-block:: text

   lammps_crack
   └── crack
      ├── crack_0
      │   ├── crack_0.err
      │   ├── crack_0.out
      │   ├── in.crack
      │   └── log.lammps
      ├── crack_1
      │   ├── crack_1.err
      │   ├── crack_1.out
      │   ├── in.crack
      │   └── log.lammps
      ├── crack_2
      │   ├── crack_2.err
      │   ├── crack_2.out
      │   ├── in.crack
      │   └── log.lammps
      └── crack_3
         ├── crack_3.err
         ├── crack_3.out
         ├── in.crack
         └── log.lammps

   5 directories, 16 files



Experiment Script
=================

The full script for the previously described experiment

.. code-block:: python

   from smartsim import Experiment

   # Create the Experiment object
   experiment = Experiment("lammps_crack", launcher="local")

   # Set the run settings for each member of the
   # ensemble.
   run_settings = {
      "executable": "mpirun",
      "exe_args": "-np 2 lmp_mpi -i in.crack"
   }

   # Set the parameter space for the ensemble
   # The default strategy is to generate all permutations
   # so all permutations of STEPS and THERMO will
   # be generated as a single model
   model_params = {
      "STEPS": [20000, 40000],
      "THERMO": [150, 250]
   }
   # Create ensemble with the model params and
   # run settings defined
   ensemble = experiment.create_ensemble("crack",
                                       params=model_params,
                                       run_settings=run_settings)

   # attach files to be generated at runtime
   # in each model directory where the executable
   # will be invoked
   ensemble.attach_generator_files(to_configure="./in.crack")
   experiment.generate(ensemble, overwrite=True)

   # Start the experiment
   experiment.start(ensemble, summary=True)

   # show what happened in the experiment
   print(experiment.summary())
