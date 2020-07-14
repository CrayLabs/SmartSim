********************
Generating Ensembles
********************


This example shows the use of SmartSim ensemble
capabilities with the molecular dynamics (MD)
code LAMMPS.  In this example, the ``THERMO`` and
``STEPS`` input variables of the LAMMPS simulation
will be set to an ensemble of values defined in the
SmartSim experiment script.


Experiment Setup
================

In *run-atm.py*, the dictionary *param_dict* defines
the ensemble values that will replace the user-defined
tags described in the previous section.  The SmartSim
function experiment.generate() will take the
dictionary of parameters and generate a set of
folders for each combination of ensemble parameters.
By default, every combination of ensemble parameters
will be created by SmartSim, but other generation
strategies are available.

Tagging Input Files
-------------------

To enable the replacement of input file parameters
with ensemble values, user-defined tags are placed
in the simulation input file.  In this example,
the ``;THERMO;`` and ``;STEPS;`` tags have been added to
the ``in.atm`` input file.  The variable names bracketed
by the ``;`` character correspond to keys in a dictionary object
that are used by SmartSim to set the ensemble values.
If the ``;`` character is already used in the input
file, another character tag can be specified
when calling ``experiment.generate()`` to create
the ensemble.


To generate new instances of a simulation model, find and tag the simulation sepcific
configuration file. The following experiment will use a particle simulation model
named ``LAMMPS``. The model configuration file, ``in.atm`` specifies LAMMPS to run
a Axilrod-Teller-Muto (ATM) potential calculation.

.. code-block:: text

   # Axilrod-Teller-Muto potential example

   variable        x index 1
   variable        y index 1
   variable        z index 1

   variable        xx equal 10*$x
   variable        yy equal 10*$y
   variable        zz equal 10*$z

   units           lj
   atom_style      atomic

   lattice         fcc 0.65
   region          box block 0 ${xx} 0 ${yy} 0 ${zz}
   create_box      1 box
   create_atoms    1 box

   pair_style      hybrid/overlay lj/cut 4.5 atm 4.5 2.5
   pair_coeff      * * lj/cut 1.0 1.0
   pair_coeff      * * atm * 0.072

   mass            * 1.0
   velocity        all create 1.033 12345678 loop geom

   fix             1 all nvt temp 1.033 1.033 0.05

   timestep        0.002
   thermo          5

   run             25

The last variable listed, ``run``, tells the simulation to run for a number of steps
before exiting. For this example, we will tag, and modify the number of steps in order
to generate four simulation models that run for 4 different lengths of time.

Tagging the configuration file lets the ``Generator`` know which of and where in
the model files to edit given a file or folder of model files to modify.

To tag the ``in.atm`` configuration file listed above, place semicolons on either
side of the values that you wish to change, and put in a good placeholder name
so that you can remember which values are being edited.

.. code-block:: text

   # Axilrod-Teller-Muto potential example

   variable        x index 1
   variable        y index 1
   variable        z index 1

   variable        xx equal 10*$x
   variable        yy equal 10*$y
   variable        zz equal 10*$z

   units           lj
   atom_style      atomic

   lattice         fcc 0.65
   region          box block 0 ${xx} 0 ${yy} 0 ${zz}
   create_box      1 box
   create_atoms    1 box

   pair_style      hybrid/overlay lj/cut 4.5 atm 4.5 2.5
   pair_coeff      * * lj/cut 1.0 1.0
   pair_coeff      * * atm * 0.072

   mass            * 1.0
   velocity        all create 1.033 12345678 loop geom

   fix             1 all nvt temp 1.033 1.033 0.05

   timestep        0.002
   thermo          5

   run             ;STEPS;

The tag can also be set through a call to ``Generator.set_tag()``. The tag can be
anything that wont be already represented within the configuration file itself.
For instance, in the example above, we wouldnt want to use dollar signs or curly
braces for the tag.


Creating the Ensembles
----------------------

show configuration
show python code


Running the Experiment
----------------------

To run this example, run the following command
after installing SmartSim:

.. code-block:: bash

   python run-atm.py


Experiment Script
=================

The full script for the previously described experiment

.. code-block:: python

   from smartsim import Experiment

   experiment = Experiment("lammps_atm")

   # Create ensembles
   run_settings = {
      "executable": "lmp_mpi",
      "run_command": "mpirun",
      "run_args": "-np 4",
      "exe_args": "-in in.atm"
   }

   param_dict = {"STEPS": [5, 10], "THERMO": 5}
   experiment.create_ensemble("atm", params=param_dict, run_settings=run_settings)

   base_config = "./in.atm"
   experiment.generate(model_files=base_config)
   experiment.start(launcher="local")
