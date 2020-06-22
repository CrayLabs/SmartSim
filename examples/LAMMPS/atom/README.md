# Ensemble Generation with SmartSim and LAMMPS

    This example shows the use of SmartSim ensemble
    capabilities with the molecular dynamics (MD)
    code LAMMPS.  In this example, the *thermo* and
    *steps* input variables of the LAMMPS simulation
    will be set to an ensemble of values defined in the
    SmartSim experiment script.

## Ensemble Input File Syntax

   To enable the replacement of input file parameters
   with ensemble values, user-defined tags are placed
   in the simulation input file.  In this example,
   the ";THERMO;" and ";STEPS;" tags have been added to
   the *in.atm* input file.  The variable names bracketed
   by the ";" character correspond to keys in a dictionary object
   that are used by SmartSim to set the ensemble values.
   If the ";" character is already used in the input
   file, another character tag can be specified
   when calling experiment.generate() to create
   the ensemble.

## Ensemble Creation and Execution

   In *run-atm.py*, the dictionary *param_dict* defines
   the ensemble values that will replace the user-defined
   tags described in the previous section.  The SmartSim
   function experiment.generate() will take the
   dictionary of parameters and generate a set of
   folders for each combination of ensemble parameters.
   By default, every combination of ensemble parameters
   will be created by SmartSim, but other generation
   strategies are available.

   To run this example, run the following command
   after installing SmartSim:

   python run-atm.py
