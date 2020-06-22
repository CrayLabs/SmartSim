# LAMMPS Examples

    In this directory are two examples of the molecular dynamics (MD)
    code LAMMPS used in a SmartSim experiment.  These examples are
    derived from the *atom* and *melt* examples that are included
    with LAMMPS.

## Atom Example

   In the *atom* directory is an example of using LAMMPS with the
   ensemble capabilities of SmartSim.  In this example,
   the *thermo* and *steps* input variables of the LAMMPS simulation
   will be set to an ensemble of values defined in the SmartSim
   experiment script.

## Melt Example

   In the *melt* directory is an example of using LAMMPS with
   the SmartSim clients to send data from a running LAMMPS
   simulation to a data processing script via the SmartSim
   in-memory database.  Changes to LAMMPS that are required
   to run this example are described in the example README.
