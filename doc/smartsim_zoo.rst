#####################
Contributing Examples
#####################

.. _smartsim_zoo:

=========================
What Is the SmartSim Zoo?
=========================

Given that SmartSim is a community developed and maintained project, we have
introduced a `SmartSim Example Zoo <https://github.com/CrayLabs/SmartSim-Zoo>`__
that contains CrayLabs and user contributed examples of using SmartSim for
various simulation and machine learning applications.

--------------------------------------
The Two Categories of the SmartSim Zoo
--------------------------------------

 1. SmartSim Deployments (running SmartSim on various HPC Systems)

  * The source code for the repository serves the purpose of showing diverse

  * examples of how to get SmartSim running on different HPC Systems.
    If you are looking for working examples on a specific machine, then the source
    code in the SmartSim-Zoo repository is for you. The
    SmartSim development team strives to keep these examples updated with each
    release so that users will always have robust examples for their needs.

 2. SmartSim Applications (completed projects that use SmartSim)

  * The README for the repository describes some of the larger applications of
  * SmartSim. These examples fall under two categories:
    examples by paper and examples by simulation model. The examples by paper are
    based on existing research papers, and the examples by simulation models are
    integrations of SmartSim with existing simulation models.

=================
How To Contribute
=================

We support, encourage, and welcome all contributions to the `SmartSim Zoo
<https://github.com/CrayLabs/SmartSim-Zoo>`__ repository. Instructions for
contributing examples varies whether you are contributing a SmartSim deployment
or a SmartSim application.

 1. Contributing SmartSim Deployment Examples

  * For contributing examples of SmartSim running on a HPC System, we ask that
    you include a description and all references to code and relevant previous
    implementations or open source code that the work is based on for the benefit
    of anyone who would like to try out your example.

 2. Contributing SmartSim Application Examples

  * For contributing examples of completed projects that use SmartSim, we ask that you follow the subsequent contribution template:

   1. Title of the project
   2. High-level description (the 'elevator pitch')
   3. SmartSim features used (e.g. online analysis, ensemble)
   4. Prerequisites: hardware and software
   5. Installation guide
   6. Quickstart
   7. Contact information
   8. Citation

=================
Existing Examples
=================

The subsequent tables summarize the examples provided in the SmartSim Zoo. You
can find a more detailed description of each example in the `SmartSim Zoo
<https://github.com/CrayLabs/SmartSim-Zoo>`__.

.. list-table:: SmartSim Deployment Examples
   :widths: 50 100
   :header-rows: 1
   :align: center

   * - HPC System
     - Organization / Machine Owner
   * - Casper
     - National Center for Atmospheric Research (NCAR)
   * - Cheyenne
     - National Center for Atmospheric Research (NCAR)
   * - Summit
     - Oak Ridge National Lab
   * - Theta
     - Argonne National Lab
   * - ThetaGPU
     - Argonne National Lab

.. list-table:: Current CrayLabs Collaborations
   :widths: 100 150 100
   :header-rows: 1
   :align: center

   * - Paper/Simulation Name
     - Collaborators
     - Links
   * - DeepDriveMD
     - CrayLabs, Argonne National Lab, Oak Ridge National Lab
     - `Implementation <https://github.com/CrayLabs/smartsim-openmm>`__ `Original Paper <https://arxiv.org/abs/1909.07817>`__
   * - TensorFlowFoam
     - CrayLabs, Argonne National Lab
     - `Implementation <https://github.com/CrayLabs/smartsim-openFOAM>`__ `Original Paper <https://arxiv.org/abs/2012.00900>`__
   * - ML-EKE
     - CrayLabs, NCAR, University of Victoria
     - `Implementation <https://github.com/CrayLabs/NCAR_ML_EKE>`__ `Original Paper <https://arxiv.org/abs/2104.09355>`__
   * - LAMMPS + SmartSim
     - CrayLabs, Sandia National Laboratories
     - `Implementation <https://github.com/CrayLabs/smartsim-lammps>`__ `Forked Model <https://github.com/CrayLabs/LAMMPS>`__

----------------------------------------
Summary of SmartSim Application Examples
----------------------------------------

* **DeepDriveMD:** Based on the original DeepDriveMD work, extended to
  orchestrate complex workflows with coupled applications without using the
  filesystem for exchanging information.

* **TensorFlowFoam:** Uses TensorFlow inside of OpenFOAM simulations using
  SmartSim. Displays SmartSim's capability to evaluate a machine learning model
  from within a simulation with minimal external library code and minimal API
  calls.

* **ML-EKE:** Runs an ensemble of simulations all using the SmartSim
  architecture to replace a parameterization (MEKE) within each global ocean
  simulation (MOM6).

* **LAMMPS + SmartSim:** Implementation of a ``SMARTSIM`` dump style which uses
  the SmartRedis clients to stream data to an Orchestrator database created by
  SmartSim.
