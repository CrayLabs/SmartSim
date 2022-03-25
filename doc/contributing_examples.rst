*********************
Contributing Examples
*********************

What Is the SmartSim Zoo?
#########################
Given that SmartSim is a community developed and maintained project, we have introduced
a `SmartSim Example Zoo <https://github.com/CrayLabs/SmartSim-Zoo>`_ that contains CrayLabs and user contributed examples of using SmartSim
for various simulation and machine learning applications.

The Two Categories of the SmartSim Zoo
**************************************************************
 1. SmartSim Deployments (running SmartSim on various HPC Systems)

  * The source code for the repository serves the purpose of showing diverse examples of how to get SmartSim running on
    different HPC Systems. If you are looking for working examples on a specific machine, then the source code in the
    SmartSim-Zoo repository is for you. The SmartSim development team strives to keep these examples updated with each
    release so that users will always have robust examples for their needs.

 2. SmartSim Applications (completed projects that use SmartSim)

  * The README for the repository describes some of the larger applications of SmartSim. These examples fall under two categories;
    examples by paper and examples by simulation model. The examples by paper are based on existing research papers e.g. , and the examples
    by simulation models are integrations of SmartSim with existing simulation models.

How To Contribute
#################
We support, encourage, and welcome all contributions to the `SmartSim Zoo <https://github.com/CrayLabs/SmartSim-Zoo>`_ repository. Instructions for
contributing examples varies whether you are contributing a SmartSim deployment or a SmartSim application.

 1. Contributing SmartSim Deployment Examples

  * For contributing examples of SmartSim running on a HPC System, we ask that you include a description and all references to code and
    relevant previous implementations or open source code that the work is based on for the benefit of anyone who would like to
    try out your example.

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

Existing Examples
#################
The subsequent tables summarize the examples provided in the SmartSim Zoo. You can find a more detailed
description of each example in the `SmartSim Zoo <https://github.com/CrayLabs/SmartSim-Zoo>`_.

.. list-table:: SmartSim Deployment Examples
   :widths: 50 50
   :header-rows: 1
   :align: center

   * - HPC System
     - Contributing User(s)
   * - Casper
     - @jedwards4b
   * - Cheyenne
     - CrayLabs
   * - Summit
     - CrayLabs
   * - Theta
     - CrayLabs
   * - ThetaGPU
     - CrayLabs

.. list-table:: SmartSim Application Examples
   :widths: 50 50
   :header-rows: 1
   :align: center

   * - Name
     - Contributing User(s)
   * - `DeepDriveMD <https://github.com/CrayLabs/smartsim-openmm>`_
     - CrayLabs
   * - `TensorFlowFoam <https://arxiv.org/abs/2012.00900>`_
     - CrayLabs
   * - `ML-EKE <https://github.com/CrayLabs/NCAR_ML_EKE>`_
     - CrayLabs
   * - `LAMMPS + SmartSim <https://github.com/CrayLabs/smartsim-lammps>`_
     - CrayLabs
