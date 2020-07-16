

**********************************
Model Parameter Optimization (MPO)
**********************************


Background
----------

Complex physical models often have tens or hundreds of free parameters which
to modify the behavior of a simulation. Notably, some of these parameters
control parameterizations that approximate unresolved physical processes. For
example, Modular Ocean Model 6 (MOM6) has over 300 free parameters. Model
users must tune these free parameters to optimize the skill of their models.

Model tuning is currently an expensive, mostly ad-hoc process.
Models must be tuned for the resolution they will ultimately be
executed at, and tuning at high-resolution is a time-consuming process
that involves many executions of the model with different sets of
parameters at the resolution of interest. Scientists must also
analyze each run to inform the tuning process.

SmartSim automates the tuning process by integrating with CrayAI.
Effectively, SmartSim handles the generation, configuration, and
launch of the simulations and CrayAI tracks and optimizes over the
model parameter space.

Improving the tuning process of these physical models improves the
productivity of computational scientists by allowing them to quickly
iterate and experiment with new models, data, and scenarios. This
allows faster time-to-insight and improved scientific results
in climate science and other domains.

Prerequisites
=============

To run MPO, in addition to SmartSim being setup, CrayAI must also
be setup and available. For instructions on how to setup CrayAI
see the `github repository <https://cray.github.io/crayai/hpo/hpo.html>`_

How to use MPO
==============

Optimizing model parameters with Smartsim and CrayAI is very similar to how
one would optimize hyperparameters of a machine learning model with a few
extra steps involved. The steps to use SmartSim to optimize model parameters
are listed below:

 1. Compile and install the simulation model
 2. Tag the parameters to optimize in the model input files.
 3. Choose an optimization strategy and write the CrayAI driver script.
 4. Choose an evaluation metric for your model
 5. Write the SmartSim evaluation script using the MPO class

There are two scripts required to run MPO with SmartSim and CrayAI:

 1) Evaluation script, using SmartSim's MPO class
 2) Driver script using typical CrayAI HPO methods.

For a more detailed example see the examples for running `MPO on MOM6 <../examples/MPO/README.html>`_
