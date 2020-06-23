Running MOM6 within SmartSim
----------------------------

.. note::
    The version of the MOM6 codebase discussed here is based on commit 
    `e006df <https://github.com/ashao/MOM6/commit/e006df21cabb620666dca8b6a8aaa59c4f51822c>`_

The MOM6 ocean general circulation model has been setup to simulate a classic
oceanographic problem: a two-layer model of a double-gyre system.

The script `run-double-gyre.py` manages a SmartSim experiment that sets up
and runs 8 total simulations of the double gyre simulations varying resolution, viscosity, and strength of a turbulence parameterization.

The script first initializes a SmartSim experiment called 'double_gyre'
(`create_experiment`) and submits a job allocation request (`get_allocation`)
to slurm for 8 nodes (one node per ensemble member). The script then defines
two ensembles to be included as part of the experiment, each with a different
resolution (i.e. number of points in the horizontal directions). No
generation scheme has been specified, so the ensemble members cover every
permutation of the tuneable model parameters KH and KHTR.

Up to this point, the script has primarily configured the SmartSim experiment
and has no information about the model itself. `attach_generator_files`
specifies wthe files which MOM6 itself it needs to configure the baseline
double-gyre directory, which is specified by the `to_copy` argument. Each
ensemble member will have its own copy of the this directory. Individual files
specified in the `to_configure` argument will be modified when the experiment is
created. These files have been modified already to indicate where the 
parameters KH, KHTH, x, y, and months unique to each ensemble member will be
subsituted. These have been tokenized by enclosing the parameter with
semicolons (e.g. in `MOM_input` `KH` was made the configurable parameter by
replacing it with `;KH;`.

The configuration of the SmartSim experiment is now complete and
`generate_experiment` is then called to create the run directories on disk
for each ensemble member and to substitute the tokens for specific values.

Lastly, each ensemble member is then run by calling `experiment.start()`
(with as many concurrently as allowed by the original allocation and
resources requested). The status of the experiment is then polled every so often
to ensure that each ensemble member is still running (`experiment.poll()`). Upon
completion, the allocation is released (`experiment.release()`) which frees all
the compute resources used by the experiment.