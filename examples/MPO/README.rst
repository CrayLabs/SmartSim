
*************************
Optimize Model Parameters
*************************

In this example, we will show how to use SmartSim and CrayAI to optimize
simulation model parameters using techniques commonly used in
hyper-parameter optimization for machine learning models.

There are two script required to run the example:

 1) The evaluation script
 2) The CrayAI driver script

In the evaluation script, users will use the MPO class provided by
SmartSim to configure, generate, and run an instance of the simulation,
as well as evaluate that model to provide the optimization strategy
some figure of merit. This figure of merit (FoM) is used by CrayAI
to optimize over the parameter space of the simulation.

The CrayAI driver script tracks and optimizes over the parameter space
of the simulation provided by the user. Each evaluation, a new set
of parameters are given to the evaluation script and a figure of
merit is retrieved from the evaluation script and stored with CrayAI.

In this example, we will optimize parameters for the Modular Ocean
Model 6 (MOM6). The two parameters we will optimize are KH (eddy viscosity)
and KHTH (thickness diffusion). To evaluate how each set of values for
KH and KHTH perform in the model, we choose to calculate Jet Penetration
for each model evaluated and compare it to the Jet Penetration of a
high resolution (5km) version of the same model. The difference between
the jet penetration values squared represents the figure of merit for
each model provided to CrayAI.

Below we will walk through every step and show how a user of SmartSim
could apply this optimization strategy to whatever model they please.


Writing the Evaluation Script
=============================

Tagging Model Input Files
-------------------------

Assuming the simulation is compiled and installed, the first step in setting up an
MPO experiment is identifying and tagging the input files that correspond to the
parameters that you want to optimize.

Below is an example of an input file from Modular Ocean Model 6 (MOM6) where the
value of KH has been tagged with the default tag: ``;``

.. code-block::

    ! === module MOM_hor_visc ===
    LAPLACIAN = True                !   [Boolean] default = False
                                    ! If true, use a Laplacian horizontal viscosity.
    BIHARMONIC = False              !   [Boolean] default = True
                                    ! If true, se a biharmonic horizontal viscosity.
                                    ! BIHARMONIC may be used with LAPLACIAN.
    KH = ;KH;                    !   [m2 s-1] default = 0.0
                                    ! The background Laplacian horizontal viscosity.
    KH_VEL_SCALE = 0.003            !   [m s-1] default = 0.0
                                    ! The velocity scale which is multiplied by the grid
                                    ! spacing to calculate the Laplacian viscosity.
                                    ! The final viscosity is the largest of this scaled
                                    ! viscosity, the Smagorinsky viscosity and KH

    ! === module MOM_thickness_diffuse ===
    KHTH = ;KHTH;                    !   [m2 s-1] default = 0.0
                                    ! The background horizontal thickness diffusivity.
    KHTH_USE_FGNV_STREAMFUNCTION = False !   [Boolean] default = False
                                    ! If true, use the streamfunction formulation of
                                    ! Ferrari et al., 2010, which effectively emphasizes
                                    ! graver vertical modes by smoothing in the vertical.

Configuring the Model Evaluation
--------------------------------

The value within the tag (currently a placeholder ``KH`` must correspond directly)
with the value provided as the ``tunable_params`` argument to the ``MPO`` initializer
and the name of the value provided to the ``crayai.hpo.params`` initializer.

In the evaluation script, we provide the parameter names with defaults:

.. code-block:: python

    tunable_params = {"KH": 1000,
                     "KHTH": 1000}

Any other parameters to be written into the input files of each simulation
can be provided as ``model_params`` to the ``MPO`` class. For intance, in
this example, we also keep the resolution and length of the simulation as
controllable parameters from the evaluation script so that they can be
easily changed. Note that this parameters must also be tagged in the
input files of the simulation as well.

.. code-block:: python

    model_params =  {"months": 12,
                     "x_resolution": 40,
                     "y_resolution": 40}

Finally, we configure how each model should be run on our machine. We
do this by specifying the ``run_settings`` of the model. In this case
we will run each model on a single node with 48 processors. We
specify the executable as ``MOM6`` and our slurm partition as ``iv24``.
For more on configuring how models are run within SmartSim on different
architectures, see `the launcher documentation <../../doc/launchers.html>`_.

.. code-block:: python

    run_settings = {"nodes":1,
                    "ppn": 48,
                    "executable":"MOM6",
                    "partition": "iv24"}

After we have configured these parts of the simulation, we are ready to
initalize the MPO class and the model that will be evaluated.

.. code-block:: python

    from smartsim import MPO

    # initialize the fields needed by MPO in each
    # evaluation run.
    tunable_params = {"KH": 1000,
                    "KHTH": 1000}
    model_params =  {"months": 12,
                    "x_resolution": 40,
                    "y_resolution": 40}
    run_settings = {"nodes":1,
                    "ppn": 48,
                    "executable":"MOM6",
                    "partition": "iv24"}

    # intialize the MPO instance and name the data directory "MOM6-mpo"
    mpo = MPO(tunable_params, data_dir="MOM6-mpo")

    # initialize the model we want to evaluate.
    # configure and copy needed model files into the
    # directory where the evaluation model will be run.
    model = mpo.init_model(run_settings, model_params=model_params)


Input Datasets and Configurations
---------------------------------

In some cases, your model may rely on input files that are not to be
read and written by SmartSim, but do need to be included in the
directory in which the executable will be run. SmartSim can handle
three types of these files: files to copy, files to configure, and
files to symlink. In this case, we will copy over the base configuration
for the double gyre model of MOM6, and specify that we need to be
able to configure the files: ``MOM_input`` and ``input.nml``. We
do this through a method call on the object returned by the ``MPO.init_model``
method as follows:

.. code-block:: python

    model.attach_generator_files(
        to_copy=["../MOM6/MOM6_base_config/"],
        to_configure=["../MOM6/MOM6_base_config/input.nml",
                    "../MOM6/MOM6_base_config/MOM_input"])


Executing the Model
-------------------

Finally, to run the model we call ``MPO.run()``:

.. code-block:: python

    mpo.run()


At this point, users can test the script and ensure that their
model runs with the default configuration values specified in
the ``tunable_params`` dictionary. The only extra step required
is that if one is testing this, they will need to obtain and
relay an allocation id from slurm. To run the MPO script at this
point one can simply call the script with the ``alloc`` argument
as follows:

.. code-block:: bash

    python eval-script.py --alloc 123456

Later we will show how to remove the requirement of obtaining an
allocation for each run.


Performing the Evaluation
-------------------------

Once the user has tested and ensured their model can run
from the evaluation script, the next step is to calcuate the
figure of merit for the model that CrayAI will use for optimization.

The MPO class provides a couple of helpful methods to retrieve
the evaluation parameters provided by CrayAI at each iteration
and files generated by the simulation for analysis. We utilize
one of these methods, ``MPO.get_model_file()`` to retrieve two
files that we need to calcuate the figure of merit, jet penetration,
for each model.

Once the figure of merit has been calculated, we need to print
it so that CrayAI can obtain and track it for the optimization
process.

.. code-block:: python

    import xarray as xr

    # get data produced by the simulation
    data_path = mpo.get_model_file("ocean_mean_month.nc")
    grid_path = mpo.get_model_file("ocean_geometry.nc")

    # perform evaluation to calculate figure of merit
    data = xr.open_dataset(data_path, decode_times=False)
    grid = xr.open_dataset(grid_path,
                            decode_times=False).rename({'lonh' : 'xh',
                                                        'lath' : 'yh'})

    # calculate MSE of jet penetration between the
    # evaluated model and high resolution data which
    # we will use at the figure of merit
    num = (data.KE.sum("zl")*grid.geolon*grid.Ah).sum(("xh","yh"))
    denom = (data.KE.sum("zl")*grid.Ah).sum(("xh","yh"))
    jp = (num/denom).mean("time").values
    fom = (jp - 17)**2 # 17 is a rough guess; squaring the error

    # print figure of merit for CrayAI optimizer
    print("FoM:", fom)


CrayAI Driver Script
====================

Choosing a Strategy
-------------------

Three scripts are included in this directory to show how
optimization methods can be interchanged easily. We will
dicuss them briefly here, but for more information, please
see the CrayAI repository.

The ``Grid`` strategy is useful if you have a specific set of
parameters that you wish for the evaluation script to be
run with. This strategy performs a grid search of parameters
within the ranges provided by the user.

The ``Random`` strategy randomly chooses parameters from within
the ranges provided by the user. This strategy is usually a useful
first step for optimization to get an idea of ranges to use for
something like Grid or Genetic.

The ``Genetic`` strategy intializes populations of candidate models
and evolves those populations through multiple iterations of
evaluation. This strategy often produces the best results for
many use cases.


Initializing the Parameter Space
--------------------------------


In the CrayAI driver script, we provide the parameter names with defaults
and ranges to optimize over. Note that these must be the exact same names
as the placeholders in the input files and the values within the ``tunable_params``
argument within the evaluation script.

.. code-block:: python

    # Define model parameter space
    params = hpo.Params([["--KH", 2000, (0, 4000)],
                        ["--KHTH", 2000, (0, 4000)]])


Setting up the Evaluator
------------------------

SmartSim controls the configuration and launch of each of the candidate
models, however, the optimization process is controlled by the evaluator
within CrayAI. This includes the number of parallel executions of
candidate models.

Since SmartSim is actually launching the model, we tell CrayAI to
launch the evaluation script "locally", but we do specify that we
would like 20 candidate evaluations to be run in parallel as follows:

.. code-block:: python

    # Define the evaluator
    cmd = f"python -u eval-script.py --alloc {alloc}"

    evaluator = hpo.Evaluator(cmd,
                            workload_manager='local',
                            num_parallel_evals=20,
                            verbose=True)

One important piece is the ``--alloc {alloc}`` portion of the
command specified to the evaluator. This string allows for
SmartSim to obtain and provide an allocation for each
evaluation model. A single allocation is used for every single
candidate model. We let SmartSim control the allocation by
obtaining it in the CrayAI driver script as follows:

.. code-block:: python

    exp = Experiment("MPO")
    alloc = exp.get_allocation(nodes=20, partition="iv24", time="10:00:00")

    # .. <CrayAI code goes here>

    exp.release()


Initializing the Optimizer
--------------------------

The optimizer can be initialized and executed exactly like it would be
in CrayAI. We provide three examples with different strategies, but
only one is needed.

.. code-block:: python

    # Define random Optimizer
    optimizer = hpo.RandomOptimizer(evaluator,
                                    num_iters=100,
                                    verbose=True


    optimizer = hpo.GridOptimizer(evaluator,
                                verbose=True,
                                grid_size=10,
                                chunk_size=20)

    optimizer = hpo.genetic.Optimizer(evaluator,
                                    pop_size= 10,
                                    num_demes=2,
                                    generations=5,
                                    mutation_rate=0.05,
                                    crossover_rate=0.4,
                                    verbose=True )


    # Run the optimizer over the model parameter space
    optimizer.optimize(params)


Full CrayAI Driver Script
-------------------------

The full driver script for CrayAI

.. code-block:: python

    from crayai import hpo
    from smartsim import Experiment

    exp = Experiment("MPO")
    alloc = exp.get_allocation(nodes=20, partition="iv24",
                            time="10:00:00", exclusive=None)

    # Define model parameters and ranges
    params = hpo.Params([["--KH", 2000, (0, 4000)],
                        ["--KHTH", 2000, (0, 4000)]])

    # Define the evaluator
    cmd = f"python -u eval-script.py --alloc {alloc}"

    evaluator = hpo.Evaluator(cmd,
                            workload_manager='local',
                            num_parallel_evals=20,
                            verbose=True)

    optimizer = hpo.GridOptimizer(evaluator,
                                verbose=True,
                                grid_size=10,
                                chunk_size=20)

    # Run the optimizer over the model parameters
    optimizer.optimize(params)

    exp.release()


Full Evaluation Script
----------------------

The full evaluation script

.. code-block::

    import xarray as xr
    from smartsim import MPO

    # initialize the fields needed by MPO in each
    # evaluation run.
    tunable_params = {"KH": 1000,
                    "KHTH": 1000}
    model_params =  {"months": 12,
                    "x_resolution": 40,
                    "y_resolution": 40}
    run_settings = {"nodes":1,
                    "ppn": 48,
                    "executable":"MOM6",
                    "partition": "iv24"}

    # intialize the MPO instance and name the data directory "MOM6-mpo"
    mpo = MPO(tunable_params, data_dir="MOM6-mpo")

    # initialize the model we want to evaluate.
    # configure and copy needed model files into the
    # directory where the evaluation model will be run.
    model = mpo.init_model(run_settings, model_params=model_params)
    model.attach_generator_files(
        to_copy=["../MOM6/MOM6_base_config/"],
        to_configure=["../MOM6/MOM6_base_config/input.nml",
                    "../MOM6/MOM6_base_config/MOM_input"])

    # Start the underlying experiment that
    # contains the generated and configured model
    # we are optimizing.
    mpo.run()

    # get data produced by the simulation
    data_path = mpo.get_model_file("ocean_mean_month.nc")
    grid_path = mpo.get_model_file("ocean_geometry.nc")

    # perform evaluation to calculate figure of merit
    data = xr.open_dataset(data_path, decode_times=False)
    grid = xr.open_dataset(grid_path,
                            decode_times=False).rename({'lonh' : 'xh',
                                                        'lath' : 'yh'})

    # calculate MSE of jet penetration between the
    # evaluated model and high resolution data which
    # we will use at the figure of merit
    num = (data.KE.sum("zl")*grid.geolon*grid.Ah).sum(("xh","yh"))
    denom = (data.KE.sum("zl")*grid.Ah).sum(("xh","yh"))
    jp = (num/denom).mean("time").values
    fom = (jp - 17)**2 # 17 is a rough guess; squaring the error

    # print figure of merit for CrayAI optimizer
    print("FoM:", fom)




Background on Parameters Optimized in this Example
===================================================

KH effectively increases the friction throughout the ocean basin. KH reduces horizontal gradients in velocity
as energy is transferred more efficiently between fluids moving at two
different speeds. KH also serves to remove energy from the system by
acting as a dampening effect on momentum. In the case of the double
gyre system, a high viscosity stretches the western boundary current
over a wider distance. This weakens the boundary current’s effect on
the basin.

KHTH acts on another type of turbulence in the model that
arises from baroclinic instability. Turbulence from baroclinic
instability arises from the vertical changes in density of the
ocean model. KHTH serves to extract energy from the sloping gradients
in the vertical and flattens them. The total amount of turbulence
in the eddy-permitting cases is directly effected by the tunable parameter space.
