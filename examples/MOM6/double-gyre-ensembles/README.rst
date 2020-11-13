*************************************
Running a Suite of Ensembles on Slurm
*************************************

This tutorial shows how to use SmartSim to automate the process of ensemble
creation and execution at scale. The tutorial is meant to give the user an
example of different ways in which SmartSim can be used to create, and launch
ensembles of varying configurations on slurm based supercomputers.

The MOM6 ocean general circulation model has been setup to simulate a classic
oceanographic problem: a two-layer model of a double-gyre system.

The script ``run-double-gyre.py`` manages a SmartSim experiment that sets up
and executes 3 total ensembles. All the ensembles are simulating the same
double gyre system with different model and execution configurations. A total of 32 models are
generated and executed in total.

.. note::

    This tutorial requires that MOM6 be compiled and available in PATH
    as well as a machine with Slurm as the workload manager. Most likely
    the allocation settings will need to be customized for your system.


Overview
========

This experiment will do the following:

 1) Obtain two slurm allocations on different partitions, one with Knights Landing CPUs,
    and the other with IvyBridge CPUs.
 2) Configure and generate two "high resolution" ensembles
 3) Configure and generate a new low resolution ensemble
 4) Launch the first two ensembles onto the two allocations respectively
 5) Monitor and print the statuses of the two ensembles
 6) Release the Knights Landing allocation
 7) Launch the low resolution ensemble
 8) Monitor and print the status of each ensemble member

Experiment Setup
================

This experiment is detailed and the descriptions will skip over some of the more
basic portions that are detailed in documentation.

In this experiment, we demonstrate how to create a more complex workflow
that launches multiple ensembles on different allocations and re-uses
allocations to launch new ensembles once the first runs have been completed.

First we initialize our experiment and obtain two allocations on the two
partitions we want our high resolution ensembles to run on.

.. code-block:: python

    from smartsim import Experiment
    from smartsim import slurm

    # initialize our Experiment and obtain
    # allocations for our ensemble suite
    experiment = Experiment("double_gyre", launcher="slurm")

    iv24_opts = {
        "partition": "iv24",
        "ntasks-per-node": 48
    }
    iv24_alloc = slurm.get_slurm_allocation(nodes=16, add_opts=iv24_opts)

    knl_opts = {
        "partition": "knl",
        "ntasks-per-node": 48
    }
    knl_alloc = slurm.get_slurm_allocation(nodes=8, add_opts=knl_opts)


Once we have those allocations, we configure how our ensembles should
run on those allocations by specifying their ``run_settings``. The run
settings that are provided to the ``create_ensemble`` method are specifying
how each model in the ensemble should be executed through Slurm.

We also provide the ensembles with the values of model parameters that will
be used to create the sets of model parameters for each model in our
ensemble.

We have already tagged the fields: ``KH``, ``KHTH``, ``x_resolution``,
``y_resolution``, and ``months`` in our models input files so that
our ensemble generator can easily find and populate those fields with
our model parameters.

.. code-block:: python

    high_res_model_params = {
        "KH": [250, 500, 750, 1000],
        "KHTH": [250, 500],
        "x_resolution": 80,
        "y_resolution": 40,
        "months": 3
    }

    iv24_run_settings = {
        "nodes": 2,
        "ntasks-per-node": 48,
        "executable": "MOM6",
        "alloc": iv24_alloc
    }

    knl_run_settings = {
        "nodes": 1,
        "ntasks-per-node": 96,
        "executable": "MOM6",
        "alloc": knl_alloc
    }

    # create the ensemble that will run on the iv24 nodes
    high_res_iv24 = experiment.create_ensemble(
        "high-res-iv24",
        params=high_res_model_params,
        run_settings=iv24_run_settings
        )

    # create the ensemble that will run on the knights landing nodes
    high_res_knl = experiment.create_ensemble(
        "high-res-knl",
        params=high_res_model_params,
        run_settings=knl_run_settings
        )


Since we are writing the model configurations into the input files and
our model requires a directory of input datasets and files, we attach
files to each ensemble instance so that when we generate them, each
model will be populated with configured input files and the input
datasets necessary to run the generated models.

.. code-block:: python

    high_res_iv24.attach_generator_files(
        to_copy=["./MOM6_base_config"],
        to_configure=["./MOM6_base_config/input.nml",
                    "./MOM6_base_config/MOM_input"]
        )
    high_res_knl.attach_generator_files(
        to_copy=["./MOM6_base_config"],
        to_configure=["./MOM6_base_config/input.nml",
                    "./MOM6_base_config/MOM_input"]
        )

We follow the exact same process as the high-resolution ensembles,
for the low resolution ensemble we want to create. We then generate
the files needed for all the ensembles needed which will write the
model parameters into the configuration files we attached.

.. code-block:: python

    # configure and create the low resolution
    # double gyre ensemble
    low_res_model_params = {
        "KH": [250, 500, 750, 1000],
        "KHTH": [250, 500, 750, 1000],
        "x_resolution": 40,
        "y_resolution": 40,
        "months": 3
    }
    low_res_run_settings = {
        "nodes": 1,
        "ntasks-per-node": 48,
        "executable": "MOM6",
        "alloc": iv24_alloc
    }

    low_res_iv24 = experiment.create_ensemble(
        "low-res-iv24",
        params=low_res_model_params,
        run_settings=low_res_run_settings
        )
    low_res_iv24.attach_generator_files(
        to_copy=["./MOM6_base_config"],
        to_configure=["./MOM6_base_config/input.nml",
                    "./MOM6_base_config/MOM_input"])

    # generate the files needed for all of our models
    experiment.generate(high_res_knl, high_res_iv24, low_res_iv24

Now that we have created and generated our three ensembles
we will execute two of them and wait for them to finish with the argument
``block=True`` in the ``Experiment.start`` method. We utilize
the ``Experiment.get_status()`` method on both ensembles which
returns the SmartSim status for each model in the
ensemble after completion.

.. code-block:: python

    # start the two high resolution models on the IV24 and KNL
    # partitions.
    experiment.start(high_res_knl, high_res_iv24, block=True, summary=True)

    # print out the statuses of the model we just ran
    iv24_statuses = experiment.get_status(high_res_iv24)
    print(f"Statuses of IV24 Models: {iv24_statuses}")

    knl_statuses = experiment.get_status(high_res_knl)
    print(f"Statuses of KNL Models: {knl_statuses}")


Since we won't be running any more models on the Knights Landing
nodes of our system, we will be a kind HPC user and release that
allocation for others to use. We do this by specifying to
``slurm.release_slurm_allocation()`` the allocation id for the KNL allocation
we obtained earlier.

.. code-block:: python

    # Release the KNL partition because we dont need it anymore
    slurm.release_slurm_allocation(knl_alloc)

Next we want to re-use our IvyBridge allocation to run another
ensemble with the same tunable model parameter space, but at a
lower resolution. We already created, configured, and generated
this model, so we can just pass it to the ``Experiment.start()``
method. We also get the statuses upon completion and release
the iv24 allocation.

.. code-block:: python

    # start the low resolution simulation on the same
    # allocation as the IV24 high resolution model
    experiment.start(low_res_iv24, block=True, summary=False)

    # print the statuses of the low resolution ensemble
    # after it has completed.
    iv24_low_res_statuses = experiment.get_status(low_res_iv24)
    print(f"Statuses of IV24 Models (low res): {iv24_low_res_statuses}")

    # Release the iv24 partition
    slurm.release_slurm_allocation(iv24_alloc)

The final experiment directory should contain the following
directories with the input, output, and data from each model.

.. code-block:: text

    .
    ├── high-res-iv24
    │   ├── high-res-iv24_0
    │   ├── high-res-iv24_1
    │   ├── high-res-iv24_2
    │   ├── high-res-iv24_3
    │   ├── high-res-iv24_4
    │   ├── high-res-iv24_5
    │   ├── high-res-iv24_6
    │   └── high-res-iv24_7
    ├── high-res-knl
    │   ├── high-res-knl_0
    │   ├── high-res-knl_1
    │   ├── high-res-knl_2
    │   ├── high-res-knl_3
    │   ├── high-res-knl_4
    │   ├── high-res-knl_5
    │   ├── high-res-knl_6
    │   └── high-res-knl_7
    └── low-res-iv24
        ├── low-res-iv24_0
        ├── low-res-iv24_1
        ├── low-res-iv24_10
        ├── low-res-iv24_11
        ├── low-res-iv24_12
        ├── low-res-iv24_13
        ├── low-res-iv24_14
        ├── low-res-iv24_15
        ├── low-res-iv24_2
        ├── low-res-iv24_3
        ├── low-res-iv24_4
        ├── low-res-iv24_5
        ├── low-res-iv24_6
        ├── low-res-iv24_7
        ├── low-res-iv24_8
        └── low-res-iv24_9



Experiment Script
=================

The full script from the experiment described above.

.. code-block:: python

    from smartsim import Experiment
    from smartsim import slurm

    # initialize our Experiment and obtain
    # allocations for our ensemble suite
    experiment = Experiment("double_gyre", launcher="slurm")

    iv24_opts = {
        "partition": "iv24",
        "ntasks-per-node": 48
    }
    iv24_alloc = slurm.get_slurm_allocation(nodes=16, add_opts=iv24_opts)

    knl_opts = {
        "partition": "knl",
        "ntasks-per-node": 48
    }
    knl_alloc = slurm.get_slurm_allocation(nodes=8, add_opts=knl_opts)

    high_res_model_params = {
        "KH": [250, 500, 750, 1000],
        "KHTH": [250, 500],
        "x_resolution": 80,
        "y_resolution": 40,
        "months": 3
    }

    iv24_run_settings = {
        "nodes": 2,
        "ntasks-per-node": 48,
        "executable": "MOM6",
        "alloc": iv24_alloc
    }

    knl_run_settings = {
        "nodes": 1,
        "ntasks-per-node": 96,
        "executable": "MOM6",
        "alloc": knl_alloc
    }

    # create the ensemble that will run on the iv24 nodes
    high_res_iv24 = experiment.create_ensemble(
        "high-res-iv24",
        params=high_res_model_params,
        run_settings=iv24_run_settings
        )
    high_res_iv24.attach_generator_files(
        to_copy=["./MOM6_base_config"],
        to_configure=["./MOM6_base_config/input.nml",
                    "./MOM6_base_config/MOM_input"]
        )

    # create the ensemble that will run on the knights landing nodes
    high_res_knl = experiment.create_ensemble(
        "high-res-knl",
        params=high_res_model_params,
        run_settings=knl_run_settings
        )
    high_res_knl.attach_generator_files(
        to_copy=["./MOM6_base_config"],
        to_configure=["./MOM6_base_config/input.nml",
                    "./MOM6_base_config/MOM_input"]
        )

    # configure and create the low resolution
    # double gyre ensemble
    low_res_model_params = {
        "KH": [250, 500, 750, 1000],
        "KHTH": [250, 500, 750, 1000],
        "x_resolution": 40,
        "y_resolution": 40,
        "months": 3
    }
    low_res_run_settings = {
        "nodes": 1,
        "ntasks-per-node": 48,
        "executable": "MOM6",
        "alloc": iv24_alloc
    }

    low_res_iv24 = experiment.create_ensemble(
        "low-res-iv24",
        params=low_res_model_params,
        run_settings=low_res_run_settings
        )
    low_res_iv24.attach_generator_files(
        to_copy=["./MOM6_base_config"],
        to_configure=["./MOM6_base_config/input.nml",
                    "./MOM6_base_config/MOM_input"])

    # generate the files needed for all of our models
    experiment.generate(high_res_knl, high_res_iv24, low_res_iv24)

    # start the two high resolution models on the IV24 and KNL
    # partitions.
    experiment.start(high_res_knl, high_res_iv24, block=True, summary=True)

    # print out the statuses of the model we just ran
    iv24_statuses = experiment.get_status(high_res_iv24)
    print(f"Statuses of IV24 Models: {iv24_statuses}")

    knl_statuses = experiment.get_status(high_res_knl)
    print(f"Statuses of KNL Models: {knl_statuses}")

    # Release the KNL partition because we dont need it anymore
    slurm.release_slurm_allocation(knl_alloc)

    # start the low resolution simulation on the same
    # allocation as the IV24 high resolution model
    experiment.start(low_res_iv24, block=True, summary=False)

    # print the statuses of the low resolution ensemble
    # after it has completed.
    iv24_low_res_statuses = experiment.get_status(low_res_iv24)
    print(f"Statuses of IV24 Models (low res): {iv24_low_res_statuses}")

    # Release the iv24 partition
    slurm.release_slurm_allocation(iv24_alloc)