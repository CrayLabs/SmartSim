*********************************
Running a Suite of MOM6 Ensembles
*********************************

This tutorial shows how to use SmartSim to automate the process of ensemble
creation and execution at scale. The tutorial is meant to give the user an
example of different ways in which SmartSim can be used to create, launch,
and monitor simulations.

The MOM6 ocean general circulation model has been setup to simulate a classic
oceanographic problem: a two-layer model of a double-gyre system.

The script ``run-double-gyre.py`` manages a SmartSim experiment that sets up
and executes 3 total ensembles. All the ensembles are simulating the same
double gyre system with different model and execution configurations. A total of 24 models are
generated and executed in total.

.. note::

    This tutorial requires that MOM6 be compiled and available in PATH
    as well as a machine with Slurm as the workload manager. Most likely
    the allocation settings will need to be customized for your system.


Overview
========

This experiment will do the following:

 1) Obtain two allocations on different partitions, one with Knights Landing CPUs,
    and the other with IvyBridge CPUs.
 2) Generate and configure two "high resolution" ensembles
 3) Launch the two ensembles onto the two allocations respectively
 4) Monitor and print the statuses of the two ensembles
 5) Release the Knights Landing allocation
 6) Configure and generate a new low resolution ensemble
 7) Launch the low resolution ensemble
 8) Monitor and print the status of each ensemble member

Experiment Setup
================

This experiment is detailed and the descriptions will skip over some of the more
basic portions that are detailed in the `Quick Start <../../../doc/examples/basic.html>`_
and `Ensemble <../../LAMMPS/crack/readme.html>`_ tutorials.

In this experiment, we demonstrate how to create a more complex workflow
that launches multiple ensembles on different allocations and re-uses
allocations to launch new ensembles once the first runs have been completed.

First we initialize our experiment and obtain two allocations on the two
paritions we want our high resolution ensembles to run on.

.. code-block:: python

    from smartsim import Experiment

    # intialize our Experiment and obtain
    # an allocation for our ensemble suite
    experiment = Experiment("double_gyre")
    iv24_alloc = experiment.get_allocation(nodes=16, ppn=48,
                                        partition="iv24", exclusive=None)
    knl_alloc = experiment.get_allocation(nodes=8, ppn=96,
                                        partition="knl", exclusive=None)

Once we have those allocations, we configure how our ensembles should
run on those allocations by specifying their ``run_settings``. The run
settings that are provided to the ``create_ensemble`` method are specifying
how each model in the ensemble should be executed.

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
        "ppn": 48,
        "executable": "MOM6",
        "alloc": iv24_alloc
    }

    knl_run_settings = {
        "nodes": 1,
        "ppn": 96,
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


Once we have defined the run and model parameters and created our ensembles,
we import the ``Generator`` class. This is not usually required except
for cases like this were we want to generate ensembles at different
times within the execution of our script. This can also be very useful
when iteratively programming within a Jupyter Notebook or Python shell.

Since we are writing the model configurations into the input files and
our model requires a directory of input datasets and files, we attach
files to each ensemble instance so that when we generate them, each
model will be populated with configured input files and the input
datasets necessary to run the generated models.

.. code-block:: python

    from smartsim.generation import Generator

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

    # intialize a Generator instance for greater control
    # over when and where generation happens
    generator = Generator()
    generator.generate_ensemble(experiment.exp_path,
                                [high_res_knl, high_res_iv24])


Now that we have created and generated our first two ensembles
we will execute them and wait for them to finish with the
``Experiment.poll()`` method. We turn ``verbose`` to ``False``
because there will be many models running. Since we still would
like to see if the models completed successfully, we utilize
the ``Experiment.get_status()`` method on both ensembles which
returns the workload manager status for each model in the
ensemble after completion.

.. code-block:: python

    # start the two high resolution models on the IV24 and KNL
    # partitions.
    experiment.start(ensembles=[high_res_knl, high_res_iv24])
    experiment.poll(verbose=False)

    # print out the statuses of the model we just ran
    iv24_statuses = experiment.get_status(high_res_iv24)
    print(f"Statuses of IV24 Models: {iv24_statuses}")

    knl_statuses = experiment.get_status(high_res_knl)
    print(f"Statuses of KNL Models: {knl_statuses}")


Since we won't be running any more models on the Knights Landing
nodes of our system, we will be a kind HPC user and release that
allocation for others to use. We do this by specifying to
``Experiment.release()`` the allocation id for the KNL allocation
we obtained earlier.

.. code-block:: python

    # Release the KNL partition because we dont need it anymore
    experiment.release(alloc_id=knl_alloc)

Next we want to re-use our IvyBridge allocation to run another
ensemble with the same tunable model parameter space, but at a
lower resolution. To do this we will configure and generate a
new ensemble just like we did earlier, but change the resolution
values and provide the id of the IV24 allocation.

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
        "ppn": 48,
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
    generator.generate_ensemble(experiment.exp_path, low_res_iv24)

Lastly, we will execute and get the final status of our model
just like we did with the previous ensembles. We will also
release the IvyBridge partition allocation.

.. code-block:: python

    # start the low resolution simulation on the same
    # allocation as the IV24 high resolution model
    experiment.start(ensembles=low_res_iv24)
    experiment.poll(verbose=False)

    # print the statuses of the low resolution ensemble
    # after it has completed.
    iv24_low_res_statuses = experiment.get_status(low_res_iv24)
    print(f"Statuses of IV24 Models (low res): {iv24_low_res_statuses}")

    # Release the iv24 partition
    experiment.release(alloc_id=iv24_alloc)


Experiment Script
=================

The full script from the experiment described above.

.. code-block:: python

    from smartsim import Experiment
    from smartsim.generation import Generator

    # intialize our Experiment and obtain
    # an allocation for our ensemble suite
    experiment = Experiment("double_gyre")
    iv24_alloc = experiment.get_allocation(nodes=16, ppn=48,
                                        partition="iv24", exclusive=None)
    knl_alloc = experiment.get_allocation(nodes=8, ppn=96,
                                        partition="knl", exclusive=None)


    high_res_model_params = {
        "KH": [250, 500, 750, 1000],
        "KHTH": [250, 500],
        "x_resolution": 80,
        "y_resolution": 40,
        "months": 3
    }

    iv24_run_settings = {
        "nodes": 2,
        "ppn": 48,
        "executable": "MOM6",
        "alloc": iv24_alloc
    }

    knl_run_settings = {
        "nodes": 1,
        "ppn": 96,
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

    # intialize a Generator instance for greater control
    # over when and where generation happens
    generator = Generator()
    generator.generate_ensemble(experiment.exp_path,
                                [high_res_knl, high_res_iv24])

    # start the two high resolution models on the IV24 and KNL
    # partitions.
    experiment.start(ensembles=[high_res_knl, high_res_iv24])
    experiment.poll(verbose=False)

    # print out the statuses of the model we just ran
    iv24_statuses = experiment.get_status(high_res_iv24)
    print(f"Statuses of IV24 Models: {iv24_statuses}")

    knl_statuses = experiment.get_status(high_res_knl)
    print(f"Statuses of KNL Models: {knl_statuses}")

    # Release the KNL partition because we dont need it anymore
    experiment.release(alloc_id=knl_alloc)



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
        "ppn": 48,
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
    generator.generate_ensemble(experiment.exp_path, low_res_iv24)


    # start the low resolution simulation on the same
    # allocation as the IV24 high resolution model
    experiment.start(ensembles=low_res_iv24)
    experiment.poll(verbose=False)

    # print the statuses of the low resolution ensemble
    # after it has completed.
    iv24_low_res_statuses = experiment.get_status(low_res_iv24)
    print(f"Statuses of IV24 Models (low res): {iv24_low_res_statuses}")

    # Release the iv24 partition
    experiment.release(alloc_id=iv24_alloc)