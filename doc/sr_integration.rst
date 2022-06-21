*****************************
Integrating into a Simulation
*****************************

========
Overview
========

This document provides some general guidelines to integrate the SmartRedis client
into existing simulation codebase. Developers of these simulation codebases will
need to identify the exact places to add the code, generally only SmartRedis
calls will need to be added in two places:

1. Initialization
2. Main loop

==============
Initialization
==============

+++++++++++++++++++
Creating the client
+++++++++++++++++++

The SmartRedis client must be initialized before it can be used to communicate
with the database. In the C++ and Python versions of the clients, this is done
when creating a new client whereas in the C and Fortran client an `initialize`
method that must be called.

C++::

    #include "client.h"
    SmartRedis::Client client(use_cluster);

Python::

    from smartredis import Client
    client = Client(use_cluster)

Fortran::

    use smartredis_client, only : client_type
    type(client_type) :: client
    return_code = client%initialize(use_cluster)

C::

    #include "client.h"
    void* client = NULL;
    return_code = SmartRedisCClient(use_cluster), &client

All these methods only have one configurable parameter, indicated in the
above cases by the variable `use_cluster`. If this parameter, is true,
then the client expects to be able to communicate with a database with
three or more shards.

++++++++++++++++++++++++++++++++++
(Parallel Programs): Creating keys
++++++++++++++++++++++++++++++++++

For parallel applications, each rank or thread that is communicating with
the database will likely need to create a unique prefix for keys to prevent
another rank or thread inadvertently overwriting data. Any identifier can be
used, though typically the MPI rank number (or equivalent identifier) is a
useful, unique number.

C++::

    const std::string key_prefix << std::format("{:06}_", *rank_id);

Python::

    key_prefix = f"{rank_id:06d}_"

Fortran::

    character(len=12) :: key_prefix
    write(key_prefix,'(A,I6.6)') rank_id

C::

    char[7] key_prefix;
    key_prefix = sprintf(key_prefix", "%06d" rank_id);

++++++++++++++++++++++++++
Storing scripts and models
++++++++++++++++++++++++++

The last task that typically needs to be done is to store models or scripts
that will be used later in the simulation. When using a clustered database,
this only needs to be done by one client (unless ranks required a specific
model or script).

C++::

    if (root_client) {
        client.set_model_from_file(model_key, model_file, backend, device)
    }

Python::

    if root_client:
        client.set_model_from_file(model_key, model_file, backend, device)

Fortran::

    if (root_client) return_code = client%set_model_from_file(model_key, model_file, "TORCH", "CPU")

C::

    if (root_client) {
        return_code = client.set_model_from_file(client, model_key, model_file, backend, device)
    }

=========
Main loop
=========

Within the main loop of the code (e.g. every timestep or iteration of a solver),
the developer uses the SmartRedis client methods to implement a workflow which
may include receiving data, sending data, running a script or model, and/or
retrieving a result. These workflows are covered extensively in the walkthroughs
for the Fortran, C++, and python clients and the integrations with MOM6, OpenFOAM,
LAMMPS, and others.

Generally though, developers are advised to

1. Find locations where file I/O would normally happen and either replace
   or add code to use the SmartRedis client and store the data in the
   database
2. Use the `key_prefix` created during initialization to avoid accidental
   writes/reads from different clients
3. Use the SmartSim `dataset` type when using clients representing decomposed
   subdomains to make the retrieval/use of the data more performant

============
Full example
============

The following pseudocode is used to demonstrate various aspects of instrumenting an
existing simulation code with SmartRedis. This code is representative of solving
the time-evolving heat equation. but will be augmented using an ML model to
provide a preconditioning step each iteration and post the state of the simulation
to the database. ::

    program main

        ! Initialize the model, setup MPI, communications, read input files
        call initialize_model( temperature, number_of_timesteps )

        main_loop: do i=1,number_of_timesteps

            ! Write the current state of the simulation to a file
            call write_current_state(temperature)

            ! Call a time integrator to step the temperature field forward
            call timestep_simulation(temperature)

        enddo
    end program main

Following the guidelines from above, the first step is to initialize the client
and create a unique identifier for the given processor. This should be done
within roughly the same portion of the code where the rest of the model. ::

    ! Import SmartRedis modules
    use, only smartredis_client : client_type

    ! Declare a new variable called client and a string to create a unique
    ! name for keys
    type(client_type) :: smartredis_client
    character(len=7)  :: key_prefix
    integer :: mpi_rank, mpi_code, smartredis_code

    ! Note adding use_cluster as an additional runtime argument for SmartRedis
    call initialize_model(temperature, number_of_timesteps, use_cluster)
    call smartredis_client%initialize(use_cluster)
    call MPI_Comm_rank(MPI_COMM_WORLD, mpi_rank, mpi_code)
    write(key_prefix,'(I6.6,A)') mpi_rank, '_'

    ! Assume all ranks will use the same machine learning model
    if (mpi_rank==0) call set_model_from_file("example_model_key", "path/to/model.pt", "TORCH", "gpu")

Next, add the calls in the main loop to send the temperature to the database ::

    character(len=10), dimension(1) :: model_input, model_output

    main_loop: do i=1,number_of_timesteps

        ! Write the current state of the simulation to a file
        call write_current_state(temperature)
        model_input(1) = key_prefix//"temperature"
        model_output(1) = key_prefix//"temperature_out"
        call smartredis_client%put_tensor(model_input(1))

        ! Run the machine learning model
        return_code = smartredis_client%run_model("example_model_key", model_input, model_output)
        ! The following line overwrites the prognostic temperature array
        return_code = smartredis_client%unpack_tensor(model_output(1), temperature)

        ! Call a time integrator to step the temperature field forward
        call timestep_simulation(temperature)

    enddo

Now when this program runs, every time step the client will be used to the
temperature array in the database, the database will call a machine learning
model to do the inference, the simulation will request the inference,
and finally unpack the array into the existing temperature array. For more
complex examples, please see some of the integrations in the SmartSim Zoo or
feel free to contact the team at CrayLabs@hpe.com
