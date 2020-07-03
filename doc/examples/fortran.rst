SmartSim Fortran Client
=======================

Requirements
------------
- Fortran 90 (or later) compliant codebase
- Fortran 2003 (or later) compliant compiler
- Compiler chain that can cross-compile Fortran and C, C and C++

Design philosophy
-----------------
The SmartSim Fortran Client is designed to require minimimal modification to
Fortran codebases in order to communicate with the rest of SmartSim. The
functionality of the C and C++ clients are contained within a single, Fortran
2000 compliant module. The public interfaces rely only on primitive Fortran
data structures (e.g. n-dimensional numeric arrays and character arrays).

The Fortran Client relies on the formalized interoperability between C and
Fortran without users needing to ever call the underlying C functions. The
conversion of Fortran primitive types to C-compatible types is handled within
the client. The Fortran Client handles the following operations, all of which
are opaque to the user:

    - Translate Fortran character arrays (used for strings) to a
      null-terminated, C-style string
    - Convert from column-major (Fortran) array convention to row-major
      (C) arrays
    - Handle arbitrarily-indexed Fortran arrays

Client functionality
--------------------------------
The Fortran client has the same functionality as the C and C++ clients. The
Fortran API is much simpler in comparison due to Fortran's native support of
arrays and overloading of functions and subroutines. The former means the
procedure calls are significantly shorter since the shape of the arrays can
inferred using Fortran intrinsics. The latter means that the same procedure
call can be used for different datatypes.

The following four fundamental SmartSim operations are supported for single
and double precision `real` and `integer` (see the API documentation for more
details).

- ``put_array``/``get_array``: Send or receive an arbitrarily indexed array of any
  rank or shape to the database
- ``put_scalar``/``get_scalar``: Send or receive a scalar value from the database
- ``poll_key_and_check_scalar``: A blocking function that polls the database
  for the existence and value of a specific key

Examples using the Fortran Client
--------------------------------------------------------
The following serial and MPI-parallelized examples are practical
demonstrations of how users can use SmartSim with a Fortran program.

Serial example
~~~~~~~~~~~~~~
This example shows the inclusion the SmartSim client in a Fortran program to
enable sending and receiving a matrix of single precision floating point
values. The program allocates a rectangular 2D array and fills it with random
numbers. The array is then stored in the database with an identifying key and
retrieved with the same key.

.. code-block:: fortran
  :linenos:

  program serial_example
      implicit none
      use client_fortran_api, only :: put_array, get_array, init_ssc_client
      use iso_c_binding,      only :: c_ptr

      real, dimension(20,10) :: array_to_send, array_to_receive
      type(c_ptr)            :: smartsim_client

      ! Initialize SmartSim Client
      smartsim_client = init_ssc_client()
      ! Fill array with data
      call RANDOM_NUMBER(array_to_send)
      call put_array(smartsim_client, "example_key", array_to_send)
      call get_array(smartsim_client, "example_key", array_to_receive)
      write(*,*) SUM(array_to_send(:,:) - array_to_receive(:,:))

  end program serial_example

Line 2 imports the SmartSim Fortran Client module with specific module procedures.

Lines 3, 6, and 9 allocate a ``C_PTR`` (an intrinsic Fortran datatype since
F2000) which is used to store a pointer to the C++ SmartSim client. Line 9
calls the C++-based constructor which reads in the relevant environment
variables necessary to make a connection to the database. This need only
needs to be done during the initialization of the model as the memory of the
allocated during construction of the C++ client is only destroyed when the
program completes.

Line 12 sends a 2D array to the database with the identifying key ``example_key``.
This array can then be retrieved from any other client.

Line 13 retrieves the data from the database and stores the result. The
values within this array are bit identical to the sent array.

Parallel example
~~~~~~~~~~~~~~~~
The parallel example instruments the serial example with MPI calls so that multiple
processing elements (PEs) will send arrays to the database in parallel.

.. code-block:: fortran

   :linenos:
   program parallel_example

      use mpi
      use client_fortran_api, only : init_ssc_client
      use iso_c_binding,      only : c_ptr

      implicit none

      type(c_ptr)  :: smartsim_client
      ! MPI related vars
      integer :: pe_id
      integer :: err_code
      integer :: timing_unit
      character(len=10) :: rank_suffix

      ! Initialize MPI
      call MPI_init( err_code )
      call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code )

      smartsim_client = init_ssc_client()
      write(rank_suffix, "(A,I6.6)") "_pe",pe_id

      ! Fill array with data
      call RANDOM_NUMBER(array_to_send)
      call put_array(smartsim_client, "example_key"//rank_suffix, array_to_send)
      call get_array(smartsim_client, "example_key"//rank_suffix, array_to_receive)
      write(*,*) SUM(array_to_send(:,:) - array_to_receive(:,:))

      ! Bring down the MPI communicators
      call MPI_Finalize(err_code)

   end program parallel_example

  The primary difference from the SmartSim perspective is the creation of a
  PE-specific suffix, ``rank_suffix``, based on the MPI rank of the PE. The
  suffix is then appended to the base keyname before being sent to the
  database. This effectively gives each PE its own keyspace and avoids data
  being overwritten by another PE.
