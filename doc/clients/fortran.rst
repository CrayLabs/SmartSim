
*******
Fortran
*******

Fortran Client Examples
=======================

The following serial and MPI-parallelized examples are practical
demonstrations of how users can use SmartSim with a Fortran program.

Serial example
--------------
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
----------------
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


Fortran Client API
==================

.. doxygenindex::
        :project: fortran_client

