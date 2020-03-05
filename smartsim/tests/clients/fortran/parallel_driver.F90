!> Example program used to debug SmartSim client interface to Redis
program parallel_driver 
  use mpi
  use unit_tests,         only : test_array
  use client_fortran_api, only : init_ssc_client
  use iso_c_binding,      only : c_ptr

  implicit none

  type(c_ptr)  :: ssc_client !< Pointer to initialized SSC 
  ! MPI related vars
  integer :: pe_id
  integer :: err_code
  integer, parameter :: ni = 100
  character(len=1024) :: key_suffix

  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code )

  ssc_client = init_ssc_client()
  write(key_suffix, "(A,I0.5)") "pe_",pe_id
  call test_array(ssc_client,ni,key_suffix)

  call MPI_Finalize(err_code)
end program parallel_driver 
