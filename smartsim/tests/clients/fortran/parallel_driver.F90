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
  integer :: timing_unit
  character(len=1024) :: key_suffix
  integer :: nml_unit, nml_status
  ! Namelist integers 
  namelist /control/ ni
  integer(kind=8) :: ni = 4 

  ! Set default values for namelist
  ni = 100

  open( unit=nml_unit, file='client.nml', form='FORMATTED', status='old', access='SEQUENTIAL', iostat = nml_status)
  if (nml_status == 0) then
    read(nml_unit,control)
    close(nml_unit)
  endif 

  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code )

  ssc_client = init_ssc_client()
  write(key_suffix, "(A,I6.6)") "pe_",pe_id
 
  ! Open files to store timing information
  open(newunit=timing_unit, file=trim(key_suffix), status='replace')
  call test_array(ssc_client,ni,key_suffix,timing_unit)
  close(timing_unit)
  call MPI_Finalize(err_code)
end program parallel_driver 
