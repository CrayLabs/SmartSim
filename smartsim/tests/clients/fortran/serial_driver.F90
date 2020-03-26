!> Example program used to debug SmartSim client interface to Redis
program serial_driver 
  use unit_tests,         only : test_array
  use client_fortran_api, only : init_ssc_client
  use iso_c_binding,      only : c_ptr

  implicit none

  type(c_ptr)  :: ssc_client !< Pointer to initialized SSC 

  integer(kind=8), parameter :: ni = 5000000

  ssc_client = init_ssc_client()
  call test_array(ssc_client,ni,"serial")

end program serial_driver 
