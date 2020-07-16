program exact_key_scalar_unit_test
  use iso_c_binding, only : c_ptr
  use mpi
  use client_fortran_api
  use unit_test_aux

  implicit none

  real(kind=4)                                               :: true_scalar_real_4
  real(kind=4)                                               :: recv_scalar_real_4
  real(kind=8)                                               :: true_scalar_real_8
  real(kind=8)                                               :: recv_scalar_real_8
  integer(kind=4)                                            :: true_scalar_integer_4
  integer(kind=4)                                            :: recv_scalar_integer_4
  integer(kind=8)                                            :: true_scalar_integer_8
  integer(kind=8)                                            :: recv_scalar_integer_8
  integer :: pe_id
  integer :: err_code
  character(len=9) :: key_prefix
  type(c_ptr) :: smartsim_client

  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code )
  write(key_prefix, "(A,I6.6)") "pe_",pe_id
  num_failed = 0
  smartsim_client = init_ssc_client( .false. )

  true_scalar_real_4 = 3.125
  call put_exact_key_scalar(smartsim_client,key_prefix//"test_scalar_real_4",true_scalar_real_4)
  call get_exact_key_scalar(smartsim_client,key_prefix//"test_scalar_real_4",recv_scalar_real_4)
  call check_value(true_scalar_real_4,recv_scalar_real_4,"put/get_"//key_prefix//"test_scalar_real_4")

  true_scalar_real_8 = 3.125
  call put_exact_key_scalar(smartsim_client,key_prefix//"test_scalar_real_8",true_scalar_real_8)
  call get_exact_key_scalar(smartsim_client,key_prefix//"test_scalar_real_8",recv_scalar_real_8)
  call check_value(true_scalar_real_8,recv_scalar_real_8,"put/get_"//key_prefix//"test_scalar_real_8")

  true_scalar_integer_4 = 3
  call put_exact_key_scalar(smartsim_client,key_prefix//"test_scalar_integer_4",true_scalar_integer_4)
  call get_exact_key_scalar(smartsim_client,key_prefix//"test_scalar_integer_4",recv_scalar_integer_4)
  call check_value(true_scalar_integer_4,recv_scalar_integer_4,"put/get_"//key_prefix//"test_scalar_integer_4")

  true_scalar_integer_8 = 3
  call put_exact_key_scalar(smartsim_client,key_prefix//"test_scalar_integer_8",true_scalar_integer_8)
  call get_exact_key_scalar(smartsim_client,key_prefix//"test_scalar_integer_8",recv_scalar_integer_8)
  call check_value(true_scalar_integer_8,recv_scalar_integer_8,"put/get_"//key_prefix//"test_scalar_integer_8")

  call MPI_Finalize(err_code)
  if (num_failed > 0) stop -1
end program exact_key_scalar_unit_test
