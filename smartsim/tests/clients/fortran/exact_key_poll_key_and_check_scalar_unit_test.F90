program exact_key_poll_key_and_check_scalar_unit_test
  use iso_c_binding, only : c_ptr
  use mpi
  use client_fortran_api
  use unit_test_aux

  implicit none

  logical                                                    :: result_poll_key_and_check_scalar_real_4
  real(kind=4)                                               :: true_real_4
  real(kind=4)                                               :: false_real_4
  logical                                                    :: result_poll_key_and_check_scalar_real_8
  real(kind=8)                                               :: true_real_8
  real(kind=8)                                               :: false_real_8
  logical                                                    :: result_poll_key_and_check_scalar_integer_4
  integer(kind=4)                                            :: true_integer_4
  integer(kind=4)                                            :: false_integer_4
  logical                                                    :: result_poll_key_and_check_scalar_integer_8
  integer(kind=8)                                            :: true_integer_8
  integer(kind=8)                                            :: false_integer_8
  integer :: pe_id
  integer :: err_code
  character(len=9) :: key_prefix
  type(c_ptr) :: smartsim_client

  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code )
  write(key_prefix, "(A,I6.6)") "pe_",pe_id
  num_failed = 0
  smartsim_client = init_ssc_client( .false. )

  true_real_4 = 3.125
  false_real_4 = 4.125
  call put_exact_key_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_real_4",true_real_4)
  result_poll_key_and_check_scalar_real_4 = &
        poll_exact_key_and_check_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_real_4",false_real_4,10,1)
  call check_value(.false.,result_poll_key_and_check_scalar_real_4,&
        TRIM(key_prefix//"poll_key_and_check_scalar_real_4")//"_expect_false")
  result_poll_key_and_check_scalar_real_4 = &
        poll_exact_key_and_check_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_real_4",true_real_4,10,1)
  call check_value(.true.,&
        result_poll_key_and_check_scalar_real_4,&
        key_prefix//"poll_key_and_check_scalar_real_4"//"_expect_true")

  true_real_8 = 3.125
  false_real_8 = 4.125
  call put_exact_key_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_real_8",true_real_8)
  result_poll_key_and_check_scalar_real_8 = &
        poll_exact_key_and_check_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_real_8",false_real_8,10,1)
  call check_value(.false.,result_poll_key_and_check_scalar_real_8,&
        TRIM(key_prefix//"poll_key_and_check_scalar_real_8")//"_expect_false")
  result_poll_key_and_check_scalar_real_8 = &
        poll_exact_key_and_check_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_real_8",true_real_8,10,1)
  call check_value(.true.,&
        result_poll_key_and_check_scalar_real_8,&
        key_prefix//"poll_key_and_check_scalar_real_8"//"_expect_true")

  true_integer_4 = 3
  false_integer_4 = 4
  call put_exact_key_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_integer_4",true_integer_4)
  result_poll_key_and_check_scalar_integer_4 = &
        poll_exact_key_and_check_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_integer_4",false_integer_4,10,1)
  call check_value(.false.,result_poll_key_and_check_scalar_integer_4,&
        TRIM(key_prefix//"poll_key_and_check_scalar_integer_4")//"_expect_false")
  result_poll_key_and_check_scalar_integer_4 = &
        poll_exact_key_and_check_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_integer_4",true_integer_4,10,1)
  call check_value(.true.,&
        result_poll_key_and_check_scalar_integer_4,&
        key_prefix//"poll_key_and_check_scalar_integer_4"//"_expect_true")

  true_integer_8 = 3
  false_integer_8 = 4
  call put_exact_key_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_integer_8",true_integer_8)
  result_poll_key_and_check_scalar_integer_8 = &
        poll_exact_key_and_check_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_integer_8",false_integer_8,10,1)
  call check_value(.false.,result_poll_key_and_check_scalar_integer_8,&
        TRIM(key_prefix//"poll_key_and_check_scalar_integer_8")//"_expect_false")
  result_poll_key_and_check_scalar_integer_8 = &
        poll_exact_key_and_check_scalar(smartsim_client,key_prefix//"poll_key_and_check_scalar_integer_8",true_integer_8,10,1)
  call check_value(.true.,&
        result_poll_key_and_check_scalar_integer_8,&
        key_prefix//"poll_key_and_check_scalar_integer_8"//"_expect_true")

  call MPI_Finalize(err_code)
  if (num_failed > 0) stop -1
end program exact_key_poll_key_and_check_scalar_unit_test
