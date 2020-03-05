module unit_tests

use iso_c_binding,  only : c_ptr
use client_fortran_api, only : put_array, get_array

implicit none; private

public :: test_array

contains

  subroutine test_array( ssc_client, ni, key_prefix )
    type(c_ptr), value           :: ssc_client !< Initialized data client
    integer, intent(in)          :: ni         !< Size of 1d array to make
    character(len=*), intent(in) :: key_prefix !< Prefix to add to the key (usually the PE number)  
    logical             :: error_flag
    integer             :: num_errors

    ! Local variables
    integer :: ni2d
    integer :: nj2d
    real(kind=8), dimension(ni)        :: array_1d_send_dbl, array_1d_recv_dbl
    real(kind=8), dimension(:,:), allocatable :: array_2d_send_dbl, array_2d_recv_dbl
    real, dimension(ni)                :: array_send_flt, array_recv_flt
    integer,      dimension(ni)        :: array_send_int, array_recv_int
    integer :: i,j
    real, parameter :: scale_fact = 100000
  
    !---1-dimensional tests---!
    call random_number(array_1d_send_dbl)
    array_1d_recv_dbl(:) = 0.

    do i = 1,ni
      array_send_flt(i) = real(array_1d_send_dbl(i),4)
      array_send_int(i) = nint(array_1d_send_dbl(i)*scale_fact)
    enddo
  
    write(*,*) "Testing send of 1D array"
    call put_array(ssc_client, trim(key_prefix)//"1d_dbl", array_1d_send_dbl)
    write(*,*) "Testing get of 1D array"
    call get_array(ssc_client, trim(key_prefix)//"1d_dbl", array_1d_recv_dbl)

    error_flag = .false.
    num_errors = 0
    do i = 1,ni
      if (array_1d_send_dbl(i) /= array_1d_recv_dbl(i)) then
        print *, "WRONG!!!!:", array_1d_send_dbl(i), array_1d_recv_dbl(i)
        error_flag = .true.
        num_errors = num_errors+1
      endif
    enddo
    call throw_error(error_flag, "Send/Receive 1d Double: Failed", num_errors )

    !---2-dimensional tests---!
    if (mod(ni,2) /= 0) then
      write(*,*) 'Array size must be a multiple of 2'
      stop -1 
    endif
    ni2d = ni/2
    nj2d = 2
    allocate(array_2d_send_dbl(ni2d,nj2d))
    allocate(array_2d_recv_dbl(ni2d,nj2d))
    call random_number(array_2d_send_dbl)
    
    write(*,*) "Testing send of 2D array"
    call put_array(ssc_client, trim(key_prefix)//"2d_dbl", array_2d_send_dbl)
    write(*,*) "Testing recv of 2D array"
    call get_array(ssc_client, trim(key_prefix)//"2d_dbl", array_2d_recv_dbl)

    do i=1,ni2d; do j=1,nj2d
      if (array_2d_send_dbl(i,j) /= array_2d_recv_dbl(i,j) ) then
        print *, "WRONG!!!!:", array_2d_send_dbl(i,j), array_2d_recv_dbl(i,j)
        error_flag = .true.
        num_errors = num_errors+1
      endif
    enddo; enddo;
    call throw_error(error_flag, "Send/Receive 2d Double: Failed", num_errors )
    
  end subroutine test_array

  subroutine throw_error(error_flag, error_string, num_errors)
    logical          :: error_flag   !< If true, throw an error
    character(len=*) :: error_string !< Error string to write
    integer          :: num_errors   !< Number of erros when testing array

    if (error_flag) then
      write(*,*) error_string, "Number of errors: ", num_errors
      stop -1
    ELSE
      write(*,*) "Success!"
    endif
  end subroutine throw_error
end module unit_tests
