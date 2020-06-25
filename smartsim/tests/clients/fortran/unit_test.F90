module unit_tests

use mpi
use iso_c_binding,  only : c_ptr
use client_fortran_api, only : put_array, get_array, put_scalar, get_scalar_int64
use client_fortran_api, only : poll_key_and_check_scalar

implicit none; private

integer, parameter :: POLL_FREQ = 50
integer, parameter :: NUM_TRIES = 2
integer(kind=8), parameter :: FALSE_INT64 = -123
integer(kind=8), parameter :: TRUE_INT64  = 234

public :: test_array

contains

  subroutine test_array( ssc_client, ni, key_prefix, timing_unit )
    type(c_ptr), value            :: ssc_client  !< Initialized data client
    integer(kind=8),   intent(in) :: ni          !< Size of 1d array to make
    character(len=*),  intent(in) :: key_prefix  !< Prefix to add to the key (usually the PE number)
    integer, optional, intent(in) :: timing_unit !< If present, record timings
    ! Local variables
    integer :: ni2d
    integer :: ierror
    integer :: nj2d
    integer :: num_errors
    logical :: error_flag
    logical :: poll_status

    real(kind=8), dimension(ni)               :: array_1d_send_dbl, array_1d_recv_dbl
    real(kind=8), dimension(:,:), allocatable :: array_2d_send_dbl, array_2d_recv_dbl
    integer(kind=8), dimension(ni)            :: array_1d_send_int64, array_1d_recv_int64
    integer(kind=8) :: send_int64, recv_int64
    integer :: i,j
    real, parameter :: scale_fact = 100000.
    real(kind=8)    :: time_start, time_end

    ! Initialize error detection
    error_flag = .false.
    num_errors = 0

    !---Scalar tests---!
    send_int64 = TRUE_INT64; recv_int64 = FALSE_INT64
    call put_scalar(ssc_client, trim(key_prefix)//"_int64", send_int64)
    recv_int64 = get_scalar_int64(ssc_client, trim(key_prefix)//"_int64")
    if (send_int64 /= recv_int64) then
      print *, "WRONG!!!!:", send_int64,' ', recv_int64
      error_flag = .true.
      num_errors = num_errors + 1
    endif
    call throw_error(error_flag, "Send/Receive scalar integer : Failed", num_errors )
    write(*,*) "Send/receive scalar int64: Success"

    !---1-dimensional tests---!
    !----Double Precision----!
    call random_number(array_1d_send_dbl)
    array_1d_recv_dbl(:) = 0.

    if (present(timing_unit))  time_start = start_timing()
    call put_array(ssc_client, trim(key_prefix)//"1d_dbl", array_1d_send_dbl)
    if (present(timing_unit)) call log_timing(timing_unit, key_prefix, "1d_dbl_send", time_start, ni)
    call mpi_barrier(MPI_COMM_WORLD, ierror)

    if (present(timing_unit))  time_start = start_timing()
    call get_array(ssc_client, trim(key_prefix)//"1d_dbl", array_1d_recv_dbl)
    if (present(timing_unit)) call log_timing(timing_unit, key_prefix, "1d_dbl_recv", time_start, ni)
    call mpi_barrier(MPI_COMM_WORLD, ierror)

    do i = 1,ni
      if (array_1d_send_dbl(i) /= array_1d_recv_dbl(i)) then
        print *, "WRONG!!!!:", array_1d_send_dbl(i), array_1d_recv_dbl(i)
        error_flag = .true.
        num_errors = num_errors+1
      endif
    enddo
    call throw_error(error_flag, "Send/Receive 1d Double: Failed", num_errors )
    write(*,*) "Send/receive 1D double array: Success"

    !----8 byte integers----!
    do i=1,ni; array_1d_send_int64(i) = i*2; enddo
    if (present(timing_unit))  time_start = start_timing()
    call put_array(ssc_client, trim(key_prefix)//"1d_int64", array_1d_send_int64)
    if (present(timing_unit)) call log_timing(timing_unit, key_prefix, "1d_int64_send", time_start, ni)
    call mpi_barrier(MPI_COMM_WORLD, ierror)

    if (present(timing_unit))  time_start = start_timing()
    call get_array(ssc_client, trim(key_prefix)//"1d_int64", array_1d_recv_int64)
    if (present(timing_unit)) call log_timing(timing_unit, key_prefix, "1d_int64_recv", time_start, ni)
    call mpi_barrier(MPI_COMM_WORLD, ierror)

    error_flag = .false.
    num_errors = 0
    do i = 1,ni
      if (array_1d_send_int64(i) /= array_1d_recv_int64(i)) then
        print *, "WRONG!!!!:", array_1d_send_int64(i), array_1d_recv_int64(i)
        error_flag = .true.
        num_errors = num_errors+1
      endif
    enddo
    call throw_error(error_flag, "Send/Receive 1d Int64: Failed", num_errors )
    write(*,*) "Send/receive 1D int64 array: Success"

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

    if (present(timing_unit))  time_start = start_timing()
    call put_array(ssc_client, trim(key_prefix)//"2d_dbl", array_2d_send_dbl)
    if (present(timing_unit)) call log_timing(timing_unit, key_prefix, "2d_dbl_send", time_start, ni)
    call mpi_barrier(MPI_COMM_WORLD, ierror)

    if (present(timing_unit))  time_start = start_timing()
    call get_array(ssc_client, trim(key_prefix)//"2d_dbl", array_2d_recv_dbl)
    if (present(timing_unit)) call log_timing(timing_unit, key_prefix, "2d_dbl_recv", time_start, ni)
    call mpi_barrier(MPI_COMM_WORLD, ierror)

    do i=1,ni2d; do j=1,nj2d
      if (array_2d_send_dbl(i,j) /= array_2d_recv_dbl(i,j) ) then
        print *, "WRONG!!!!:", array_2d_send_dbl(i,j), array_2d_recv_dbl(i,j)
        error_flag = .true.
        num_errors = num_errors+1
      endif
    enddo; enddo;
    call throw_error(error_flag, "Send/Receive 2d Double: Failed", num_errors )
    write(*,*) "Send/receive 2D double array: Success"

   !---Database-interaction tests---!
   call put_scalar(ssc_client, trim(key_prefix)//"_int64", TRUE_INT64)
   ! Make sure that the poll fails when matching the false value
   poll_status = poll_key_and_check_scalar(ssc_client, trim(key_prefix)//"_int64", FALSE_INT64, POLL_FREQ, NUM_TRIES )
   if (poll_status) then
     num_errors = num_errors + 1; error_flag = .true.
     call throw_error(error_flag, "Polling failed. Returned true with wrong value", num_errors)
   endif
   write(*,*) "Polling correctly failed with the 'wrong' value"

   poll_status = poll_key_and_check_scalar(ssc_client, trim(key_prefix)//"_int64", TRUE_INT64, POLL_FREQ, NUM_TRIES )
   if (.not. poll_status) then
     num_errors = num_errors + 1; error_flag = .true.
     call throw_error(error_flag, "Polling failed: Returned false with true value", num_errors)
   endif
   write(*,*) "Polling succceeded with the correct value"

  end subroutine test_array

  real(kind=8) function start_timing()
    integer      :: count_start, count_max
    real(kind=8) :: count_rate
    call system_clock(count_start, count_rate, count_max)
    start_timing = count_start/count_rate
  end function start_timing

  subroutine log_timing(timing_unit, key_prefix, test_name, time_start, ni)
    integer,          intent(in) :: timing_unit
    character(len=*), intent(in) :: key_prefix
    character(len=*), intent(in) :: test_name
    real(kind=8),     intent(in) :: time_start
    integer(kind=8),  intent(in) :: ni

    integer      :: count_end, count_max
    real(kind=8) :: count_rate, time_end

    call system_clock(count_end, count_rate, count_max)
    time_end = count_end/count_rate
    write(timing_unit,'(A,X,A,E23.15E3,X,I0)') trim(key_prefix), trim(test_name), time_end-time_start, ni

  end subroutine log_timing

  subroutine throw_error(error_flag, error_string, num_errors)
    logical          :: error_flag   !< If true, throw an error
    character(len=*) :: error_string !< Error string to write
    integer          :: num_errors   !< Number of erros when testing array

    if (error_flag) then
      write(*,*) error_string, "Number of errors: ", num_errors
      stop -1
    ELSE
      !write(*,*) "Success!"
    endif
  end subroutine throw_error
end module unit_tests
