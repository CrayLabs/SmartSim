module unit_test_aux

implicit none; private

integer :: num_failed = 0
integer(kind=4), parameter :: integer4_0 = 0
integer(kind=8), parameter :: integer8_0 = 0
real(kind=4), parameter :: real4_0 = 0.
real(kind=8), parameter :: real8_0 = 0.

interface check_value
  module procedure check_value_real_4, check_value_real_8, check_value_integer_4, &
                   check_value_integer_8, check_value_logical
end interface check_value

public :: check_value, integer4_0, integer8_0, real4_0, real8_0, num_failed

contains

  !! Check whether the result and calculated value match and log the result
  subroutine check_value_real_4(expected_value, actual_value, test_name)
    real(kind=4)     :: expected_value
    real(kind=4)     :: actual_value
    character(len=*) :: test_name

    if (expected_value /= actual_value) then
      write(*,*) test_name, " FAILED"
      num_failed = num_failed + 1
    endif
  end subroutine check_value_real_4
  subroutine check_value_real_8(expected_value, actual_value, test_name)
    real(kind=8)     :: expected_value
    real(kind=8)     :: actual_value
    character(len=*) :: test_name

    if (expected_value /= actual_value) then
      write(*,*) test_name, " FAILED"
      num_failed = num_failed + 1
    endif
  end subroutine check_value_real_8
  subroutine check_value_integer_4(expected_value, actual_value, test_name)
    integer(kind=4)     :: expected_value
    integer(kind=4)     :: actual_value
    character(len=*)    :: test_name
    if (expected_value /= actual_value) then
      write(*,*) test_name, " FAILED"
      num_failed = num_failed + 1
    endif
  end subroutine check_value_integer_4
  subroutine check_value_integer_8(expected_value, actual_value, test_name)
    integer(kind=8)     :: expected_value
    integer(kind=8)     :: actual_value
    character(len=*)    :: test_name

    if (expected_value /= actual_value) then
      write(*,*) test_name, " FAILED"
      num_failed = num_failed + 1
    endif
  end subroutine check_value_integer_8
  subroutine check_value_logical(expected_value, actual_value, test_name)
    logical :: expected_value
    logical :: actual_value
    character(len=*)    :: test_name

    if (expected_value .neqv. actual_value) then
      write(*,*) test_name, " FAILED"
      num_failed = num_failed + 1
    endif
  end subroutine check_value_logical
end module unit_test_aux
