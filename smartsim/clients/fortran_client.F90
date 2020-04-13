!> Contains C-bound interfaces for sending data to client
module client_fortran_api

use iso_c_binding, only : c_ptr, c_loc, c_char, c_double, c_int, c_float, c_null_char, c_f_pointer
use iso_c_binding, only : c_int32_t, c_int64_t

implicit none; private

public :: init_ssc_client
public :: put_array, get_array, put_scalar
public :: get_scalar_int32, get_scalar_int64
public :: poll_key_and_check_scalar

! Module level parameter values for some hardcoded quantities
integer, parameter :: MAX_RANK = 31

! Overloaded functions for ease of use on the simulation side

!> Put a Fortran array (of any rank) to the database with a given key. Currently supported types are:
!! REAL*8 and INTEGER*8
interface put_array
  module procedure put_array_double, put_array_int64, put_array_int32
end interface put_array
!> Get a Fortran array (of any rank) from the database with a given key. Currently supported types are:
!! REAL*8 and INTEGER*8
interface get_array
  module procedure get_array_double, get_array_int64, get_array_int32
end interface get_array
!> Put a single scalar value into the database with a given key: Currently supported types: INTEGER*8
interface put_scalar
  module procedure put_scalar_int64, put_scalar_int32
end interface put_scalar
!> Get a single scalar value from the database with a given key: Currently supported types: INTEGER*8
!> Check the database for both the existence of a key and that it matches the given value
!! Currently supported types: INTEGER*8
interface poll_key_and_check_scalar
  module procedure poll_key_and_check_scalar_int64, poll_key_and_check_scalar_int32
end interface poll_key_and_check_scalar

!---C-interfaces for C++ class---!
interface
  function init_ssc_client_c() bind(c, name='GetObject') result( ssc_ptr )
    use iso_c_binding, only : c_ptr
    type(c_ptr) :: ssc_ptr  !< Pointer to instanced SmartSim Client
  end function init_ssc_client_c
end interface

!---C-interfaces for arrays---!
!----Double Precision----!
interface
  subroutine put_array_double_ssc( ssc_obj, key, array_ptr, dims, ndims) bind(c, name="put_array_double_c")
    use iso_c_binding, only : c_ptr, c_char, c_int
    type(c_ptr), value                :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)            :: key(*)    !< The key used in the database
    type(c_ptr),value                 :: array_ptr !< Pointer to the array to be sent
    integer(kind=c_int), dimension(:) :: dims      !< Array containing the shape of the array
    integer(kind=c_int)               :: ndims     !< Number of dimensions in array
  end subroutine put_array_double_ssc
end interface
interface
  subroutine get_array_double_ssc( ssc_obj, key, array_ptr, dims, ndims) bind(c, name="get_array_double_c")
    use iso_c_binding, only : c_ptr, c_char, c_int
    type(c_ptr), value                :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)            :: key(*)    !< The key used in the database
    type(c_ptr),value                 :: array_ptr !< Pointer to the array to be sent
    integer(kind=c_int), dimension(:) :: dims      !< Array containing the shape of the array
    integer(kind=c_int)               :: ndims     !< Number of dimensions in array
  end subroutine get_array_double_ssc
end interface
!----8 byte integers----!
interface
  subroutine put_array_int64_ssc( ssc_obj, key, array_ptr, dims, ndims) bind(c, name="put_array_int64_c")
    use iso_c_binding, only : c_ptr, c_char, c_int
    type(c_ptr), value                :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)            :: key(*)    !< The key used in the database
    type(c_ptr),value                 :: array_ptr !< Pointer to the array to be sent
    integer(kind=c_int), dimension(:) :: dims      !< Array containing the shape of the array
    integer(kind=c_int)               :: ndims     !< Number of dimensions in array
  end subroutine put_array_int64_ssc
end interface
interface
  subroutine get_array_int64_ssc( ssc_obj, key, array_ptr, dims, ndims) bind(c, name="get_array_int64_c")
    use iso_c_binding, only : c_ptr, c_char, c_int
    type(c_ptr), value                :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)            :: key(*)    !< The key used in the database
    type(c_ptr),value                 :: array_ptr !< Pointer to the array to be sent
    integer(kind=c_int), dimension(:) :: dims      !< Array containing the shape of the array
    integer(kind=c_int)               :: ndims     !< Number of dimensions in array
  end subroutine get_array_int64_ssc
end interface
!----4 byte integers----!
interface
  subroutine put_array_int32_ssc( ssc_obj, key, array_ptr, dims, ndims) bind(c, name="put_array_int32_c")
    use iso_c_binding, only : c_ptr, c_char, c_int
    type(c_ptr), value                :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)            :: key(*)    !< The key used in the database
    type(c_ptr),value                 :: array_ptr !< Pointer to the array to be sent
    integer(kind=c_int), dimension(:) :: dims      !< Array containing the shape of the array
    integer(kind=c_int)               :: ndims     !< Number of dimensions in array
  end subroutine put_array_int32_ssc
end interface
interface
  subroutine get_array_int32_ssc( ssc_obj, key, array_ptr, dims, ndims) bind(c, name="get_array_int32_c")
    use iso_c_binding, only : c_ptr, c_char, c_int
    type(c_ptr), value                :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)            :: key(*)    !< The key used in the database
    type(c_ptr),value                 :: array_ptr !< Pointer to the array to be sent
    integer(kind=c_int), dimension(:) :: dims      !< Array containing the shape of the array
    integer(kind=c_int)               :: ndims     !< Number of dimensions in array
  end subroutine get_array_int32_ssc
end interface

!---C-interfaces for scalars---!
!----8 byte integers----!
interface
  subroutine put_scalar_int64_ssc( ssc_obj, key, value ) bind(c, name="put_scalar_int64_c")
    use iso_c_binding, only : c_ptr, c_char, c_int64_t
    type(c_ptr), value             :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)         :: key(*)    !< The key used in the database
    integer(kind=c_int64_t), value :: value     !< Integer value of the key for the database
  end subroutine
end interface
interface
  integer(kind=c_int64_t) function get_scalar_int64_ssc( ssc_obj, key ) bind(c, name="get_scalar_int64_c")
    use iso_c_binding, only : c_ptr, c_char, c_int64_t
    type(c_ptr), value             :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)         :: key(*)    !< The key used in the database
  end function
end interface
!----8 byte integers----!
interface
  subroutine put_scalar_int32_ssc( ssc_obj, key, value ) bind(c, name="put_scalar_int32_c")
    use iso_c_binding, only : c_ptr, c_char, c_int32_t
    type(c_ptr), value             :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)         :: key(*)    !< The key used in the database
    integer(kind=c_int32_t), value :: value     !< Integer value of the key for the database
  end subroutine
end interface
interface
  integer(kind=c_int32_t) function get_scalar_int32_ssc( ssc_obj, key ) bind(c, name="get_scalar_int32_c")
    use iso_c_binding, only : c_ptr, c_char, c_int32_t
    type(c_ptr), value             :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)         :: key(*)    !< The key used in the database
  end function
end interface

!---C-interfaces for interacting with the database---!
interface
  function poll_key_and_check_scalar_int64_ssc( ssc_obj, key, value, poll_frequency, num_tries ) &
                         bind(c, name="poll_key_and_check_scalar_int64_c") result(success)
    use iso_c_binding, only : c_ptr, c_char, c_int64_t, c_int, c_bool
    type(c_ptr), value             :: ssc_obj        !< Pointer to initialized SSC instance
    character(kind=c_char)         :: key(*)         !< The key used in the database
    integer(kind=c_int64_t), value :: value          !< Integer value of the key to check in the database
    integer(kind=c_int),     value :: poll_frequency !< How frequently to poll the database (in ms)
    integer(kind=c_int),     value :: num_tries      !< Number of tries before returning false
    logical(kind=c_bool)           :: success        !< If true, the key was found and matched the value
  end function
end interface
interface
  function poll_key_and_check_scalar_int32_ssc( ssc_obj, key, value, poll_frequency, num_tries ) &
                         bind(c, name="poll_key_and_check_scalar_int32_c") result(success)
    use iso_c_binding, only : c_ptr, c_char, c_int32_t, c_int, c_bool
    type(c_ptr), value             :: ssc_obj        !< Pointer to initialized SSC instance
    character(kind=c_char)         :: key(*)         !< The key used in the database
    integer(kind=c_int32_t), value :: value          !< Integer value of the key to check in the database
    integer(kind=c_int),     value :: poll_frequency !< How frequently to poll the database (in ms)
    integer(kind=c_int),     value :: num_tries      !< Number of tries before returning false
    logical(kind=c_bool)           :: success        !< If true, the key was found and matched the value
  end function
end interface

contains

function init_ssc_client() result(ssc_obj)
  type(c_ptr) :: ssc_obj
  ssc_obj = init_ssc_client_c()
end function init_ssc_client

!---Routines for arrays---!
!----Double Precision----!
subroutine put_array_double(ssc_obj, key, array)
  type(c_ptr), value                  :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                    :: key     !< The key used in the database
  real(kind=8), dimension(..), target :: array   !< Data to be sent

  character(kind=c_char) :: c_key(len(trim(key))+1)
  type(c_ptr) :: array_ptr
  integer :: ndims, i
  integer(kind=c_int), dimension(MAX_RANK) :: rev_dims

  ! Store the shape of the arrays in reverse order
  ndims = size(shape(array))
  do i=1,ndims
    rev_dims(i) = size(array,ndims+1-i)
  enddo

  c_key = make_c_string(key)
  array_ptr = c_loc(array)
  call put_array_double_ssc( ssc_obj, c_key, array_ptr, rev_dims(1:ndims), ndims)
end subroutine put_array_double

subroutine get_array_double(ssc_obj, key, array)
  type(c_ptr), value                  :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                    :: key     !< The key used in the database
  real(kind=8), dimension(..), target :: array   !< Fortran pointer to the retrived array

  character(kind=c_char) :: c_key(len(trim(key))+1)
  type(c_ptr) :: array_ptr
  integer :: ndims, i
  integer(kind=c_int), dimension(MAX_RANK) :: rev_dims

  ! Store the shape of the arrays in reverse order
  ndims = size(shape(array))
  do i=1,ndims
    rev_dims(i) = size(array,ndims+1-i)
  enddo

  c_key = make_c_string(key)
  array_ptr = c_loc(array)
  call get_array_double_ssc(ssc_obj, c_key, array_ptr, rev_dims(1:ndims), ndims)
end subroutine get_array_double

!----Integer 8-byte----!
subroutine put_array_int64(ssc_obj, key, array)
  type(c_ptr), value                             :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                               :: key     !< The key used in the database
  integer(kind=c_int64_t), dimension(..), target :: array   !< Data to be sent

  character(kind=c_char) :: c_key(len(trim(key))+1)
  type(c_ptr) :: array_ptr
  integer :: ndims, i
  integer(kind=c_int), dimension(MAX_RANK) :: rev_dims

  ! Store the shape of the arrays in reverse order
  ndims = size(shape(array))
  do i=1,ndims
    rev_dims(i) = size(array,ndims+1-i)
  enddo

  c_key = make_c_string(key)
  array_ptr = c_loc(array)
  call put_array_int64_ssc( ssc_obj, c_key, array_ptr, rev_dims(1:ndims), ndims )
end subroutine put_array_int64

subroutine get_array_int64(ssc_obj, key, array)
  type(c_ptr), value                             :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                               :: key     !< The key used in the database
  integer(kind=c_int64_t), dimension(..), target :: array   !< Fortran pointer to the retrived array

  character(kind=c_char) :: c_key(len(trim(key))+1)
  type(c_ptr) :: array_ptr
  integer :: ndims, i
  integer(kind=c_int), dimension(MAX_RANK) :: rev_dims

  ! Store the shape of the arrays in reverse order
  ndims = size(shape(array))
  do i=1,ndims
    rev_dims(i) = size(array,ndims+1-i)
  enddo

  c_key = make_c_string(key)
  array_ptr = c_loc(array)
  call get_array_int64_ssc(ssc_obj, c_key, array_ptr, rev_dims(1:ndims), ndims)
end subroutine get_array_int64

!----Integer 4-byte----!
subroutine put_array_int32(ssc_obj, key, array)
  type(c_ptr), value                             :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                               :: key     !< The key used in the database
  integer(kind=c_int32_t), dimension(..), target :: array   !< Data to be sent

  character(kind=c_char) :: c_key(len(trim(key))+1)
  type(c_ptr) :: array_ptr
  integer :: ndims, i
  integer(kind=c_int), dimension(MAX_RANK) :: rev_dims

  ! Store the shape of the arrays in reverse order
  ndims = size(shape(array))
  do i=1,ndims
    rev_dims(i) = size(array,ndims+1-i)
  enddo

  c_key = make_c_string(key)
  array_ptr = c_loc(array)
  call put_array_int32_ssc( ssc_obj, c_key, array_ptr, rev_dims(1:ndims), ndims )
end subroutine put_array_int32

subroutine get_array_int32(ssc_obj, key, array)
  type(c_ptr), value                             :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                               :: key     !< The key used in the database
  integer(kind=c_int32_t), dimension(..), target :: array   !< Fortran pointer to the retrived array

  character(kind=c_char) :: c_key(len(trim(key))+1)
  type(c_ptr) :: array_ptr
  integer :: ndims, i
  integer(kind=c_int), dimension(MAX_RANK) :: rev_dims

  ! Store the shape of the arrays in reverse order
  ndims = size(shape(array))
  do i=1,ndims
    rev_dims(i) = size(array,ndims+1-i)
  enddo

  c_key = make_c_string(key)
  array_ptr = c_loc(array)
  call get_array_int32_ssc(ssc_obj, c_key, array_ptr, rev_dims(1:ndims), ndims)
end subroutine get_array_int32

!---Routines for scalars---!
!----Integer 8-byte----!
subroutine put_scalar_int64(ssc_obj, key, value)
  type(c_ptr), value                             :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                               :: key     !< The key used in the database
  integer(kind=c_int64_t)                        :: value   !< Data to be sent

  character(kind=c_char) :: c_key(len(trim(key))+1)
  c_key = make_c_string(key)
  call put_scalar_int64_ssc( ssc_obj, c_key, value )
end subroutine put_scalar_int64

integer(kind=8) function get_scalar_int64(ssc_obj, key)
  type(c_ptr), value                             :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                               :: key     !< The key used in the database

  character(kind=c_char) :: c_key(len(trim(key))+1)
  c_key = make_c_string(key)
  get_scalar_int64 = get_scalar_int64_ssc( ssc_obj, c_key )
end function get_scalar_int64

!----Integer 4-byte----!
subroutine put_scalar_int32(ssc_obj, key, value)
  type(c_ptr), value                             :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                               :: key     !< The key used in the database
  integer(kind=c_int32_t)                        :: value   !< Data to be sent

  character(kind=c_char) :: c_key(len(trim(key))+1)
  c_key = make_c_string(key)
  call put_scalar_int32_ssc( ssc_obj, c_key, value )
end subroutine put_scalar_int32

integer(kind=4) function get_scalar_int32(ssc_obj, key)
  type(c_ptr), value                             :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                               :: key     !< The key used in the database

  character(kind=c_char) :: c_key(len(trim(key))+1)
  c_key = make_c_string(key)
  get_scalar_int32 = get_scalar_int32_ssc( ssc_obj, c_key )
end function get_scalar_int32

!---Database related functions---!
function poll_key_and_check_scalar_int64( ssc_obj, key, value, poll_frequency, num_tries ) &
                                          result(success)
  type(c_ptr), value :: ssc_obj        !< Pointer to initialized SSC instance
  character(len=*)   :: key            !< The key used in the database
  integer(kind=8)    :: value          !< Integer value of the key to check in the database
  integer            :: poll_frequency !< How frequently to poll the database (in ms)
  integer            :: num_tries      !< Number of tries before returning false
  logical            :: success        !< Key found and matched the value

  character(kind=c_char) :: c_key(len_trim(key)+1)
  c_key = make_c_string(key)
  success = poll_key_and_check_scalar_int64_ssc( ssc_obj, c_key, value, poll_frequency, num_tries )

end function poll_key_and_check_scalar_int64
function poll_key_and_check_scalar_int32( ssc_obj, key, value, poll_frequency, num_tries ) &
                                          result(success)
  type(c_ptr), value :: ssc_obj        !< Pointer to initialized SSC instance
  character(len=*)   :: key            !< The key used in the database
  integer(kind=4)    :: value          !< Integer value of the key to check in the database
  integer            :: poll_frequency !< How frequently to poll the database (in ms)
  integer            :: num_tries      !< Number of tries before returning false
  logical            :: success        !< Key found and matched the value

  character(kind=c_char) :: c_key(len_trim(key)+1)
  c_key = make_c_string(key)
  success = poll_key_and_check_scalar_int32_ssc( ssc_obj, c_key, value, poll_frequency, num_tries )

end function poll_key_and_check_scalar_int32

!---Local utility functions---!
!> Make a c-style string that terminates the end of the string with the c null character
function make_c_string( f_string ) result(c_string)
  character(len=*) :: f_string                             !< Fortran string to be converted
  character(kind=c_char) :: c_string(len_trim(f_string)+1) !< Output the C-string

  c_string = transfer(trim(f_string)//c_null_char,c_string)

end function make_c_string

end module client_fortran_api
