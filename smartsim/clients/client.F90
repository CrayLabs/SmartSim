!> Contains C-bound interfaces for sending data to client
module client_fortran_api

use iso_c_binding, only : c_ptr, c_loc, c_char, c_double, c_int, c_float, c_null_char, c_f_pointer 

implicit none; private

public :: init_ssc_client
public :: put_array, get_array

interface put_array
  module procedure put_array_double
end interface put_array
interface get_array
  module procedure get_array_double
end interface get_array

! Interfaces dealing with the C++ class
interface 
  function init_ssc_client_c() bind(c, name='GetObject') result( ssc_ptr )
    use iso_c_binding, only : c_ptr
    type(c_ptr) :: ssc_ptr  !< Pointer to instanced SmartSim Client
  end function init_ssc_client_c
end interface

interface
  subroutine put_nd_array_double_c( ssc_obj, key, array_ptr, dims, ndims) bind(c, name="put_nd_array_double_ssc")
    use iso_c_binding, only : c_ptr, c_char, c_int
    type(c_ptr), value                :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)            :: key(*)    !< The key used in the database
    type(c_ptr),value                 :: array_ptr !< Pointer to the array to be sent
    integer(kind=c_int), dimension(:) :: dims      !< Array containing the shape of the array
    integer(kind=c_int)               :: ndims     !< Number of dimensions in array
  end subroutine put_nd_array_double_c
end interface
interface
  subroutine get_nd_array_double_c( ssc_obj, key, array_ptr, dims, ndims) bind(c, name="get_nd_array_double_ssc")
    use iso_c_binding, only : c_ptr, c_char, c_int
    type(c_ptr), value                :: ssc_obj   !< Pointer to initialized SSC instance
    character(kind=c_char)            :: key(*)    !< The key used in the database
    type(c_ptr),value                 :: array_ptr !< Pointer to the array to be sent
    integer(kind=c_int), dimension(:) :: dims      !< Array containing the shape of the array
    integer(kind=c_int)               :: ndims     !< Number of dimensions in array
  end subroutine get_nd_array_double_c
end interface

contains

function init_ssc_client() result(ssc_obj)
  type(c_ptr) :: ssc_obj
  ssc_obj = init_ssc_client_c()
end function init_ssc_client

subroutine put_array_double(ssc_obj, key, array)
  type(c_ptr), value          :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)            :: key     !< The key used in the database
  real(kind=8), dimension(..), target :: array   !< Data to be sent
 
  character(kind=c_char) :: c_key(len(trim(key))+1)
  type(c_ptr) :: array_ptr
  
  c_key = make_c_string(key) 
  array_ptr = c_loc(array)

  call put_nd_array_double_c( ssc_obj, key, array_ptr, shape(array), size(shape(array))) 
end subroutine put_array_double

subroutine get_array_double(ssc_obj, key, array)
  type(c_ptr), value               :: ssc_obj !< Pointer to initialized SSC instance
  character(len=*)                 :: key     !< The key used in the database
  real(kind=8), dimension(..), target      :: array   !< Fortran pointer to the retrived array

  character(kind=c_char) :: c_key(len(trim(key))+1)
  type(c_ptr) :: array_ptr

  c_key = make_c_string(key) 
  array_ptr = c_loc(array)
  call get_nd_array_double_c(ssc_obj, key, array_ptr, shape(array), size(shape(array)))

end subroutine get_array_double

!> Make a c-style string that terminates the end of the string with the c null character
function make_c_string( f_string ) result(c_string)
  character(len=*) :: f_string !< Fortran string to be converted
  
  character(kind=c_char) :: c_string
  c_string = transfer(trim(f_string)//c_null_char,c_string)

end function make_c_string

end module client_fortran_api 
