!> Contains C-bound interfaces for sending data to client
module client_fortran_api

use iso_c_binding, only : c_ptr, c_loc, c_char, c_double, c_int, c_float, c_null_char, c_f_pointer
use iso_c_binding, only : c_int32_t, c_int64_t, c_bool

implicit none; private

public :: init_ssc_client

! Include the public declaration and procedure definitions from the generated code
#include "fortran_header.inc"

! Module level parameter values for some hardcoded quantities
integer, parameter :: MAX_RANK = 31

!---C-interfaces for C++ class---!
interface
  function init_ssc_client_c( cluster ) bind(c, name='initialize_fortran_client') result( ssc_ptr )
    use iso_c_binding, only : c_ptr, c_bool
    logical(kind=c_bool), value :: cluster !< True if database is a distributed across multiple nodes
    type(c_ptr) :: ssc_ptr  !< Pointer to instanced SmartSim Client
  end function init_ssc_client_c
end interface

! Include all explicitly defined bindings to C functions
#include "fortran_interface.inc"

contains
 !> Return the pointer to an initialized SmartSim client
 function init_ssc_client( cluster ) result(ssc_obj)
   logical :: cluster !< True if database is a distributed across multiple nodes
   type(c_ptr) :: ssc_obj
   logical(kind=c_bool) :: cluster_as_bool
   cluster_as_bool = cluster
   ssc_obj = init_ssc_client_c( cluster_as_bool )
 end function init_ssc_client

!---Local utility functions---!
!> Make a c-style string that terminates the end of the string with the c null character
function make_c_string( f_string ) result(c_string)
  character(len=*) :: f_string                             !< Fortran string to be converted
  character(kind=c_char) :: c_string(len_trim(f_string)+1) !< Output the C-string

  c_string = transfer(trim(f_string)//c_null_char,c_string)

end function make_c_string

! Define the native Fortran functions
#include "fortran_routines.inc"

end module client_fortran_api
