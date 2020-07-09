! This code was automatically generated

interface
  subroutine put_array_int32_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="put_array_int32_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine put_array_int32_ssc
end interface
interface
  subroutine put_array_int64_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="put_array_int64_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine put_array_int64_ssc
end interface
interface
  subroutine put_array_float_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="put_array_float_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine put_array_float_ssc
end interface
interface
  subroutine put_array_double_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="put_array_double_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine put_array_double_ssc
end interface
interface
  logical(kind=c_bool)function poll_key_and_check_scalar_int32_ssc( ssc_obj, c_key, check_value, poll_frequency, num_tries ) &
        bind(c, name="poll_key_and_check_scalar_int32_c") &
        result(success)
    use iso_c_binding, only : c_int, c_bool, c_ptr, c_char, c_int32_t
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    integer(kind=c_int32_t), value,              intent(in   ) :: check_value       !< Value against which the key will be compared
    integer(kind=c_int), value,                  intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer(kind=c_int), value,                  intent(in   ) :: num_tries         !< How many times to query the database before failing
  end function poll_key_and_check_scalar_int32_ssc
end interface
interface
  logical(kind=c_bool)function poll_key_and_check_scalar_int64_ssc( ssc_obj, c_key, check_value, poll_frequency, num_tries ) &
        bind(c, name="poll_key_and_check_scalar_int64_c") &
        result(success)
    use iso_c_binding, only : c_int, c_bool, c_ptr, c_int64_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    integer(kind=c_int64_t), value,              intent(in   ) :: check_value       !< Value against which the key will be compared
    integer(kind=c_int), value,                  intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer(kind=c_int), value,                  intent(in   ) :: num_tries         !< How many times to query the database before failing
  end function poll_key_and_check_scalar_int64_ssc
end interface
interface
  logical(kind=c_bool)function poll_key_and_check_scalar_float_ssc( ssc_obj, c_key, check_value, poll_frequency, num_tries ) &
        bind(c, name="poll_key_and_check_scalar_float_c") &
        result(success)
    use iso_c_binding, only : c_int, c_float, c_bool, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    real(kind=c_float), value,                   intent(in   ) :: check_value       !< Value against which the key will be compared
    integer(kind=c_int), value,                  intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer(kind=c_int), value,                  intent(in   ) :: num_tries         !< How many times to query the database before failing
  end function poll_key_and_check_scalar_float_ssc
end interface
interface
  logical(kind=c_bool)function poll_key_and_check_scalar_double_ssc( ssc_obj, c_key, check_value, poll_frequency, num_tries ) &
        bind(c, name="poll_key_and_check_scalar_double_c") &
        result(success)
    use iso_c_binding, only : c_int, c_bool, c_ptr, c_char, c_double
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    real(kind=c_double), value,                  intent(in   ) :: check_value       !< Value against which the key will be compared
    integer(kind=c_int), value,                  intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer(kind=c_int), value,                  intent(in   ) :: num_tries         !< How many times to query the database before failing
  end function poll_key_and_check_scalar_double_ssc
end interface
interface
  subroutine get_array_int32_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="get_array_int32_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine get_array_int32_ssc
end interface
interface
  subroutine get_array_int64_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="get_array_int64_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine get_array_int64_ssc
end interface
interface
  subroutine get_array_float_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="get_array_float_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine get_array_float_ssc
end interface
interface
  subroutine get_array_double_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="get_array_double_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine get_array_double_ssc
end interface
interface
  subroutine put_scalar_int32_ssc( ssc_obj, c_key, scalar )&
        bind(c, name="put_scalar_int32_c")
    use iso_c_binding, only : c_ptr, c_int32_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    integer(kind=c_int32_t), value,              intent(in   ) :: scalar            !< Scalar value to send to database
  end subroutine put_scalar_int32_ssc
end interface
interface
  subroutine put_scalar_int64_ssc( ssc_obj, c_key, scalar )&
        bind(c, name="put_scalar_int64_c")
    use iso_c_binding, only : c_ptr, c_int64_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    integer(kind=c_int64_t), value,              intent(in   ) :: scalar            !< Scalar value to send to database
  end subroutine put_scalar_int64_ssc
end interface
interface
  subroutine put_scalar_float_ssc( ssc_obj, c_key, scalar )&
        bind(c, name="put_scalar_float_c")
    use iso_c_binding, only : c_ptr, c_float, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    real(kind=c_float), value,                   intent(in   ) :: scalar            !< Scalar value to send to database
  end subroutine put_scalar_float_ssc
end interface
interface
  subroutine put_scalar_double_ssc( ssc_obj, c_key, scalar )&
        bind(c, name="put_scalar_double_c")
    use iso_c_binding, only : c_ptr, c_double, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    real(kind=c_double), value,                  intent(in   ) :: scalar            !< Scalar value to send to database
  end subroutine put_scalar_double_ssc
end interface
interface
  integer(kind=c_int32_t)function get_scalar_int32_ssc( ssc_obj, c_key ) &
        bind(c, name="get_scalar_int32_c")
    use iso_c_binding, only : c_ptr, c_int32_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
  end function get_scalar_int32_ssc
end interface
interface
  integer(kind=c_int64_t)function get_scalar_int64_ssc( ssc_obj, c_key ) &
        bind(c, name="get_scalar_int64_c")
    use iso_c_binding, only : c_ptr, c_int64_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
  end function get_scalar_int64_ssc
end interface
interface
  real(kind=c_float)function get_scalar_float_ssc( ssc_obj, c_key ) &
        bind(c, name="get_scalar_float_c")
    use iso_c_binding, only : c_ptr, c_float, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
  end function get_scalar_float_ssc
end interface
interface
  real(kind=c_double)function get_scalar_double_ssc( ssc_obj, c_key ) &
        bind(c, name="get_scalar_double_c")
    use iso_c_binding, only : c_ptr, c_double, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
  end function get_scalar_double_ssc
end interface
interface
  subroutine put_exact_key_array_int32_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="put_exact_key_array_int32_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine put_exact_key_array_int32_ssc
end interface
interface
  subroutine put_exact_key_array_int64_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="put_exact_key_array_int64_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine put_exact_key_array_int64_ssc
end interface
interface
  subroutine put_exact_key_array_float_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="put_exact_key_array_float_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine put_exact_key_array_float_ssc
end interface
interface
  subroutine put_exact_key_array_double_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="put_exact_key_array_double_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine put_exact_key_array_double_ssc
end interface
interface
  logical(kind=c_bool)function poll_exact_key_and_check_scalar_int32_ssc( ssc_obj, c_key, check_value, poll_frequency, num_tries ) &
        bind(c, name="poll_exact_key_and_check_scalar_int32_c") &
        result(success)
    use iso_c_binding, only : c_int, c_bool, c_ptr, c_char, c_int32_t
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    integer(kind=c_int32_t), value,              intent(in   ) :: check_value       !< Value against which the key will be compared
    integer(kind=c_int), value,                  intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer(kind=c_int), value,                  intent(in   ) :: num_tries         !< How many times to query the database before failing
  end function poll_exact_key_and_check_scalar_int32_ssc
end interface
interface
  logical(kind=c_bool)function poll_exact_key_and_check_scalar_int64_ssc( ssc_obj, c_key, check_value, poll_frequency, num_tries ) &
        bind(c, name="poll_exact_key_and_check_scalar_int64_c") &
        result(success)
    use iso_c_binding, only : c_int, c_bool, c_ptr, c_int64_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    integer(kind=c_int64_t), value,              intent(in   ) :: check_value       !< Value against which the key will be compared
    integer(kind=c_int), value,                  intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer(kind=c_int), value,                  intent(in   ) :: num_tries         !< How many times to query the database before failing
  end function poll_exact_key_and_check_scalar_int64_ssc
end interface
interface
  logical(kind=c_bool)function poll_exact_key_and_check_scalar_float_ssc( ssc_obj, c_key, check_value, poll_frequency, num_tries ) &
        bind(c, name="poll_exact_key_and_check_scalar_float_c") &
        result(success)
    use iso_c_binding, only : c_int, c_float, c_bool, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    real(kind=c_float), value,                   intent(in   ) :: check_value       !< Value against which the key will be compared
    integer(kind=c_int), value,                  intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer(kind=c_int), value,                  intent(in   ) :: num_tries         !< How many times to query the database before failing
  end function poll_exact_key_and_check_scalar_float_ssc
end interface
interface
  logical(kind=c_bool)function poll_exact_key_and_check_scalar_double_ssc( ssc_obj, c_key, check_value, poll_frequency, num_tries ) &
        bind(c, name="poll_exact_key_and_check_scalar_double_c") &
        result(success)
    use iso_c_binding, only : c_int, c_bool, c_ptr, c_char, c_double
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    real(kind=c_double), value,                  intent(in   ) :: check_value       !< Value against which the key will be compared
    integer(kind=c_int), value,                  intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer(kind=c_int), value,                  intent(in   ) :: num_tries         !< How many times to query the database before failing
  end function poll_exact_key_and_check_scalar_double_ssc
end interface
interface
  subroutine get_exact_key_array_int32_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="get_exact_key_array_int32_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine get_exact_key_array_int32_ssc
end interface
interface
  subroutine get_exact_key_array_int64_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="get_exact_key_array_int64_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine get_exact_key_array_int64_ssc
end interface
interface
  subroutine get_exact_key_array_float_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="get_exact_key_array_float_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine get_exact_key_array_float_ssc
end interface
interface
  subroutine get_exact_key_array_double_ssc( ssc_obj, c_key, array_ptr, dims, ndims )&
        bind(c, name="get_exact_key_array_double_c")
    use iso_c_binding, only : c_int, c_ptr, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    type(c_ptr), value,                          intent(in   ) :: array_ptr         !< Pointer to the array
    integer(kind=c_int), dimension(:),           intent(in   ) :: dims              !< Length along each array dimensions
    integer(kind=c_int),                         intent(in   ) :: ndims             !< Number of dimensions in array
  end subroutine get_exact_key_array_double_ssc
end interface
interface
  subroutine put_exact_key_scalar_int32_ssc( ssc_obj, c_key, scalar )&
        bind(c, name="put_exact_key_scalar_int32_c")
    use iso_c_binding, only : c_ptr, c_int32_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    integer(kind=c_int32_t), value,              intent(in   ) :: scalar            !< Scalar value to send to database
  end subroutine put_exact_key_scalar_int32_ssc
end interface
interface
  subroutine put_exact_key_scalar_int64_ssc( ssc_obj, c_key, scalar )&
        bind(c, name="put_exact_key_scalar_int64_c")
    use iso_c_binding, only : c_ptr, c_int64_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    integer(kind=c_int64_t), value,              intent(in   ) :: scalar            !< Scalar value to send to database
  end subroutine put_exact_key_scalar_int64_ssc
end interface
interface
  subroutine put_exact_key_scalar_float_ssc( ssc_obj, c_key, scalar )&
        bind(c, name="put_exact_key_scalar_float_c")
    use iso_c_binding, only : c_ptr, c_float, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    real(kind=c_float), value,                   intent(in   ) :: scalar            !< Scalar value to send to database
  end subroutine put_exact_key_scalar_float_ssc
end interface
interface
  subroutine put_exact_key_scalar_double_ssc( ssc_obj, c_key, scalar )&
        bind(c, name="put_exact_key_scalar_double_c")
    use iso_c_binding, only : c_ptr, c_double, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
    real(kind=c_double), value,                  intent(in   ) :: scalar            !< Scalar value to send to database
  end subroutine put_exact_key_scalar_double_ssc
end interface
interface
  integer(kind=c_int32_t)function get_exact_key_scalar_int32_ssc( ssc_obj, c_key ) &
        bind(c, name="get_exact_key_scalar_int32_c")
    use iso_c_binding, only : c_ptr, c_int32_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
  end function get_exact_key_scalar_int32_ssc
end interface
interface
  integer(kind=c_int64_t)function get_exact_key_scalar_int64_ssc( ssc_obj, c_key ) &
        bind(c, name="get_exact_key_scalar_int64_c")
    use iso_c_binding, only : c_ptr, c_int64_t, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
  end function get_exact_key_scalar_int64_ssc
end interface
interface
  real(kind=c_float)function get_exact_key_scalar_float_ssc( ssc_obj, c_key ) &
        bind(c, name="get_exact_key_scalar_float_c")
    use iso_c_binding, only : c_ptr, c_float, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
  end function get_exact_key_scalar_float_ssc
end interface
interface
  real(kind=c_double)function get_exact_key_scalar_double_ssc( ssc_obj, c_key ) &
        bind(c, name="get_exact_key_scalar_double_c")
    use iso_c_binding, only : c_ptr, c_double, c_char
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(kind=c_char),                      intent(in   ) :: c_key(*)          !< Key used in the database for the object
  end function get_exact_key_scalar_double_ssc
end interface
! End of generated code

