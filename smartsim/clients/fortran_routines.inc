! This code was automatically generated

!>\file
!>\brief User-facing Fortran wrappers to SmartSim
  !> Store an array of type int32 in the database
  subroutine put_array_int32( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4), dimension(..), target,      intent(in   ) :: array             !< Array to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call put_array_int32_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine put_array_int32

  !> Store an array of type int64 in the database
  subroutine put_array_int64( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8), dimension(..), target,      intent(in   ) :: array             !< Array to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call put_array_int64_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine put_array_int64

  !> Store an array of type float in the database
  subroutine put_array_float( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4), dimension(..), target,         intent(in   ) :: array             !< Array to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call put_array_float_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine put_array_float

  !> Store an array of type double in the database
  subroutine put_array_double( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8), dimension(..), target,         intent(in   ) :: array             !< Array to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call put_array_double_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine put_array_double

  !> Check for existence of a key and whether it matches the requested value of type int32
  function poll_key_and_check_scalar_int32( ssc_obj, key, check_value, poll_frequency, num_tries ) &
        result(success)
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4),                             intent(in   ) :: check_value       !< Value against which the key will be compared
    integer,                                     intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer,                                     intent(in   ) :: num_tries         !< How many times to query the database before failing
    logical                                                    :: success           !< True if the key was found AND matched the requested value
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    success = poll_key_and_check_scalar_int32_ssc(ssc_obj, c_key, check_value, poll_frequency, num_tries)
  end function poll_key_and_check_scalar_int32

  !> Check for existence of a key and whether it matches the requested value of type int64
  function poll_key_and_check_scalar_int64( ssc_obj, key, check_value, poll_frequency, num_tries ) &
        result(success)
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8),                             intent(in   ) :: check_value       !< Value against which the key will be compared
    integer,                                     intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer,                                     intent(in   ) :: num_tries         !< How many times to query the database before failing
    logical                                                    :: success           !< True if the key was found AND matched the requested value
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    success = poll_key_and_check_scalar_int64_ssc(ssc_obj, c_key, check_value, poll_frequency, num_tries)
  end function poll_key_and_check_scalar_int64

  !> Check for existence of a key and whether it matches the requested value of type float
  function poll_key_and_check_scalar_float( ssc_obj, key, check_value, poll_frequency, num_tries ) &
        result(success)
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4),                                intent(in   ) :: check_value       !< Value against which the key will be compared
    integer,                                     intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer,                                     intent(in   ) :: num_tries         !< How many times to query the database before failing
    logical                                                    :: success           !< True if the key was found AND matched the requested value
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    success = poll_key_and_check_scalar_float_ssc(ssc_obj, c_key, check_value, poll_frequency, num_tries)
  end function poll_key_and_check_scalar_float

  !> Check for existence of a key and whether it matches the requested value of type double
  function poll_key_and_check_scalar_double( ssc_obj, key, check_value, poll_frequency, num_tries ) &
        result(success)
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8),                                intent(in   ) :: check_value       !< Value against which the key will be compared
    integer,                                     intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer,                                     intent(in   ) :: num_tries         !< How many times to query the database before failing
    logical                                                    :: success           !< True if the key was found AND matched the requested value
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    success = poll_key_and_check_scalar_double_ssc(ssc_obj, c_key, check_value, poll_frequency, num_tries)
  end function poll_key_and_check_scalar_double

  !> Get an array of type int32 from the database
  subroutine get_array_int32( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4), dimension(..), target,      intent(inout) :: array             !< Array to be received from database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call get_array_int32_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine get_array_int32

  !> Get an array of type int64 from the database
  subroutine get_array_int64( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8), dimension(..), target,      intent(inout) :: array             !< Array to be received from database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call get_array_int64_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine get_array_int64

  !> Get an array of type float from the database
  subroutine get_array_float( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4), dimension(..), target,         intent(inout) :: array             !< Array to be received from database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call get_array_float_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine get_array_float

  !> Get an array of type double from the database
  subroutine get_array_double( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8), dimension(..), target,         intent(inout) :: array             !< Array to be received from database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call get_array_double_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine get_array_double

  !> Put a scalar of type int32 to the database
  subroutine put_scalar_int32( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4),                             intent(in   ) :: scalar            !< Scalar value to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    call put_scalar_int32_ssc(ssc_obj, c_key, scalar)
  end subroutine put_scalar_int32

  !> Put a scalar of type int64 to the database
  subroutine put_scalar_int64( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8),                             intent(in   ) :: scalar            !< Scalar value to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    call put_scalar_int64_ssc(ssc_obj, c_key, scalar)
  end subroutine put_scalar_int64

  !> Put a scalar of type float to the database
  subroutine put_scalar_float( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4),                                intent(in   ) :: scalar            !< Scalar value to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    call put_scalar_float_ssc(ssc_obj, c_key, scalar)
  end subroutine put_scalar_float

  !> Put a scalar of type double to the database
  subroutine put_scalar_double( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8),                                intent(in   ) :: scalar            !< Scalar value to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    call put_scalar_double_ssc(ssc_obj, c_key, scalar)
  end subroutine put_scalar_double

  !> Get a scalar of type int32 from the database
  subroutine get_scalar_int32( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4),                             intent(  out) :: scalar            !< Scalar value to get from the database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    scalar = get_scalar_int32_ssc(ssc_obj, c_key)
  end subroutine get_scalar_int32

  !> Get a scalar of type int64 from the database
  subroutine get_scalar_int64( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8),                             intent(  out) :: scalar            !< Scalar value to get from the database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    scalar = get_scalar_int64_ssc(ssc_obj, c_key)
  end subroutine get_scalar_int64

  !> Get a scalar of type float from the database
  subroutine get_scalar_float( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4),                                intent(  out) :: scalar            !< Scalar value to get from the database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    scalar = get_scalar_float_ssc(ssc_obj, c_key)
  end subroutine get_scalar_float

  !> Get a scalar of type double from the database
  subroutine get_scalar_double( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8),                                intent(  out) :: scalar            !< Scalar value to get from the database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    scalar = get_scalar_double_ssc(ssc_obj, c_key)
  end subroutine get_scalar_double

  !> Store an array of type int32 in the database using exact key
  subroutine put_exact_key_array_int32( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4), dimension(..), target,      intent(in   ) :: array             !< Array to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call put_exact_key_array_int32_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine put_exact_key_array_int32

  !> Store an array of type int64 in the database using exact key
  subroutine put_exact_key_array_int64( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8), dimension(..), target,      intent(in   ) :: array             !< Array to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call put_exact_key_array_int64_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine put_exact_key_array_int64

  !> Store an array of type float in the database using exact key
  subroutine put_exact_key_array_float( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4), dimension(..), target,         intent(in   ) :: array             !< Array to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call put_exact_key_array_float_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine put_exact_key_array_float

  !> Store an array of type double in the database using exact key
  subroutine put_exact_key_array_double( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8), dimension(..), target,         intent(in   ) :: array             !< Array to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call put_exact_key_array_double_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine put_exact_key_array_double

  !> Check for existence of an exact key and whether it matches the requested value of type int32
  function poll_exact_key_and_check_scalar_int32( ssc_obj, key, check_value, poll_frequency, num_tries ) &
        result(success)
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4),                             intent(in   ) :: check_value       !< Value against which the key will be compared
    integer,                                     intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer,                                     intent(in   ) :: num_tries         !< How many times to query the database before failing
    logical                                                    :: success           !< True if the key was found AND matched the requested value
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    success = poll_exact_key_and_check_scalar_int32_ssc(ssc_obj, c_key, check_value, poll_frequency, num_tries)
  end function poll_exact_key_and_check_scalar_int32

  !> Check for existence of an exact key and whether it matches the requested value of type int64
  function poll_exact_key_and_check_scalar_int64( ssc_obj, key, check_value, poll_frequency, num_tries ) &
        result(success)
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8),                             intent(in   ) :: check_value       !< Value against which the key will be compared
    integer,                                     intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer,                                     intent(in   ) :: num_tries         !< How many times to query the database before failing
    logical                                                    :: success           !< True if the key was found AND matched the requested value
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    success = poll_exact_key_and_check_scalar_int64_ssc(ssc_obj, c_key, check_value, poll_frequency, num_tries)
  end function poll_exact_key_and_check_scalar_int64

  !> Check for existence of an exact key and whether it matches the requested value of type float
  function poll_exact_key_and_check_scalar_float( ssc_obj, key, check_value, poll_frequency, num_tries ) &
        result(success)
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4),                                intent(in   ) :: check_value       !< Value against which the key will be compared
    integer,                                     intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer,                                     intent(in   ) :: num_tries         !< How many times to query the database before failing
    logical                                                    :: success           !< True if the key was found AND matched the requested value
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    success = poll_exact_key_and_check_scalar_float_ssc(ssc_obj, c_key, check_value, poll_frequency, num_tries)
  end function poll_exact_key_and_check_scalar_float

  !> Check for existence of an exact key and whether it matches the requested value of type double
  function poll_exact_key_and_check_scalar_double( ssc_obj, key, check_value, poll_frequency, num_tries ) &
        result(success)
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8),                                intent(in   ) :: check_value       !< Value against which the key will be compared
    integer,                                     intent(in   ) :: poll_frequency    !< How often to query the database for the key (ms)
    integer,                                     intent(in   ) :: num_tries         !< How many times to query the database before failing
    logical                                                    :: success           !< True if the key was found AND matched the requested value
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    success = poll_exact_key_and_check_scalar_double_ssc(ssc_obj, c_key, check_value, poll_frequency, num_tries)
  end function poll_exact_key_and_check_scalar_double

  !> Get an array of type int32 from the database using the exact key
  subroutine get_exact_key_array_int32( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4), dimension(..), target,      intent(inout) :: array             !< Array to be received from database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call get_exact_key_array_int32_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine get_exact_key_array_int32

  !> Get an array of type int64 from the database using the exact key
  subroutine get_exact_key_array_int64( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8), dimension(..), target,      intent(inout) :: array             !< Array to be received from database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call get_exact_key_array_int64_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine get_exact_key_array_int64

  !> Get an array of type float from the database using the exact key
  subroutine get_exact_key_array_float( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4), dimension(..), target,         intent(inout) :: array             !< Array to be received from database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call get_exact_key_array_float_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine get_exact_key_array_float

  !> Get an array of type double from the database using the exact key
  subroutine get_exact_key_array_double( ssc_obj, key, array )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8), dimension(..), target,         intent(inout) :: array             !< Array to be received from database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    type(c_ptr) :: array_ptr
    integer :: ndims, i
    integer(kind=c_int), dimension(MAX_RANK) :: dims
    ! Store the shape of the arrays in reverse order
    ndims = size(shape(array))
    do i=1,ndims
      dims(i) = size(array,ndims+1-i)
    enddo
    array_ptr = c_loc(array)
    c_key = make_c_string(key)
    call get_exact_key_array_double_ssc(ssc_obj, c_key, array_ptr, dims(1:ndims), ndims)
  end subroutine get_exact_key_array_double

  !> Put a scalar of type int32 to the database using the exact key
  subroutine put_exact_key_scalar_int32( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4),                             intent(in   ) :: scalar            !< Scalar value to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    call put_exact_key_scalar_int32_ssc(ssc_obj, c_key, scalar)
  end subroutine put_exact_key_scalar_int32

  !> Put a scalar of type int64 to the database using the exact key
  subroutine put_exact_key_scalar_int64( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8),                             intent(in   ) :: scalar            !< Scalar value to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    call put_exact_key_scalar_int64_ssc(ssc_obj, c_key, scalar)
  end subroutine put_exact_key_scalar_int64

  !> Put a scalar of type float to the database using the exact key
  subroutine put_exact_key_scalar_float( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4),                                intent(in   ) :: scalar            !< Scalar value to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    call put_exact_key_scalar_float_ssc(ssc_obj, c_key, scalar)
  end subroutine put_exact_key_scalar_float

  !> Put a scalar of type double to the database using the exact key
  subroutine put_exact_key_scalar_double( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8),                                intent(in   ) :: scalar            !< Scalar value to send to database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    call put_exact_key_scalar_double_ssc(ssc_obj, c_key, scalar)
  end subroutine put_exact_key_scalar_double

  !> Get a scalar of type int32 from the database using the exact key
  subroutine get_exact_key_scalar_int32( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=4),                             intent(  out) :: scalar            !< Scalar value to get from the database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    scalar = get_exact_key_scalar_int32_ssc(ssc_obj, c_key)
  end subroutine get_exact_key_scalar_int32

  !> Get a scalar of type int64 from the database using the exact key
  subroutine get_exact_key_scalar_int64( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    integer(kind=8),                             intent(  out) :: scalar            !< Scalar value to get from the database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    scalar = get_exact_key_scalar_int64_ssc(ssc_obj, c_key)
  end subroutine get_exact_key_scalar_int64

  !> Get a scalar of type float from the database using the exact key
  subroutine get_exact_key_scalar_float( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=4),                                intent(  out) :: scalar            !< Scalar value to get from the database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    scalar = get_exact_key_scalar_float_ssc(ssc_obj, c_key)
  end subroutine get_exact_key_scalar_float

  !> Get a scalar of type double from the database using the exact key
  subroutine get_exact_key_scalar_double( ssc_obj, key, scalar )
    type(c_ptr), value,                          intent(in   ) :: ssc_obj           !< Pointer to initialized SmartSim client
    character(len=*),                            intent(in   ) :: key               !< Key used in the database for the object
    real(kind=8),                                intent(  out) :: scalar            !< Scalar value to get from the database
    ! Local variables
    character(kind=c_char) :: c_key(len(trim(key))+1)

    c_key = make_c_string(key)
    scalar = get_exact_key_scalar_double_ssc(ssc_obj, c_key)
  end subroutine get_exact_key_scalar_double

! End of generated code

