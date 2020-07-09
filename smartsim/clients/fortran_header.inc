! This code was automatically generated

!>\file
!>\brief Defines user-facing interfaces for overloaded SmartSim functions
public :: put_array_int32
public :: put_array_int64
public :: put_array_float
public :: put_array_double
public :: put_array
public :: poll_key_and_check_scalar_int32
public :: poll_key_and_check_scalar_int64
public :: poll_key_and_check_scalar_float
public :: poll_key_and_check_scalar_double
public :: poll_key_and_check_scalar
public :: get_array_int32
public :: get_array_int64
public :: get_array_float
public :: get_array_double
public :: get_array
public :: put_scalar_int32
public :: put_scalar_int64
public :: put_scalar_float
public :: put_scalar_double
public :: put_scalar
public :: get_scalar_int32
public :: get_scalar_int64
public :: get_scalar_float
public :: get_scalar_double
public :: get_scalar
public :: put_exact_key_array_int32
public :: put_exact_key_array_int64
public :: put_exact_key_array_float
public :: put_exact_key_array_double
public :: put_exact_key_array
public :: poll_exact_key_and_check_scalar_int32
public :: poll_exact_key_and_check_scalar_int64
public :: poll_exact_key_and_check_scalar_float
public :: poll_exact_key_and_check_scalar_double
public :: poll_exact_key_and_check_scalar
public :: get_exact_key_array_int32
public :: get_exact_key_array_int64
public :: get_exact_key_array_float
public :: get_exact_key_array_double
public :: get_exact_key_array
public :: put_exact_key_scalar_int32
public :: put_exact_key_scalar_int64
public :: put_exact_key_scalar_float
public :: put_exact_key_scalar_double
public :: put_exact_key_scalar
public :: get_exact_key_scalar_int32
public :: get_exact_key_scalar_int64
public :: get_exact_key_scalar_float
public :: get_exact_key_scalar_double
public :: get_exact_key_scalar
!> Generic interface for putting an array into the database
interface put_array
  module procedure put_array_int32,&
    put_array_int64,&
    put_array_float,&
    put_array_double
end interface put_array

!> Generic interface for polling database for a key and checking its value
interface poll_key_and_check_scalar
  module procedure poll_key_and_check_scalar_int32,&
    poll_key_and_check_scalar_int64,&
    poll_key_and_check_scalar_float,&
    poll_key_and_check_scalar_double
end interface poll_key_and_check_scalar

!> Generic interface for retrieving an array from the database
interface get_array
  module procedure get_array_int32,&
    get_array_int64,&
    get_array_float,&
    get_array_double
end interface get_array

!> Generic interface for putting a scalar into the database
interface put_scalar
  module procedure put_scalar_int32,&
    put_scalar_int64,&
    put_scalar_float,&
    put_scalar_double
end interface put_scalar

!> Generic interface for retrieving a scalar from the database
interface get_scalar
  module procedure get_scalar_int32,&
    get_scalar_int64,&
    get_scalar_float,&
    get_scalar_double
end interface get_scalar

!> Generic interface for putting an array into the database with the exact key
interface put_exact_key_array
  module procedure put_exact_key_array_int32,&
    put_exact_key_array_int64,&
    put_exact_key_array_float,&
    put_exact_key_array_double
end interface put_exact_key_array

!> Generic interface for polling database for an exact key and checking its value
interface poll_exact_key_and_check_scalar
  module procedure poll_exact_key_and_check_scalar_int32,&
    poll_exact_key_and_check_scalar_int64,&
    poll_exact_key_and_check_scalar_float,&
    poll_exact_key_and_check_scalar_double
end interface poll_exact_key_and_check_scalar

!> Generic interface for retrieving an array from the database with the exact key
interface get_exact_key_array
  module procedure get_exact_key_array_int32,&
    get_exact_key_array_int64,&
    get_exact_key_array_float,&
    get_exact_key_array_double
end interface get_exact_key_array

!> Generic interface for putting a scalar into the database with the exact key
interface put_exact_key_scalar
  module procedure put_exact_key_scalar_int32,&
    put_exact_key_scalar_int64,&
    put_exact_key_scalar_float,&
    put_exact_key_scalar_double
end interface put_exact_key_scalar

!> Generic interface for retrieving a scalar from the database with the exact_key
interface get_exact_key_scalar
  module procedure get_exact_key_scalar_int32,&
    get_exact_key_scalar_int64,&
    get_exact_key_scalar_float,&
    get_exact_key_scalar_double
end interface get_exact_key_scalar

! End of generated code

