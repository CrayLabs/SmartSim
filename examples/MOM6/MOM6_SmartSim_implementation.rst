Instrumenting MOM6 for SmartSim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Base implementation
-------------------
MOM6 is a 3-dimensional general circulation model- a class of computational
fluid dynamics models specifically designed to solve geophysical flows. A
modified version of the Navier-Stokes equations are solved using a
finite-volume discretization and logically rectangular cells on a structured
grid or unstructured mesh. When run in parallel, the domain is decomposed and
distributed among the available processing elements. The halos (i.e. ghost
cells) are updated via MPI.

Every module within MOM6 stores its own configuration and persistent data in
a "control structure", a derived type. In this example, a new module is
written which will send 3D temperature and salinity at the end
of every modelled day.

.. code-block:: fortran

    module smartsim_connector    
        use client_fortran_api, only :: put_array, init_ssc_client
        use iso_c_binding,      only :: c_ptr
        implicit none; private
    
        type smartsim_connector_type, public :: private
            type(c_ptr)        :: smartsim_client    !< Pointer to an initialized Smartsim Client
            character(len=255) :: unique_id          !< A string used to identify this PE
            logical            :: send_temp = .true. !< If true, send temperatrue
            logical            :: send_salt = .true. !< If true, send salinity
        end type smartsim_connector_type
        
        public :: initialize_smartsim_connector
        public :: send_variables_to_database
    
        contains
    
        !> Initalize the SmartSim client and send any static metadata
        subroutine initialize_smartsim_connector(CS, pe_rank, isg, jsg)
            type(smartsim_connector_type), intent(in) :: CS      !< Control structure for the smartsim connector
            integer,                       intent(in) :: pe_rank !< The rank of the processor, used as an identifier
            integer,                       intent(in) :: isg     !< Index within the global domain along the i-axis
            integer,                       intent(in) :: jsg     !< Index within the global domain along the i-axis
    
            allocate(CS)
            CS%smartsim_client = init_ssc_client()
            write(CS%unique_id,'(I5.5)') pe_rank
            call put_array(CS%smartsim_client, CS%unique_id//"_rank-meta", REAL([/ isg,jsg /]))
    
        end subroutine initialize_smartsim_connector
        
        !> Send the current state of temperature and salinity to the database
        subroutine send_variables_to_database(CS, temp, salt)
            type(smartsim_connector_type), intent(in) :: CS   !< Control structure for the smartsim connector
            real, dimension(:,:,:),        intent(in) :: temp !< 3D array of ocean temperatures
            real, dimension(:,:,:),        intent(in) :: salt !< 3D array of ocean salinity
    
            if (CS%send_temp) call put_array(CS%smartsim_client, CS%unique_id//"_temp", temp)
            if (CS%send_salt) call put_array(CS%smartsim_client, CS%unique_id//"_salt", salt)
        end subroutine send_variables_to_database
    
    end module smartsim_connector

During model intialization, ``initialize_smartsim_connector`` is called. The
SmartSim client is initialized in the same way as in the previous toy
example, however an additional two steps are taken because data will be sent
from multiple processors. First, the rank of the PE will be stored and used
when constructing a key to be sent to the database. Without this, every PE
would send their own arrays using the same key resulting in data being
overwritten. Second, a two integer array is sent to the database containing
the index within the global (non-decomposed) domain corresponding to the
first element of the arrays containing temperature and salinity.

In ``send_variables_to_database``, the arrays containing temperature and
salinity are sent to the database. The unique id is again added to the key
to ensure that data is not overwritten by another processor.