!  Fortran example
   program hello
   use MPI
   use iso_c_binding
   use smartredis_client, only : client_type

   integer :: rank, size, ierror, tag, status(MPI_STATUS_SIZE)
   type(client_type) :: client

   call MPI_INIT(ierror)
   call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)
   call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

   call client%initialize(.true.)

   print*, 'node', rank, ': Hello world'
   call MPI_FINALIZE(ierror)

   end program
