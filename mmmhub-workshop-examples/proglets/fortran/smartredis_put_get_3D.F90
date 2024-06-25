! BSD 2-Clause License
!
! Copyright (c) 2021-2024, Hewlett Packard Enterprise
! All rights reserved.
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions are met:
!
! 1. Redistributions of source code must retain the above copyright notice, this
!    list of conditions and the following disclaimer.
!
! 2. Redistributions in binary form must reproduce the above copyright notice,
!    this list of conditions and the following disclaimer in the documentation
!    and/or other materials provided with the distribution.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
! AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
! IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
! DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
! FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
! DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
! SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
! OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
! OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

program main

  use mpi
  use iso_c_binding
  use smartredis_client, only : client_type

  implicit none

#include "enum_fortran.inc"

  integer, parameter :: dim1 = 10
  integer, parameter :: dim2 = 20
  integer, parameter :: dim3 = 30

  real(kind=8),    dimension(dim1, dim2, dim3) :: recv_array_real_64

  real(kind=c_double),    dimension(dim1, dim2, dim3) :: true_array_real_64

  integer :: i, j, k, result
  type(client_type) :: client

  integer :: err_code, pe_id
  character(len=9) :: key_prefix

  ! Initialize MPI
  call MPI_init( err_code )
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code)
  write(key_prefix, "(A,I6.6)") "pe_",pe_id

  call random_number(true_array_real_64)
  call random_number(recv_array_real_64)

  ! Initialize a client
  result = client%initialize("smartredis_put_get_3D")
  if (result .ne. SRNoError) error stop 'client%initialize failed'

  ! Add a tensor to the database and verify that we can retrieve it
  result = client%put_tensor(key_prefix//"true_array_real_64", true_array_real_64, shape(true_array_real_64))
  if (result .ne. SRNoError) error stop 'client%put_tensor failed'
  result = client%unpack_tensor(key_prefix//"true_array_real_64", recv_array_real_64, shape(recv_array_real_64))
  if (result .ne. SRNoError) error stop 'client%unpack_tensor failed'
  if (.not. all(true_array_real_64 == recv_array_real_64)) error stop 'true_array_real_64: FAILED'

  ! Shut down MPI
  call mpi_finalize(err_code)
  if (pe_id == 0) write(*,*) "SmartRedis MPI Fortran example 3D put/get finished."

  ! Done
  call exit()

end program main
