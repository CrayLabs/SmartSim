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

program mnist_example

  use mpi
  use iso_c_binding
  use smartredis_client, only : client_type

  implicit none

#include "enum_fortran.inc"

  character(len=*), parameter :: model_key = "mnist_model"
  character(len=*), parameter :: model_file = "./mnist_cnn.pt"
  character(len=*), parameter :: script_key = "mnist_script"
  character(len=*), parameter :: script_file = "./data_processing_script.txt"

  type(client_type) :: client
  integer :: err_code, pe_id, result
  character(len=2) :: key_suffix

  ! Initialize MPI and get the rank of the processor
  call MPI_init(err_code)
  call MPI_comm_rank( MPI_COMM_WORLD, pe_id, err_code)

  ! Format the suffix for a key as a zero-padded version of the rank
  write(key_suffix, "(A,I1.1)") "_",pe_id

  ! Initialize a client
  result = client%initialize("smartredis_mnist")
  if (result .ne. SRNoError) error stop 'client%initialize failed'

  ! Set up model and script for the computation
  if (pe_id == 0) then
    result = client%set_model_from_file(model_key, model_file, "TORCH", "CPU")
    if (result .ne. SRNoError) error stop 'client%set_model_from_file failed'
    result = client%set_script_from_file(script_key, "CPU", script_file)
    if (result .ne. SRNoError) error stop 'client%set_script_from_file failed'
  endif

  ! Get all PEs lined up
  call MPI_barrier(MPI_COMM_WORLD, err_code)

  ! Run the main computation
  call run_mnist(client, key_suffix, model_key, script_key)

  ! Shut down MPI
  call MPI_finalize(err_code)

  ! Check final result
  if (pe_id == 0) then
    print *, "SmartRedis Fortran MPI MNIST example finished without errors."
  endif

  ! Done
  call exit()

contains

subroutine run_mnist( client, key_suffix, model_name, script_name )
  type(client_type), intent(in) :: client
  character(len=*),  intent(in) :: key_suffix
  character(len=*),  intent(in) :: model_name
  character(len=*),  intent(in) :: script_name

  integer, parameter :: mnist_dim1 = 28
  integer, parameter :: mnist_dim2 = 28
  integer, parameter :: result_dim1 = 10

  real, dimension(1,1,mnist_dim1,mnist_dim2) :: array
  real, dimension(1,result_dim1) :: output_result

  character(len=255) :: in_key
  character(len=255) :: script_out_key
  character(len=255) :: out_key

  character(len=255), dimension(1) :: inputs
  character(len=255), dimension(1) :: outputs

  ! Construct the keys used for the specifiying inputs and outputs
  in_key = "mnist_input_rank"//trim(key_suffix)
  script_out_key = "mnist_processed_input_rank"//trim(key_suffix)
  out_key = "mnist_processed_input_rank"//trim(key_suffix)

  ! Generate some fake data for inference and send it to the database
  call random_number(array)
  result = client%put_tensor(in_key, array, shape(array))
  if (result .ne. SRNoError) error stop 'client%put_tensor failed'

  ! Prepare the script inputs and outputs
  inputs(1) = in_key
  outputs(1) = script_out_key
  result = client%run_script(script_name, "pre_process", inputs, outputs)
  if (result .ne. SRNoError) error stop 'client%run_script failed'
  inputs(1) = script_out_key
  outputs(1) = out_key
  result = client%run_model(model_name, inputs, outputs)
  if (result .ne. SRNoError) error stop 'client%run_model failed'
  output_result(:,:) = 0.
  result = client%unpack_tensor(out_key, output_result, shape(output_result))
  if (result .ne. SRNoError) error stop 'client%unpack_tensor failed'

end subroutine run_mnist

end program mnist_example
