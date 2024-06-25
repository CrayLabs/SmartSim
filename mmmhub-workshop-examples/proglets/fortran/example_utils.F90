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

module example_utils

  implicit none; private

  public :: irand
  public :: use_cluster

  contains

  !> Returns a random integer between 0 and 255
  integer function irand()
    real :: real_rand

    call random_number(real_rand)

    irand = nint(real_rand*255)
  end function irand

  logical function use_cluster()

    character(len=16) :: server_type

    call get_environment_variable('SR_DB_TYPE', server_type)
    server_type = to_lower(server_type)
    if (len_trim(server_type)>0) then
      select case (server_type)
        case ('clustered')
          use_cluster = .true.
        case ('standalone')
          use_cluster = .false.
        case default
          use_cluster = .false.
      end select
    endif

  end function use_cluster

  !> Returns a lower case version of the string. Only supports a-z
  function to_lower( str ) result(lower_str)
    character(len=*),          intent(in   ) :: str !< String
    character(len = len(str)) :: lower_str

    character(26), parameter :: caps = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    character(26), parameter :: lows = 'abcdefghijklmnopqrstuvwxyz'

    integer :: i, i_low

    lower_str = str
    do i=1,len_trim(str)
      i_low = index(caps,str(i:i))
      if (i_low > 0) lower_str(i:i) = lows(i_low:i_low)
    enddo

  end function to_lower

end module example_utils
