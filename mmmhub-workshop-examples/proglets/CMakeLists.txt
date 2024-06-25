# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Project definition for the SmartRedis-Examples-Parallel-Fortran project
cmake_minimum_required(VERSION 3.13)
project(SmartRedis-Examples-Parallel-Fortran)

# Enable language support for the examples
enable_language(Fortran)
enable_language(CXX)

# Configure the build
set(CMAKE_CXX_STANDARD 17)

# Assume by default that users should link against the install directory in this repository
if(NOT DEFINED SMARTREDIS_INSTALL_PATH)
  set(SMARTREDIS_INSTALL_PATH "$ENV{HOME}/smartredis/install/")
endif()

find_package(MPI REQUIRED)
IF(NOT MPI_Fortran_FOUND)
	message(FATAL_ERROR "Could not find Fortran MPI components")
endif()
#  . Main SmartRedis Library (C/C++ based)
add_library(smartredis SHARED IMPORTED)
set_target_properties(smartredis PROPERTIES
  IMPORTED_LOCATION "${SMARTREDIS_INSTALL_PATH}/lib/libsmartredis.so"
  INTERFACE_INCLUDE_DIRECTORIES "${SMARTREDIS_INSTALL_PATH}/include/"
)
add_library(smartredis-fortran SHARED IMPORTED)
set_target_properties(smartredis-fortran PROPERTIES
  IMPORTED_LOCATION "${SMARTREDIS_INSTALL_PATH}/lib/libsmartredis-fortran.so"
  INTERFACE_INCLUDE_DIRECTORIES "${SMARTREDIS_INSTALL_PATH}/include/"
)

# Build C++ proglets
list(APPEND CPP_EXECUTABLES
	smartredis_put_get_3D
	smartredis_mnist
)

# Build the examples
foreach(EXECUTABLE ${CPP_EXECUTABLES})
    add_executable(${EXECUTABLE}_cpp_parallel
        cpp/${EXECUTABLE}.cpp
    )
    target_link_libraries(${EXECUTABLE}_cpp_parallel
    	MPI::MPI_CXX
		smartredis
    )
	install(TARGETS ${EXECUTABLE}_cpp_parallel)
endforeach()

# Build Fortran proglets
add_library(example_utils STATIC fortran/example_utils.F90)
# Define all the examples to be built
list(APPEND FORTRAN_EXECUTABLES
	 smartredis_dataset
	 smartredis_mnist
	 smartredis_put_get_3D
)

# Build the examples
foreach(EXECUTABLE ${FORTRAN_EXECUTABLES})
	add_executable(${EXECUTABLE}_fortran_parallel
		fortran/${EXECUTABLE}.F90
	)
	target_link_libraries(${EXECUTABLE}_fortran_parallel
		smartredis
		smartredis-fortran
		MPI::MPI_Fortran
		example_utils
	)
	install(TARGETS ${EXECUTABLE}_fortran_parallel)
endforeach()
