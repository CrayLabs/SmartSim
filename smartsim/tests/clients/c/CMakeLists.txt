project(CClientTester)

set(CMAKE_BUILD_TYPE Release)

cmake_minimum_required(VERSION 3.10)

SET(CMAKE_CXX_STANDARD 11)

#TODO put in output warnings
# Determine where the hiredis library is based on if an environment variable
# is set, and if not, it can be found in the $PATH directory
if(DEFINED ENV{HIREDIS_INSTALL_PATH})
	   string(CONCAT HIREDIS_LIB_PATH $ENV{HIREDIS_INSTALL_PATH} "/lib")
	   find_library(HIREDIS_LIB hiredis PATHS ${HIREDIS_LIB_PATH} NO_DEFAULT_PATH)
	   string(CONCAT HIREDIS_INCLUDE_PATH $ENV{HIREDIS_INSTALL_PATH} "/include/")
else()
     	   find_library(HIREDIS_LIB hiredis PATH_SUFFIXES "/lib")
	   find_path(HIREDIS_INCLUDE_PATH hiredis PATH_SUFFIXES "/include")
endif()

# Determine where the protobuf library is based on if an environment variable
# is set, and if not, it can be found in the $PATH directory
if(DEFINED ENV{PROTOBUF_INSTALL_PATH})
	   string(CONCAT PROTOBUF_LIB_PATH $ENV{PROTOBUF_INSTALL_PATH} "/lib")
	   find_library(PROTOBUF_LIB protobuf PATHS ${PROTOBUF_LIB_PATH} NO_DEFAULT_PATH)
	   string(CONCAT PROTOBUF_INCLUDE_PATH $ENV{PROTOBUF_INSTALL_PATH} "/include/")
else()
     	   find_library(PROTOBUF_LIB protobuf PATH_SUFFIXES "/lib")
	   find_path(PROTOBUF_INCLUDE_PATH protobuf PATH_SUFFIXES "/include")
endif()

# Determine where the redis++ library is based on if an environment variable
# is set, and if not, it can be found in the $PATH directory
if(DEFINED ENV{REDISPP_INSTALL_PATH})
	   string(CONCAT REDISPP_LIB_PATH $ENV{REDISPP_INSTALL_PATH} "/lib")
	   find_library(REDISPP_LIB redis++ PATHS ${REDISPP_LIB_PATH} )
	   string(CONCAT REDISPP_INCLUDE_PATH $ENV{REDISPP_INSTALL_PATH} "/include/")
else()
     	   find_library(REDISPP_LIB redis++ PATH_SUFFIXES "/lib")
	   find_path(REDISPP_INCLUDE_PATH redis++ PATH_SUFFIXES "/include")
endif()

find_package(MPI)

include_directories(SYSTEM
    ${MPI_INCLUDE_PATH}
    ../../../clients/
    ../../../utils/protobuf/
    /usr/local/include
    ${HIREDIS_INCLUDE_PATH}
    ${REDISPP_INCLUDE_PATH}
    ${PROTOBUF_INCLUDE_PATH}
)

set(CLIENT_LIBRARIES ${REDISPP_LIB} ${HIREDIS_LIB} ${PROTOBUF_LIB})
set(CLIENT_SRC ../../../clients/c_client.cc ../../../clients/client.cc ../../../utils/protobuf/smartsim_protobuf.pb.cc)

add_executable(client_test_scalar_put_get
	client_test_scalar_put_get.c
	${CLIENT_SRC}
)

target_link_libraries(client_test_scalar_put_get
	MPI::MPI_C
	${CLIENT_LIBRARIES}
)

add_executable(client_test_array_1d_put_get
	client_test_array_1d_put_get.c
	${CLIENT_SRC}
)

target_link_libraries(client_test_array_1d_put_get
	MPI::MPI_C
	${CLIENT_LIBRARIES}
)

add_executable(client_test_array_2d_put_get
	client_test_array_2d_put_get.c
	${CLIENT_SRC}
)

target_link_libraries(client_test_array_2d_put_get
	MPI::MPI_C
	${CLIENT_LIBRARIES}
)

add_executable(client_test_poll_key_and_check
	client_test_poll_key_and_check_scalar.c
	${CLIENT_SRC}
)

target_link_libraries(client_test_poll_key_and_check
	MPI::MPI_C
	${CLIENT_LIBRARIES}
)

add_executable(client_test_exact_key_scalar_put_get
	client_test_exact_key_scalar_put_get.c
	${CLIENT_SRC}
)

target_link_libraries(client_test_exact_key_scalar_put_get
	MPI::MPI_C
	${CLIENT_LIBRARIES}
)

add_executable(client_test_exact_key_array_1d_put_get
	client_test_exact_key_array_1d_put_get.c
	${CLIENT_SRC}
)

target_link_libraries(client_test_exact_key_array_1d_put_get
	MPI::MPI_C
	${CLIENT_LIBRARIES}
)

add_executable(client_test_exact_key_array_2d_put_get
	client_test_exact_key_array_2d_put_get.c
	${CLIENT_SRC}
)

target_link_libraries(client_test_exact_key_array_2d_put_get
	MPI::MPI_C
	${CLIENT_LIBRARIES}
)

add_executable(client_test_poll_exact_key_and_check
	client_test_poll_key_and_check_scalar.c
	${CLIENT_SRC}
)

target_link_libraries(client_test_poll_exact_key_and_check
	MPI::MPI_C
	${CLIENT_LIBRARIES}
)
