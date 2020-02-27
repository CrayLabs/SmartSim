project(ClientTest)
set(CMAKE_BUILD_TYPE Debug)

cmake_minimum_required(VERSION 3.10)

set(BASE_DIR "/lus/snx11254/mellis")
set(LIB_EXT "a")

find_package(MPI)

include_directories(SYSTEM
    ${MPI_INCLUDE_PATH}
    ../../../clients/
    ../../../utils/protobuf/
    ${BASE_DIR}/redis-plus-plus/install/include/
    ${BASE_DIR}/hiredis/install/include/
    ${BASE_DIR}/protobuf/install/include/
    /usr/local/include/
)

SET(CMAKE_CXX_STANDARD 11)

add_executable(client_tester
  client_test.cpp
  ../../../clients/client.cc
  ../../../utils/protobuf/ss_protob_array.pb.cc
)


target_link_libraries(client_tester
  MPI::MPI_C ${CMAKE_THREAD_LIBS_INIT}
  ${BASE_DIR}/redis-plus-plus/install/lib/libredis++.${LIB_EXT}
  ${BASE_DIR}/hiredis/install/lib/libhiredis.${LIB_EXT}
  ${BASE_DIR}/protobuf/install/lib/libprotobuf.${LIB_EXT}
)
