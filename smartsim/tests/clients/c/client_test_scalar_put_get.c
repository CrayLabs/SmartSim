#include <mpi.h>
#include "c_client.h"
#include "c_client_test_utils.h"
#include <inttypes.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {

  MPI_Init(NULL,NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  void *smartsim_client = initialize_c_client( 0 );

  char scalar_int32_key[] = "scalar_int32";
  int32_t true_scalar_int32 = safe_rand();
  int32_t recv_scalar_int32;
  put_scalar_int32_c(smartsim_client, scalar_int32_key, true_scalar_int32);
  recv_scalar_int32 = get_scalar_int32_c(smartsim_client, scalar_int32_key);
  test_result( true_scalar_int32 == recv_scalar_int32, "put_scalar_int32" );

  char scalar_uint32_key[] = "scalar_uint32";
  uint32_t true_scalar_uint32 = safe_rand();
  uint32_t recv_scalar_uint32;
  put_scalar_uint32_c(smartsim_client, scalar_uint32_key, true_scalar_uint32);
  recv_scalar_uint32 = get_scalar_uint32_c(smartsim_client, scalar_uint32_key);
  test_result( true_scalar_uint32 == recv_scalar_uint32, "put_scalar_uint32" );

  char scalar_int64_key[] = "scalar_int64";
  int64_t true_scalar_int64 = safe_rand();
  int64_t recv_scalar_int64;
  put_scalar_int64_c(smartsim_client, scalar_int64_key, true_scalar_int64);
  recv_scalar_int64 = get_scalar_int64_c(smartsim_client, scalar_int64_key);
  test_result( true_scalar_int64 == recv_scalar_int64, "put_scalar_int64" );

  char scalar_uint64_key[] = "scalar_uint64";
  uint64_t true_scalar_uint64 = safe_rand();
  uint64_t recv_scalar_uint64;
  put_scalar_uint64_c(smartsim_client, scalar_uint64_key, true_scalar_uint64);
  recv_scalar_uint64 = get_scalar_uint64_c(smartsim_client, scalar_uint64_key);
  test_result( true_scalar_uint64 == recv_scalar_uint64, "put_scalar_uint64" );

  char scalar_float_key[] = "scalar_float";
  float true_scalar_float = rand_float();
  float recv_scalar_float;
  put_scalar_float_c(smartsim_client, scalar_float_key, true_scalar_float);
  recv_scalar_float = get_scalar_float_c(smartsim_client, scalar_float_key);
  test_result( true_scalar_float == recv_scalar_float, "put_scalar_float" );

  char scalar_double_key[] = "scalar_double";
  double true_scalar_double = rand_double();
  double recv_scalar_double;
  put_scalar_double_c(smartsim_client, scalar_double_key, true_scalar_double);
  recv_scalar_double = get_scalar_double_c(smartsim_client, scalar_double_key);
  test_result( true_scalar_double == recv_scalar_double, "put_scalar_double" );

  return 0;
}
