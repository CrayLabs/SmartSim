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
  int result = 0;

  int poll_freq = 10;
  int ntries = 1;

  char scalar_int32_key[] = "scalar_int32";
  int32_t true_scalar_int32 = safe_rand();
  int32_t fake_scalar_int32 = true_scalar_int32 + 1;
  put_scalar_int32_c(smartsim_client, scalar_int32_key, true_scalar_int32);
  result = poll_key_and_check_scalar_int32_c(smartsim_client, scalar_int32_key, fake_scalar_int32, poll_freq, ntries);
  test_result( result == 0, "poll_key_and_check_int32_expect_false" );
  result = poll_key_and_check_scalar_int32_c(smartsim_client, scalar_int32_key, true_scalar_int32, poll_freq, ntries);
  test_result( result == 1, "poll_key_and_check_int32_expect_true" );

  char scalar_uint32_key[] = "scalar_uint32";
  uint32_t true_scalar_uint32 = safe_rand();
  uint32_t fake_scalar_uint32 = true_scalar_uint32 + 1;
  put_scalar_uint32_c(smartsim_client, scalar_uint32_key, true_scalar_uint32);
  result = poll_key_and_check_scalar_uint32_c(smartsim_client, scalar_uint32_key, fake_scalar_uint32, poll_freq, ntries);
  test_result( result == 0, "poll_key_and_check_uint32_expect_false" );
  result = poll_key_and_check_scalar_uint32_c(smartsim_client, scalar_uint32_key, true_scalar_uint32, poll_freq, ntries);
  test_result( result == 1, "poll_key_and_check_uint32_expect_true" );

  char scalar_int64_key[] = "scalar_int64";
  int64_t true_scalar_int64 = safe_rand();
  int64_t fake_scalar_int64 = true_scalar_int64 + 1;
  put_scalar_int64_c(smartsim_client, scalar_int64_key, true_scalar_int64);
  result = poll_key_and_check_scalar_int64_c(smartsim_client, scalar_int64_key, fake_scalar_int64, poll_freq, ntries);
  test_result( result == 0, "poll_key_and_check_int64_expect_false" );
  result = poll_key_and_check_scalar_int64_c(smartsim_client, scalar_int64_key, true_scalar_int64, poll_freq, ntries);
  test_result( result == 1, "poll_key_and_check_int64_expect_true" );

  char scalar_uint64_key[] = "scalar_uint64";
  uint64_t true_scalar_uint64 = safe_rand();
  uint64_t fake_scalar_uint64 = true_scalar_uint64 + 1;
  put_scalar_uint64_c(smartsim_client, scalar_uint64_key, true_scalar_uint64);
  result = poll_key_and_check_scalar_uint64_c(smartsim_client, scalar_uint64_key, fake_scalar_uint64, poll_freq, ntries);
  test_result( result == 0, "poll_key_and_check_uint64_expect_false" );
  result = poll_key_and_check_scalar_uint64_c(smartsim_client, scalar_uint64_key, true_scalar_uint64, poll_freq, ntries);
  test_result( result == 1, "poll_key_and_check_uint64_expect_true" );

  char scalar_float_key[] = "scalar_float";
  float true_scalar_float = rand_float();
  float fake_scalar_float = true_scalar_float + 1;
  put_scalar_float_c(smartsim_client, scalar_float_key, true_scalar_float);
  result = poll_key_and_check_scalar_float_c(smartsim_client, scalar_float_key, fake_scalar_float, poll_freq, ntries);
  test_result( result == 0, "poll_key_and_check_float_expect_false" );
  result = poll_key_and_check_scalar_float_c(smartsim_client, scalar_float_key, true_scalar_float, poll_freq, ntries);
  test_result( result == 1, "poll_key_and_check_float_expect_true" );

  char scalar_double_key[] = "scalar_double";
  double true_scalar_double = rand_double();
  double fake_scalar_double = true_scalar_double + 1;
  put_scalar_double_c(smartsim_client, scalar_double_key, true_scalar_double);
  result = poll_key_and_check_scalar_double_c(smartsim_client, scalar_double_key, fake_scalar_double, poll_freq, ntries);
  test_result( result == 0, "poll_key_and_check_double_expect_false" );
  result = poll_key_and_check_scalar_double_c(smartsim_client, scalar_double_key, true_scalar_double, poll_freq, ntries);
  test_result( result == 1, "poll_key_and_check_double_expect_true" );

  return 0;
}
