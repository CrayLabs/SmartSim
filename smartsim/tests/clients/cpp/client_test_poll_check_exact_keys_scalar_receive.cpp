#include "client.h"
#include <mpi.h>
#include "client_test_utils.h"
#include <limits>

template <typename T_recv>
void check_scalar(bool (SmartSimClient::*checkFunction)
                        (const char*, T_recv, int, int),
          		    T_recv comp_value, std::string key_suffix="")
{
  SmartSimClient client;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string key = "poll_and_test_exact_key_scalar_rank_"
                    + std::to_string(rank) + key_suffix;

  std::cout<<"Rank "<<rank<<" polling for key: "<<key
    <<std::endl<<std::flush;

  if(!(client.*checkFunction)(key.c_str(),comp_value, 200, 10))
    throw std::runtime_error("Poll and check failed.");

  std::cout<<"Rank "<<rank<<" finished polling for key: "<<key
    <<std::endl<<std::flush;

  if(!(client.poll_exact_key(key.c_str(),200,10)))
    throw std::runtime_error("Poll for exact key failed.");

  if(!(client.exact_key_exists(key.c_str())))
      throw std::runtime_error("Exact key exists checked failed.");

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  check_scalar<double>(
    &SmartSimClient::poll_exact_key_and_check_scalar_double,
		std::numeric_limits<double>::max(), "_dbl");

  check_scalar<float>(
    &SmartSimClient::poll_exact_key_and_check_scalar_float,
    std::numeric_limits<float>::max(), "_flt");

  check_scalar<int64_t>(
    &SmartSimClient::poll_exact_key_and_check_scalar_int64,
		std::numeric_limits<int64_t>::min(), "_i64");

  check_scalar<int32_t>(
    &SmartSimClient::poll_exact_key_and_check_scalar_int32,
		std::numeric_limits<int32_t>::min(), "_i32");

  check_scalar<uint64_t>(
    &SmartSimClient::poll_exact_key_and_check_scalar_uint64,
	  std::numeric_limits<uint64_t>::max(),"_ui64");

  check_scalar<uint32_t>(
    &SmartSimClient::poll_exact_key_and_check_scalar_uint32,
		std::numeric_limits<uint32_t>::max(), "_ui32");

  MPI_Finalize();

  return 0;
}
