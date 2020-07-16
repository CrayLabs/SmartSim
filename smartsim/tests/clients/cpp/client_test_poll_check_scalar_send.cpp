#include "client.h"
#include <mpi.h>
#include "client_test_utils.h"
#include <limits>

template <typename T_send>
void put_scalar(void (SmartSimClient::*sendFunction)(const char*, T_send),
	              T_send send_value, std::string key_suffix="") {
  SmartSimClient client(false);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string key = "poll_and_test_scalar_rank_" + std::to_string(rank)
                    + key_suffix;

  //Put a pause here so that we can multiple times before receiving
  std::cout<<"Rank "<<rank<<" sending scalar with key: "<<key<<std::endl
           <<std::flush;

  (client.*sendFunction)(key.c_str(), send_value);

  std::cout<<"Rank "<<rank<<" finished sending key: "<<key<<std::endl
           <<std::flush;
  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  put_scalar<double>(&SmartSimClient::put_scalar_double,
    std::numeric_limits<double>::max(), "_dbl");

  put_scalar<float>(&SmartSimClient::put_scalar_float,
    std::numeric_limits<float>::max(), "_flt");

  put_scalar<int64_t>(&SmartSimClient::put_scalar_int64,
		std::numeric_limits<int64_t>::min(), "_i64");

  put_scalar<int32_t>(&SmartSimClient::put_scalar_int32,
		std::numeric_limits<int32_t>::min(), "_i32");

  put_scalar<uint64_t>(&SmartSimClient::put_scalar_uint64,
		std::numeric_limits<uint64_t>::max(), "_ui64");

  put_scalar<uint32_t>(&SmartSimClient::put_scalar_uint32,
		std::numeric_limits<uint32_t>::max(), "_ui32");

  MPI_Finalize();

  return 0;
}
