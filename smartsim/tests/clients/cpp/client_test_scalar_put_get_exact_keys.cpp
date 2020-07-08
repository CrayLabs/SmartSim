#include "client.h"
#include <mpi.h>
#include "client_test_utils.h"
#include <limits>

template <typename T_send, typename T_recv>
void put_get_scalar(
        void (SmartSimClient::*sendFunction)(const char*, T_send),
		    T_recv (SmartSimClient::*recvFunction)(const char*),
		    T_send send_value, std::string key_suffix="")
{
  SmartSimClient client;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string key = "exact_key_rank_"+std::to_string(rank) + key_suffix;

  std::cout<<"Rank "<<rank<<" starting put with key: "<<key
    <<std::endl<<std::flush;

  (client.*sendFunction)(key.c_str(), send_value);

  std::cout<<"Rank "<<rank<<" finished put with key: "<<key
    <<std::endl<<std::flush;

  if(!client.exact_key_exists(key.c_str()))
    throw std::runtime_error("Key existence could not be "\
			     "verified with key_exists()");

  std::cout<<"Rank "<<rank<<" starting get with key: "<<key
    <<std::endl<<std::flush;

  T_recv recv_value = (client.*recvFunction)(key.c_str());

  std::cout<<"Rank "<<rank<<" finished get with key: "<<key
    <<std::endl<<std::flush;

  if (recv_value != send_value)
      throw std::runtime_error("The scalars don't match");

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  put_get_scalar<double,double>(
        &SmartSimClient::put_exact_key_scalar_double,
				&SmartSimClient::get_exact_key_scalar_double,
				std::numeric_limits<double>::max(), "_dbl");

  put_get_scalar<float,float>(
            &SmartSimClient::put_exact_key_scalar_float,
			      &SmartSimClient::get_exact_key_scalar_float,
			      std::numeric_limits<float>::max(), "_flt");

  put_get_scalar<int64_t,int64_t>(
          &SmartSimClient::put_exact_key_scalar_int64,
				  &SmartSimClient::get_exact_key_scalar_int64,
				  std::numeric_limits<int64_t>::min(), "_i64");

  put_get_scalar<int32_t,int32_t>(
          &SmartSimClient::put_exact_key_scalar_int32,
				  &SmartSimClient::get_exact_key_scalar_int32,
				  std::numeric_limits<int32_t>::min(), "_i32");

  put_get_scalar<uint64_t,uint64_t>(
            &SmartSimClient::put_exact_key_scalar_uint64,
				    &SmartSimClient::get_exact_key_scalar_uint64,
				    std::numeric_limits<uint64_t>::max(), "_ui64");

  put_get_scalar<uint32_t,uint32_t>(
            &SmartSimClient::put_exact_key_scalar_uint32,
				    &SmartSimClient::get_exact_key_scalar_uint32,
				    std::numeric_limits<uint32_t>::max(), "_ui32");

  return 0;
}
