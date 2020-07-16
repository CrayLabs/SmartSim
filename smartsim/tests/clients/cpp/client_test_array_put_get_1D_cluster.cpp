#include "client.h"
#include <mpi.h>
#include "client_test_utils.h"

template <typename T_send, typename T_recv>
void put_get_1D_array(
      void (SmartSimClient::*sendFunction)(const char*, void*, int*, int),
		  void (SmartSimClient::*recvFunction)(const char*, void*, int*, int),
		  void (*fill_array)(T_send*, int), int dim1, std::string key_suffix="")
{

  SmartSimClient client(true);

  T_send* array = new T_send[dim1];
  T_recv* result = new T_recv[dim1];

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  fill_array(array, dim1);

  int dims[1] = {dim1};

  std::string key = "1d_test_rank_"+std::to_string(rank) + key_suffix;

  std::cout<<"Rank "<<rank<<" starting put."<<std::endl<<std::flush;
  (client.*sendFunction)(key.c_str(), array, dims, 1);
  std::cout<<"Rank "<<rank<<" finished put."<<std::endl<<std::flush;

  MPI_Barrier(MPI_COMM_WORLD);

  if(!client.key_exists(key.c_str()))
    throw std::runtime_error("Key existence could not be "\
			     "verified with key_exists()");

  std::cout<<"Rank "<<rank<<" starting get."<<std::endl<<std::flush;
  (client.*recvFunction)(key.c_str(), result, dims, 1);
  std::cout<<"Rank "<<rank<<" finished get."<<std::endl<<std::flush;

  if (!is_equal_1D_array<T_send, T_recv>(array, result, dim1))
      throw std::runtime_error("The arrays don't match");

  free_1D_array(array);
  free_1D_array(result);

  MPI_Barrier(MPI_COMM_WORLD);

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int dim_1 = atoi(argv[1]);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  void (SmartSimClient::*sendFunction)(const char*, void*, int*, int);
  void (SmartSimClient::*recvFunction)(const char*, void*, int*, int);

  sendFunction = &SmartSimClient::put_array_double;
  recvFunction = &SmartSimClient::get_array_double;
  put_get_1D_array<double,double>(sendFunction, recvFunction,
				  &set_1D_array_floating_point_values<double>,
				  dim_1, "_dbl");

  sendFunction = &SmartSimClient::put_array_float;
  recvFunction = &SmartSimClient::get_array_float;
  put_get_1D_array<float,float>(sendFunction, recvFunction,
				&set_1D_array_floating_point_values<float>,
				dim_1, "_flt");

  sendFunction = &SmartSimClient::put_array_int64;
  recvFunction = &SmartSimClient::get_array_int64;
  put_get_1D_array<int64_t,int64_t>(sendFunction, recvFunction,
				    &set_1D_array_integral_values<int64_t>,
				    dim_1, "_i64");

  sendFunction = &SmartSimClient::put_array_int32;
  recvFunction = &SmartSimClient::get_array_int32;
  put_get_1D_array<int32_t,int32_t>(sendFunction, recvFunction,
				    &set_1D_array_integral_values<int32_t>,
				    dim_1, "_i32");

  sendFunction = &SmartSimClient::put_array_uint64;
  recvFunction = &SmartSimClient::get_array_uint64;
  put_get_1D_array<uint64_t,uint64_t>(sendFunction, recvFunction,
				      &set_1D_array_integral_values<uint64_t>,
				      dim_1, "_ui64");

  sendFunction = &SmartSimClient::put_array_uint32;
  recvFunction = &SmartSimClient::get_array_uint32;
  put_get_1D_array<uint32_t,uint32_t>(sendFunction, recvFunction,
				      &set_1D_array_integral_values<uint32_t>,
				      dim_1, "_ui32");

  MPI_Finalize();

  if(rank==0)
    std::cout<<"Finished 1D put and get tests."<<std::endl;

  return 0;
}
