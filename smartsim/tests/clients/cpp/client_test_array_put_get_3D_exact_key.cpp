#include "client.h"
#include "client_test_utils.h"
#include <mpi.h>

template <typename T_send, typename T_recv>
void put_get_3D_array(void (SmartSimClient::*sendFunction)(const char*, void*, int*, int),
		      void (SmartSimClient::*recvFunction)(const char*, void*, int*, int),
		      void (*fill_array)(T_send***, int, int, int),
		      int dim1, int dim2, int dim3, std::string key_suffix="")
{

  SmartSimClient client;

  T_send*** array = allocate_3D_array<T_send>(dim1, dim2, dim3);

  fill_array(array, dim1, dim2, dim3);

  int dims[3] = {dim1, dim2, dim3};

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string key = "3d_exact_key_rank_"+std::to_string(rank)+"_"+key_suffix;

  std::cout<<"Rank "<<rank<<" starting put."<<std::endl<<std::flush;
  (client.*sendFunction)(key.c_str(), array, dims, 3);
  std::cout<<"Rank "<<rank<<" finished put."<<std::endl<<std::flush;

  MPI_Barrier(MPI_COMM_WORLD);

  if(!client.exact_key_exists(key.c_str()))
    throw std::runtime_error("Key existence could not be "\
			     "verified with key_exists()");

  T_recv*** result = allocate_3D_array<T_recv>(dim1, dim2, dim3);

  std::cout<<"Rank "<<rank<<" starting get."<<std::endl<<std::flush;
  (client.*recvFunction)(key.c_str(), result, dims, 3);
  std::cout<<"Rank "<<rank<<" finished get."<<std::endl<<std::flush;

  if (!is_equal_3D_array<T_send, T_recv>(array, result, dim1, dim2, dim3))
    throw std::runtime_error("The results do not match for "	\
			     "the 2d put and get test!");


  free_3D_array(array, dim1, dim2);
  free_3D_array(result, dim1, dim2);

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int dim1 = atoi(argv[1]);
  int dim2 = dim1;
  int dim3 = dim1;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  void (SmartSimClient::*sendFunction)(const char*, void*, int*, int);
  void (SmartSimClient::*recvFunction)(const char*, void*, int*, int);

  sendFunction = &SmartSimClient::put_exact_key_array_double;
  recvFunction = &SmartSimClient::get_exact_key_array_double;
  put_get_3D_array<double,double>(sendFunction, recvFunction,
				  &set_3D_array_floating_point_values<double>,
				  dim1, dim2, dim3, "_dbl");

  sendFunction = &SmartSimClient::put_exact_key_array_float;
  recvFunction = &SmartSimClient::get_exact_key_array_float;
  put_get_3D_array<float,float>(sendFunction, recvFunction,
				&set_3D_array_floating_point_values<float>,
				dim1, dim2, dim3, "_flt");

  sendFunction = &SmartSimClient::put_exact_key_array_int64;
  recvFunction = &SmartSimClient::get_exact_key_array_int64;
  put_get_3D_array<int64_t,int64_t>(sendFunction, recvFunction,
				    &set_3D_array_integral_values<int64_t>,
				    dim1, dim2, dim3, "_i64");

  sendFunction = &SmartSimClient::put_exact_key_array_int32;
  recvFunction = &SmartSimClient::get_exact_key_array_int32;
  put_get_3D_array<int32_t,int32_t>(sendFunction, recvFunction,
				    &set_3D_array_integral_values<int32_t>,
				    dim1, dim2, dim3, "_i32");

  sendFunction = &SmartSimClient::put_exact_key_array_uint64;
  recvFunction = &SmartSimClient::get_exact_key_array_uint64;
  put_get_3D_array<uint64_t,uint64_t>(sendFunction, recvFunction,
				      &set_3D_array_integral_values<uint64_t>,
				      dim1, dim2, dim3, "_ui64");

  sendFunction = &SmartSimClient::put_exact_key_array_uint32;
  recvFunction = &SmartSimClient::get_exact_key_array_uint32;
  put_get_3D_array<uint32_t,uint32_t>(sendFunction, recvFunction,
				      &set_3D_array_integral_values<uint32_t>,
				      dim1, dim2, dim3, "_ui32");

  MPI_Finalize();

  if(rank==0)
    std::cout<<"Finished 3D put and get tests."<<std::endl;

  return 0;
}
