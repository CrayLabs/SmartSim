#include "client.h"
#include <mpi.h>

void test_1d_put_cpp(int dim1, std::string key_suffix="")
{
  SmartSimClient client;
  double* array = new double[dim1];
  double* result = new double[dim1];
  int rank;
  std::string key;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  for(int i = 0; i < dim1; i++)
    array[i] = i;

  int* dims = new int[1];
  dims[0] = dim1;
  
  key = "1d_test_rank_"+std::to_string(rank) + key_suffix;

  std::cout<<"Starting put!"<<std::endl<<std::flush;
  client.put_array_double(key.c_str(), array, dims, 1);

  std::cout<<"Finished put!"<<std::endl<<std::flush;
  if(!client.exists(key.c_str()))
    throw std::runtime_error("Key existence could not be verified with key_exists()");
  
  client.get_array_double(key.c_str(), result, dims, 1);

  for(int i = 0; i < dim1; i++) {
    if(!(result[i]==array[i]))
      throw std::runtime_error("The arrays don't match");
  }
  
  delete[] array;
  delete[] result;
  delete[] dims;

  MPI_Barrier(MPI_COMM_WORLD);

  if(rank==0)
    std::cout<<"Finished 1D put/get c++ test"<<std::endl;

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  test_1d_put_cpp(atoi(argv[1]));

  MPI_Finalize();
  
  if(rank==0)
    std::cout<<"Finished all tests"<<std::endl;
  
  return 0;
}
