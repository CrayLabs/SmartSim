#include "client.h"
#include <mpi.h>

void test_2d_put_cpp(int dim1, int dim2, std::string key_suffix="")
{
  SmartSimClient client;
  int rank;
  std::string key;

  double **arr = (double **)malloc(dim1 * sizeof(double *));
  for (int i=0; i<dim1; i++) 
    arr[i] = (double *)malloc(dim2 * sizeof(double));

  int c = 0;
  for(int i = 0; i < dim1; i++)
    for(int j = 0; j < dim2; j++)
      arr[i][j] = c++;

  int* dims = new int[2];
  dims[0] = dim1;
  dims[1] = dim2;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  key = "2d_test_rank_"+std::to_string(rank) + key_suffix;

  client.put_array_double(key.c_str(), arr, dims, 2);

  MPI_Barrier(MPI_COMM_WORLD);
  
  if(!client.exists(key.c_str()))
    throw std::runtime_error("Key existence could not be verified with key_exists()");
  
  double** result = (double **)malloc(dim1 * sizeof(double *));
  for(int i=0; i<dim1; i++)
    result[i] = (double *)malloc(dim2 * sizeof(double));

  client.get_array_double(key.c_str(), result, dims, 2);
  
  for(int i=0; i<dim1; i++)
    for(int j=0; j<dim2; j++) {
      if(!(result[i][j] == arr[i][j])) {
	std::cout<<"result["<<i<<"]["<<j<<"] = "<<result[i][j]<<" arr = "<<arr[i][j]<<std::endl;
	throw std::runtime_error("The results do not match for the 2d put and get test!");
      }
    }
  
  MPI_Barrier(MPI_COMM_WORLD);

  for(int i=0; i<dim1; i++) {
    delete[] arr[i];
    delete[] result[i];
  }
  delete[] arr;
  delete[] result;
  delete[] dims;
  
  if(rank==0)
    std::cout<<"Finished 2D put/get c++ test"<<std::endl;

  return;
  
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  test_2d_put_cpp(atoi(argv[1]), atoi(argv[1]));
  MPI_Finalize();
  
  return 0;
}
