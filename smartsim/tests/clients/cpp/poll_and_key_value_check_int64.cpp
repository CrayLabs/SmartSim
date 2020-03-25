#include "client.h"
#include <mpi.h>

void test_poll_and_key_value_check_int64()
{
  SmartSimClient client;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string key = "scalar_test_rank_"+std::to_string(rank);

  int put_value = 2;
  client.put_scalar_int64(key.c_str(), put_value);

  int check_value = 2;
  if(!client.poll_key_and_check_scalar_int64(key.c_str(), check_value, 1000, 10))
    throw std::runtime_error("Key existence could not be verified with poll_and_check_scalar_int64()");

  check_value = 4;
  if(client.poll_key_and_check_scalar_int64(key.c_str(), check_value, 1000, 10))
    throw std::runtime_error("Incorrectly found key existence in database using  poll_and_check_scalar_int64()");

  if(rank == 0)
    std::cout<<"Finished poll_and_check_scalara_int64 test."<<std::endl;

  return;
}

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);

  test_poll_and_key_value_check_int64();

  MPI_Finalize();
    
  return 0;
}
