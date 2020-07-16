#include "client.h"
#include <mpi.h>
#include "client_test_utils.h"

class ClientTester : public SmartSimClient{
public:
  ClientTester(bool cluster):SmartSimClient(cluster){}
  std::string get_key_prefix(){
    return this->_get_key_prefix;
  }
  std::string put_key_prefix(){
    return this->_put_key_prefix;
  }
};

void check_put_prefix(std::string exp_val) {

  ClientTester client(false);
  std::string put_prefix = client.put_key_prefix();
  std::cout<<"put prefix = "<<put_prefix<<std::endl;
  if(put_prefix.compare(exp_val)!=0) {
    std::cout<<"Expected put prefix: "<<exp_val<<
      std::endl<<std::flush;
    std::cout<<"Returned put prefix: "<<put_prefix<<
      std::endl<<std::flush;
    throw std::runtime_error("The put prefix did not "\
			     "match the expected value");
  }
  return;
}

void check_get_prefix(std::string exp_val) {

  ClientTester client(false);
  std::cout<<"raw SSKEYIN = "<<std::getenv("SSKEYIN")<<std::endl;
  std::string get_prefix = client.get_key_prefix();
  std::cout<<"get prefix = "<<get_prefix<<std::endl;
  if(get_prefix.compare(exp_val)!=0) {
    std::cout<<"Expected get prefix: "<<exp_val<<
      std::endl<<std::flush;
    std::cout<<"Returned get prefix: "<<get_prefix<<
      std::endl<<std::flush;
    throw std::runtime_error("The get prefix did not "\
			     "match the expected value");
  }
  return;
}


int main(int argc, char* argv[]) {
  /* This will test the environment variable parsing.  Because the
     client cannot be created without the valid SSDB env
     environment variable, we will run this via the Python experiment
     API but overwrite environment variables to be sure
     we are testing and comparing the correct values.
  */

  MPI_Init(&argc, &argv);
  const char* env_name_out = "SSKEYOUT";
  const char* env_name_in = "SSKEYIN";
  const char* exp_prefix = "prefix_1";

  // SSKEYIN and SSKEYOUT one prefix
  const char* env_value_1 = "prefix_1";
  setenv(env_name_out, env_value_1, 1);
  setenv(env_name_in, env_value_1, 1);
  check_put_prefix(exp_prefix);
  check_get_prefix(exp_prefix);

  // SSKEYIN and SSKEYOUT multi-prefix
  const char* env_value_2 = "prefix_1;prefix_2;prefix_3;prefix_4";
  setenv(env_name_in, env_value_2, 1);
  check_put_prefix(exp_prefix);
  check_get_prefix(exp_prefix);

  // SSKEYIN and SSKEYOUT multi-prefix
  const char* env_value_3 = "prefix_1;prefix_2;prefix_3;";
  setenv(env_name_in, env_value_3, 1);
  check_put_prefix(exp_prefix);
  check_get_prefix(exp_prefix);

  // SSKEYIN and SSKEYOUT three prefix with
  // unexpected ; at the beginning
  const char* env_value_4 = ";prefix_1;prefix_2;prefix_3;";
  setenv(env_name_in, env_value_4, 1);
  check_put_prefix(exp_prefix);
  check_get_prefix(exp_prefix);

  return 0;
}
