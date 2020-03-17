#include "client.h"

SmartSimClient::SmartSimClient() :
    redis_cluster(_get_ssdb())
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  return;
}

SmartSimClient::~SmartSimClient()
{
}

void SmartSimClient::get_nd_array_double(const char* key, void* result, int* dims, int n_dims, bool fortran_array)
{

  if(n_dims<=0)
    return;

  int n_values = 1;
  for(int i = 0; i < n_dims; i++)
    n_values *= dims[i];

  _put_keydb_value_into_protobuff_double(key, n_values);

  // If it is a fortran array, reset dims to
  // point to dynamically allocated dimension of 1
  // so that _place_nd_array can be used.
  if (fortran_array) {
    n_dims = 1;
    dims = new int[1];
    dims[0] = n_values;
  }

  int proto_position = 0;
  _place_nd_array_double_values(result, dims, n_dims, proto_position);

  if(fortran_array)
    delete[] dims;

  _clear_protobuff_double();

  return;
}

void SmartSimClient::put_nd_array_double(const char* key, void* value, int* dims, int n_dims, bool fortran_array)
{
  /*
  const google::protobuf::Descriptor* descriptor = protob_double.GetDescriptor();
  const google::protobuf::FieldDescriptor* data_field = descriptor->FindFieldByName("data");
  const google::protobuf::Reflection* message_reflection = protob_double.GetReflection();
  */

  for(int i = 0; i < n_dims; i++)
    protob_double.add_dimension(dims[i]);

  if(fortran_array) {
    
      int n_values = 1;
      for(int i = 0; i < n_dims; i++)
	n_values *= dims[i];

      dims = new int[1];
      dims[0] = n_values;
      
      n_dims = 1;
  }
  
  _add_nd_array_double_values(value, dims, n_dims);

  if(fortran_array) {
    delete[] dims;
  }
  
  std::string output = _serialize_protobuff_double();

  _clear_protobuff_double();

  _put_to_keydb(key, output);
  
  return;
}

void SmartSimClient::_add_nd_array_double_values(void* value, int* dims, int n_dims)
{
  if(n_dims > 1) {
    double** current = (double**) value;
    for(int i = 0; i < dims[0]; i++) {
      _add_nd_array_double_values(*current, &dims[1], n_dims-1);
      current++;
    }
  }
  else {
    double* dbl_array = (double*)value;
    for(int i = 0; i < dims[0]; i++){
      protob_double.add_data(dbl_array[i]);
    }
  }
  return;
}

void SmartSimClient::_place_nd_array_double_values(void* value, int* dims, int n_dims, int& proto_position)
{
  if(n_dims > 1) {
    double** current = (double**) value;
    for(int i = 0; i < dims[0]; i++) {
      _place_nd_array_double_values(*current, &dims[1], n_dims-1, proto_position);
      current++;
    }
  }
  else {
    double* dbl_array = (double*)value;
    for(int i = 0; i < dims[0]; i++)
      dbl_array[i] = protob_double.data(proto_position++);
  }
  return;
}

void SmartSimClient::_put_to_keydb(const char* key, std::string& value)
{
  std::string prefixed_key = _build_put_key(key);

  int n_trials = 5;
  bool success = false;
  
  while(n_trials > 0) {
    try {
      success = redis_cluster.set(prefixed_key.c_str(), value);
      n_trials = -1;
    }
    catch (sw::redis::IoError& e) {
      n_trials--;
      std::cout<<"WARNING: Caught redis IOError: "<<e.what()<<std::endl;
      std::cout<<"WARNING: Could not set key "<<prefixed_key<<" in database. "<<n_trials<<" more trials will be made."<<std::endl;
    }
  }
  if(n_trials == 0)
    throw std::runtime_error("Could not set "+prefixed_key+" in database due to redis IOError.");

  if(!success)
    throw std::runtime_error("KeyDB failed to receive key: " + std::string(key));

  return;
}

std::string SmartSimClient::_build_put_key(const char* key)
{
  //This function builds the key that it will be used
  //for the put value.  The key is SSNAME + _ + key

  std::string prefix(std::getenv("SSNAME"));
  std::string suffix(key);
  std::string prefixed_key = prefix + '_' + suffix;

  return prefixed_key;
}

std::string SmartSimClient::_build_get_key(const char* key)
{
  //This function builds the key that it will be used
  //for the put value.  The key is SSDATAIN + _ + key

  std::string prefix(std::getenv("SSDATAIN"));
  std::string suffix(key);
  std::string prefixed_key = prefix + '_' + suffix;

  return prefixed_key;
}

std::string SmartSimClient::_get_from_keydb(const char* key)
{
  std::string prefixed_key = _build_get_key(key);

  int n_trials = 5;
  sw::redis::OptionalString value;

  while(n_trials > 0) {
    try {
      value = redis_cluster.get(prefixed_key.c_str());
      n_trials = -1;
    }
    catch (sw::redis::IoError& e) {
      n_trials--;
      std::cout<<"WARNING: Caught redis IOError: "<<e.what()<<std::endl;
      std::cout<<"WARNING: Could not get key "<<prefixed_key<<" from database. "<<n_trials<<" more trials will be made."<<std::endl;
    }
  }
  if(n_trials == 0)
    throw std::runtime_error("Could not retreive "+prefixed_key+" from database due to redis IOError.");

  if(!value)
    throw std::runtime_error("The key " + std::string(key) + "could not be retrieved from the database");

  return value.value();
}

std::string SmartSimClient::_get_ssdb()
{
  char* host_and_port = getenv("SSDB");
  if(host_and_port == NULL)
    throw std::runtime_error("The environment variable SSDB must be set to use the client.");
  
  std::string ssdb("tcp://");  
  ssdb.append(host_and_port);
  return ssdb;
}

std::string SmartSimClient::_serialize_protobuff_double()
{
  std::string buff;
  bool success = protob_double.SerializeToString(&buff);

  if(!success)
    throw std::runtime_error("Protobuf serialization failed");

  return buff;
}

void SmartSimClient::_put_keydb_value_into_protobuff_double(const char* key, int n_values)
{
  std::string value = _get_from_keydb(key);

  protob_double.ParseFromString(value);

  if(!(protob_double.data_size()==n_values))
    throw std::runtime_error("The protobuf array is not the same length as specified by n_values");
}

void SmartSimClient::_clear_protobuff_double()
{
  protob_double.clear_data();
  protob_double.clear_dimension();
}

void SmartSimClient::_clear_protobuff_float()
{
  protob_float.clear_data();
  protob_double.clear_dimension();
}

extern "C" void* GetObject() {
  SmartSimClient *s = new SmartSimClient();
  void* c_ptr = (void*)s;
  return c_ptr;
}

extern "C" void* ssc_constructor()
{
  return new SmartSimClient;
}

extern "C" void put_nd_array_double_ssc(void* SmartSimClient_proto, const char *key, void *value, int **dimensions, int *ndims)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  s->put_nd_array_double(key, value, *dimensions, *ndims, true);
}

extern "C" void get_nd_array_double_ssc(void* SmartSimClient_proto, const char *key, void *value, int **dimensions, int *ndims)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  s->get_nd_array_double(key, value, *dimensions, *ndims, true);
}
