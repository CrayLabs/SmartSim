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

void SmartSimClient::get_1d_array_double(const char* key, double* result, const int nx, const int x_start)
{
  _put_keydb_value_into_protobuff_double(key, nx);

  for (int i = x_start; i < nx + x_start; i++)
    result[i] = protob_double.data(i);

  _clear_protobuff_double();

  return;
}

void SmartSimClient::get_2d_array_double(const char* key, double** result, const int nx, const int ny, const int x_start, const int y_start)
{

  _put_keydb_value_into_protobuff_double(key, nx*ny);

  int m = 0;
  for (int i = x_start; i < x_start + nx; i++)
    for(int j = y_start; j < y_start + ny; j++)
      result[i][j] = protob_double.data(m++);

  _clear_protobuff_double();

  return;
}

void SmartSimClient::get_3d_array_double(const char* key, double*** result, const int nx, const int ny, const int nz, const int x_start, const int y_start, const int z_start)
{

  _put_keydb_value_into_protobuff_double(key, nx*ny*nz);

  int m = 0;
  for (int i = x_start; i < x_start + nx; i++)
    for(int j = y_start; j < y_start + ny; j++)
      for(int k = z_start; k < z_start + nz; k++)
	result[i][j][k] = protob_double.data(m++);

  _clear_protobuff_double();

  return;
}

void SmartSimClient::get_nd_array_double(const char* key, void* result, int* dims, int n_dims)
{

  if(n_dims<=0)
    return;

  int n_values = 1;
  for(int i = 0; i < n_dims; i++)
    n_values *= dims[i];
  
  _put_keydb_value_into_protobuff_double(key, n_values);

  int proto_position = 0;
  _place_nd_array_double_values(result, dims, n_dims, proto_position);

  _clear_protobuff_double();

  return;
}

void SmartSimClient::put_1d_array_double(const char* key, double* value, const int nx, const int x_start)
{
  protob_double.add_dimension(nx);

  for(int i = 0; i < nx; i++)
    protob_double.add_data(value[i]);

  std::string output = _serialize_protobuff_double();
  
  _clear_protobuff_double();

  _put_to_keydb(key, output);
  
  return;
}

void SmartSimClient::put_2d_array_double(const char* key, double** value, const int nx, const int ny, const int x_start, const int y_start)
{
  protob_double.add_dimension(nx);
  protob_double.add_dimension(ny);

  for(int i = x_start; i < x_start + nx; i++)
    for(int j = y_start; j < y_start + ny; j++)
      protob_double.add_data(value[i][j]);

  std::string output = _serialize_protobuff_double();

  _clear_protobuff_double();

  _put_to_keydb(key, output);
}

void SmartSimClient::put_3d_array_double(const char* key, double*** value, const int nx, const int ny, const int nz, const int x_start, const int y_start, const int z_start)
{
  protob_double.add_dimension(nx);
  protob_double.add_dimension(ny);
  protob_double.add_dimension(nz);

  for(int i = x_start; i < x_start + nx; i++)
    for(int j = y_start; j < y_start + ny; j++)
      for(int k = z_start; k < z_start + nz; k++)
	protob_double.add_data(value[i][j][k]);

  std::string output = _serialize_protobuff_double();

  _clear_protobuff_double();

  _put_to_keydb(key, output);
}

void SmartSimClient::put_nd_array_double(const char* key, void* value, int* dims, int n_dims)
{
  for(int i = 0; i < n_dims; i++)
    protob_double.add_dimension(dims[i]);

  _add_nd_array_double_values(value, dims, n_dims);

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
    for(int i = 0; i < dims[0]; i++)
      protob_double.add_data(dbl_array[i]);
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

std::string SmartSimClient::_get_hostname()
{
  char* hostname = std::getenv("SSDB");
  if (hostname) {
    return std::string(hostname);
  }
  else {
    throw std::runtime_error("Database not found!");
      }
}

std::string SmartSimClient::_get_ssdb_port()
{
  char* port = std::getenv("SSDBPORT");
  if (port) {
    return std::string(port);
  }
  else {
    throw std::runtime_error("Database port not found!");
  }
}

void SmartSimClient::_put_to_keydb(const char*& key, std::string& value)
{
  bool success = redis_cluster.set(key, value);

  if(!success)
    throw std::runtime_error("KeyDB failed to receive key: " + std::string(key));
}

std::string SmartSimClient::_get_from_keydb(const char*& key)
{
  sw::redis::OptionalString value = redis_cluster.get(key);
  
  if(!value)
    throw std::runtime_error("The key " + std::string(key) + "could not be retrieved from the database");

  return value.value();
}

std::string SmartSimClient::_get_ssdb()
{
  std::string hostname = _get_hostname();
  std::string port = _get_ssdb_port();
  std::string ssdb = "tcp://" + hostname +":"+ port;
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
  std::cout<<"Smart sim client address is "<<s<<std::endl;
  void* c_ptr = (void*)s;
  std::cout<<"C ptr address after construction"<<c_ptr<<std::endl;
  return c_ptr;
}

extern "C" void* ssc_constructor()
{
  return new SmartSimClient;
}


extern "C" void ssc_put_3d_array_double(void* SmartSimClient_proto, int* keylen, const char* key, double* value, int *x_start, int *y_start, int *z_start, int *x_end, int *y_end, int *z_end,  bool *f_arrays)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  int dims[3];
  dims[0] = *x_end - *x_start;
  dims[1] = *y_end - *y_start;
  dims[2] = *z_end - *z_start;
  s->put_nd_array_double(key, value, dims, 3);
}
