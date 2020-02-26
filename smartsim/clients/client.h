#include "string.h"
#include <sw/redis++/redis++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "ss_protob_array.pb.h"
#include <mpi.h>

class SmartSimClient;

class SmartSimClient
{
public:
  SmartSimClient();
  ~SmartSimClient();
  sw::redis::RedisCluster redis_cluster;
  void put_1d_array_double(const char* key, double* value, const int nx, const int x_start=0);
  void put_2d_array_double(const char* key, double** value, const int nx, const int ny, const int x_start=0, const int y_start=0);
  void put_3d_array_double(const char* key, double*** value, const int nx, const int ny, const int nz, const int x_Start=0, const int y_start=0, const int z_start=0);
  void put_nd_array_double(const char* key, void* value, int* dims, int n_dims);
  void get_1d_array_double(const char* key, double* result, const int nx, const int x_start=0);
  void get_2d_array_double(const char* key, double** result, const int nx, const int ny, const int x_start=0, const int y_start=0);
  void get_3d_array_double(const char* key, double*** result, const int nx, const int ny, const int nz, const int x_start=0, const int y_start=0, const int z_start=0);
  void get_nd_array_double(const char* key, void* result, int* dims, int n_dims);
private:
  SmartSimPBArray::ArrayDouble protob_double;
  SmartSimPBArray::ArrayFloat protob_float;
  std::string _serialize_protobuff_double();
  void _add_nd_array_double_values(void* value, int* dims, int n_dims);
  void _place_nd_array_double_values(void* value, int* dims, int n_dims, int& proto_position);
  void _put_keydb_value_into_protobuff_double(const char* key, int n_values);
  void _clear_protobuff_double();
  void _clear_protobuff_float();
  void _put_to_keydb(const char*& key, std::string& value);
  std::string _get_from_keydb(const char*& key);
  std::string _get_hostname();
  std::string _get_ssdb_port();
  std::string _get_ssdb();
};


// C-wrapped functions

typedef void *OpaqueObject;
extern "C" void *ssc_constructor();
extern "C" void ssc_put_3d_array_double(OpaqueObject SmartSimClient_proto, int* keylen, const char* key, double* value, int *x_start,
                                         int *y_start, int *z_start, int *x_end, int *y_end, int *z_end,  bool *f_arrays);
