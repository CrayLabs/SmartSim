#include "string.h"
#include <sw/redis++/redis++.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "ss_protob_array.pb.h"
#include <mpi.h>
#include <google/protobuf/reflection.h>

class SmartSimClient;

class SmartSimClient
{
public:
  SmartSimClient();
  ~SmartSimClient();
  sw::redis::RedisCluster redis_cluster;
  void put_nd_array_double(const char* key, void* value, int* dims, int n_dims, bool fotran_array=false);
  void get_nd_array_double(const char* key, void* result, int* dims, int n_dims, bool fortran_array=false);
private:
  SmartSimPBArray::ArrayDouble protob_double;
  SmartSimPBArray::ArrayFloat protob_float;
  std::string _serialize_protobuff_double();
  void _add_nd_array_double_values(void* value, int* dims, int n_dims);
  void _place_nd_array_double_values(void* value, int* dims, int n_dims, int& proto_position);
  void _put_keydb_value_into_protobuff_double(const char* key, int n_values);
  void _clear_protobuff_double();
  void _clear_protobuff_float();
  void _put_to_keydb(const char* key, std::string& value);
  std::string _build_get_key(const char* key);
  std::string _build_put_key(const char* key);
  std::string _get_from_keydb(const char* key);
  std::string _get_ssdb();
};


// C-wrapped functions
typedef void *OpaqueObject;
extern "C" void *ssc_constructor();
extern "C" void put_nd_array_double_ssc(void* SmartSimClient_proto, const char *key, void *value, int **dimensions, int *ndims);
extern "C" void get_nd_array_double_ssc(void* SmartSimClient_proto, const char *key, void *value, int **dimensions, int *ndims);
