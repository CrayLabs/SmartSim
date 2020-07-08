#ifndef SMARTSIM_CPP_CLIENT_H
#define SMARTSIM_CPP_CLIENT_H
#include "string.h"
#include "stdlib.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <sw/redis++/redis++.h>
#include "smartsim_protobuf.pb.h"
#include <google/protobuf/reflection.h>
#include <google/protobuf/stubs/port.h>

class SmartSimClient;

class SmartSimClient
{
public:
  SmartSimClient(bool fortran_array = false);
  ~SmartSimClient();
  sw::redis::RedisCluster redis_cluster;

  void set_data_source(const char* source_id);
  const char* query_get_prefix();
  void put_array_double(const char* key, void* value, int* dims, int n_dims);
  void put_array_float(const char* key, void* value, int* dims, int n_dims);
  void put_array_int64(const char* key, void* value, int* dims, int n_dims);
  void put_array_int32(const char* key, void* value, int* dims, int n_dims);
  void put_array_uint64(const char* key, void* value, int* dims, int n_dims);
  void put_array_uint32(const char* key, void* value, int* dims, int n_dims);

  void get_array_double(const char* key, void* result, int* dims, int n_dims);
  void get_array_float(const char* key, void* result, int* dims, int n_dims);
  void get_array_int64(const char* key, void* result, int* dims, int n_dims);
  void get_array_int32(const char* key, void* result, int* dims, int n_dims);
  void get_array_uint64(const char* key, void* result, int* dims, int n_dims);
  void get_array_uint32(const char* key, void* result, int* dims, int n_dims);

  void put_scalar_double(const char* key, double value);
  void put_scalar_float(const char* key, float value);
  void put_scalar_int64(const char* key, int64_t value);
  void put_scalar_int32(const char* key, int32_t value);
  void put_scalar_uint64(const char* key, uint64_t value);
  void put_scalar_uint32(const char* key, uint32_t value);

  double   get_scalar_double(const char* key);
  float    get_scalar_float(const char* key);
  int64_t  get_scalar_int64(const char* key);
  int32_t  get_scalar_int32(const char* key);
  uint64_t get_scalar_uint64(const char* key);
  uint32_t get_scalar_uint32(const char* key);

  bool key_exists(const char* key);
  bool poll_key(const char* key, int poll_frequency_ms=1000, int num_tries=-1);
  bool poll_key_and_check_scalar_double(const char* key, double value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_key_and_check_scalar_float(const char* key, float value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_key_and_check_scalar_int64(const char* key, int64_t value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_key_and_check_scalar_int32(const char* key, int32_t value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_key_and_check_scalar_uint64(const char* key, uint64_t value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_key_and_check_scalar_uint32(const char* key, uint32_t value, int poll_frequency_ms = 1000, int num_tries = -1);

  void put_exact_key_array_double(const char* key, void* value, int* dims, int n_dims);
  void put_exact_key_array_float(const char* key, void* value, int* dims, int n_dims);
  void put_exact_key_array_int64(const char* key, void* value, int* dims, int n_dims);
  void put_exact_key_array_int32(const char* key, void* value, int* dims, int n_dims);
  void put_exact_key_array_uint64(const char* key, void* value, int* dims, int n_dims);
  void put_exact_key_array_uint32(const char* key, void* value, int* dims, int n_dims);

  void get_exact_key_array_double(const char* key, void* result, int* dims, int n_dims);
  void get_exact_key_array_float(const char* key, void* result, int* dims, int n_dims);
  void get_exact_key_array_int64(const char* key, void* result, int* dims, int n_dims);
  void get_exact_key_array_int32(const char* key, void* result, int* dims, int n_dims);
  void get_exact_key_array_uint64(const char* key, void* result, int* dims, int n_dims);
  void get_exact_key_array_uint32(const char* key, void* result, int* dims, int n_dims);

  void put_exact_key_scalar_double(const char* key, double value);
  void put_exact_key_scalar_float(const char* key, float value);
  void put_exact_key_scalar_int64(const char* key, int64_t value);
  void put_exact_key_scalar_int32(const char* key, int32_t value);
  void put_exact_key_scalar_uint64(const char* key, uint64_t value);
  void put_exact_key_scalar_uint32(const char* key, uint32_t value);

  double   get_exact_key_scalar_double(const char* key);
  float    get_exact_key_scalar_float(const char* key);
  int64_t  get_exact_key_scalar_int64(const char* key);
  int32_t  get_exact_key_scalar_int32(const char* key);
  uint64_t get_exact_key_scalar_uint64(const char* key);
  uint32_t get_exact_key_scalar_uint32(const char* key);

  bool exact_key_exists(const char* key);
  bool poll_exact_key(const char* key, int poll_frequency_ms=1000, int num_tries=-1);
  bool poll_exact_key_and_check_scalar_double(const char* key, double value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_exact_key_and_check_scalar_float(const char* key, float value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_exact_key_and_check_scalar_int64(const char* key, int64_t value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_exact_key_and_check_scalar_int32(const char* key, int32_t value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_exact_key_and_check_scalar_uint64(const char* key, uint64_t value, int poll_frequency_ms = 1000, int num_tries = -1);
  bool poll_exact_key_and_check_scalar_uint32(const char* key, uint32_t value, int poll_frequency_ms = 1000, int num_tries = -1);

protected:
  bool _fortran_array;
  void _set_prefixes_from_env();
  std::vector<std::string> _get_key_prefixes;
  std::string _put_key_prefix;
  std::string _get_key_prefix;
  std::string _build_get_key(const char* key);
  std::string _build_put_key(const char* key);
  std::string _get_from_keydb(const char* key);
  std::string _get_ssdb();
  
private:
  SmartSimProtobuf::ScalarDouble protob_scalar_double;
  SmartSimProtobuf::ScalarFloat protob_scalar_float;
  SmartSimProtobuf::ScalarSInt64 protob_scalar_int64;
  SmartSimProtobuf::ScalarSInt32 protob_scalar_int32;
  SmartSimProtobuf::ScalarUInt64 protob_scalar_uint64;
  SmartSimProtobuf::ScalarUInt32 protob_scalar_uint32;

  std::string _serialize_protobuff(google::protobuf::Message* pb_message);
  void _put_keydb_value_into_protobuff(google::protobuf::Message* pb_message, const char* key, int n_values);
  void _clear_protobuff(google::protobuf::Message* pb_message);
  void _put_to_keydb(const char* key, std::string& value);

  template <class T>
    T _get_scalar(google::protobuf::Message* pb_message, const char* key, bool add_prefix=true);
  template <class T>
    void _put_scalar(google::protobuf::Message* pb_message, const char* key, T value, bool add_prefix=true);
  template <class T, class U>
    void _put_array(const char* key, void* value, int* dims, int n_dims, bool add_prefix=true);
  template <class T>
    void _serialize_array(google::protobuf::Message* pb_message, std::string& buff, void* value, int* dims, int n_dims);
  template <class T, class U>
    void _get_array(const char* key, void* result, int* dims, int n_dims, bool add_prefix=true);
  template <class T>
    void _add_array_values(const google::protobuf::MutableRepeatedFieldRef<T>& pb_repeated_field, void* value, int* dims, int n_dims);
  template <class T>
    void _place_array_values(const google::protobuf::MutableRepeatedFieldRef<T>& pb_repeated_field, void* value, int* dims, int n_dims, int& proto_position);
  template <class T>
    bool _poll_key_and_check_scalar(const char* key, T value, int poll_frequency_ms, int num_tries, bool add_prefix=true);

};

extern "C" void* initialize_c_client( );
extern "C" void* initialize_fortran_client( );

#endif //SMARTSIM_CPP_CLIENT_H
