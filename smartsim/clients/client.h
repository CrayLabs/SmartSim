#ifndef SMARTSIM_CPP_CLIENT_H
#define SMARTSIM_CPP_CLIENT_H
#ifdef __cplusplus
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
///@file
///\brief The C++ SmartSimClient class
class SmartSimClient;

class SmartSimClient
{
public:
  //! SmartSim client constructor
  SmartSimClient(
      bool cluster /*!< Flag to indicate if a database cluster is being used*/,
      bool fortran_array = false /*!< Flag to indicate if fortran arrays are being used*/
  );
  //! SmartSim client destructor
  ~SmartSimClient();

  //! Put an array of type double into the database
  void put_array_double(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type float into the database
  void put_array_float(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type int64_t into the database
  void put_array_int64(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type int32_t into the database
  void put_array_int32(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type uint64_t into the database
  void put_array_uint64(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type uint32_t into the database
  void put_array_uint32(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type double from the database
  void get_array_double(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type float from the database
  void get_array_float(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type int64_t from the database
  void get_array_int64(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type int32_t from the database
  void get_array_int32(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type uint64_t from the database
  void get_array_uint64(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type uint32_t from the database
  void get_array_uint32(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put a scalar of type double into the database
  void put_scalar_double(
      const char* key     /*!< Identifier for this object in the database */,
      double value        /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type float into the database
  void put_scalar_float(
      const char* key     /*!< Identifier for this object in the database */,
      float value         /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type int64_t into the database
  void put_scalar_int64(
      const char* key     /*!< Identifier for this object in the database */,
      int64_t value       /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type int32_t into the database
  void put_scalar_int32(
      const char* key     /*!< Identifier for this object in the database */,
      int32_t value       /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type uint64_t into the database
  void put_scalar_uint64(
      const char* key     /*!< Identifier for this object in the database */,
      uint64_t value      /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type uint32_t into the database
  void put_scalar_uint32(
      const char* key     /*!< Identifier for this object in the database */,
      uint32_t value      /*!< Scalar value to store in the database */
  );

  //! Get an scalar of type double from the database
  double get_scalar_double(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type float from the database
  float get_scalar_float(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type int64_t from the database
  int64_t get_scalar_int64(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type int32_t from the database
  int32_t get_scalar_int32(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type uint64_t from the database
  uint64_t get_scalar_uint64(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type uint32_t from the database
  uint32_t get_scalar_uint32(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Check if a key exists in the database
  bool key_exists(
      const char* key       /*!< Identifier to check for in the database */
  );

  //! Poll the database until a specified key exists.
  bool poll_key(
      const char* key      /*!< Identifier for this object in the database */,
      int poll_frequency_ms=1000  /*!< How often to check the database in milliseconds */,
      int num_tries=-1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_key_and_check_scalar_double(
      const char* key        /*!< Identifier for this object in the database */,
      double value           /*!< Scalar value against which to check */,
      int poll_frequency_ms=1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_key_and_check_scalar_float(
      const char* key        /*!< Identifier for this object in the database */,
      float value            /*!< Scalar value against which to check */,
      int poll_frequency_ms=1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_key_and_check_scalar_int64(
      const char* key        /*!< Identifier for this object in the database */,
      int64_t value          /*!< Scalar value against which to check */,
      int poll_frequency_ms=1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_key_and_check_scalar_int32(
      const char* key        /*!< Identifier for this object in the database */,
      int32_t value          /*!< Scalar value against which to check */,
      int poll_frequency_ms=1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_key_and_check_scalar_uint64(
      const char* key        /*!< Identifier for this object in the database */,
      uint64_t value          /*!< Scalar value against which to check */,
      int poll_frequency_ms=1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_key_and_check_scalar_uint32(
      const char* key        /*!< Identifier for this object in the database */,
      uint32_t value          /*!< Scalar value against which to check */,
      int poll_frequency_ms=1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Put an array of type double into the database without key prefixing
  void put_exact_key_array_double(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type float into the database without key prefixing
  void put_exact_key_array_float(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type int64_t into the database without key prefixing
  void put_exact_key_array_int64(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type int32_t into the database without key prefixing
  void put_exact_key_array_int32(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type uint64_t into the database without key prefixing
  void put_exact_key_array_uint64(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Put an array of type uint32_t into the database without key prefixing
  void put_exact_key_array_uint32(
      const char* key     /*!< Identifier for this object in the database */,
      void* value         /*!< Array to store in the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type double from the database without key prefixing
  void get_exact_key_array_double(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type float from the database without key prefixing
  void get_exact_key_array_float(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type int64_t from the database without key prefixing
  void get_exact_key_array_int64(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type int32_t from the database without key prefixing
  void get_exact_key_array_int32(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type uint64_t from the database without key prefixing
  void get_exact_key_array_uint64(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an array of type uint32_t from the database without key prefixing
  void get_exact_key_array_uint32(
      const char* key     /*!< Identifier for this object in th database */,
      void* result        /*!< Array to fill with data from the database */,
      int* dims           /*!< Length along each dimension of the array */,
      int n_dims          /*!< Number of dimensions of the array */
  );

  //! Get an scalar of type double from the database without key prefixing
  void put_exact_key_scalar_double(
      const char* key     /*!< Identifier for this object in the database */,
      double value        /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type float into the database without key prefixing
  void put_exact_key_scalar_float(
      const char* key     /*!< Identifier for this object in the database */,
      float value         /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type int64_t into the database without key prefixing
  void put_exact_key_scalar_int64(
      const char* key     /*!< Identifier for this object in the database */,
      int64_t value       /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type int32_t into the database without key prefixing
  void put_exact_key_scalar_int32(
      const char* key     /*!< Identifier for this object in the database */,
      int32_t value       /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type uint64_t into the database without key prefixing
  void put_exact_key_scalar_uint64(
      const char* key     /*!< Identifier for this object in the database */,
      uint64_t value      /*!< Scalar value to store in the database */
  );

  //! Put a scalar of type uint32_t into the database without key prefixing
  void put_exact_key_scalar_uint32(
      const char* key     /*!< Identifier for this object in the database */,
      uint32_t value      /*!< Scalar value to store in the database */
  );

  //! Get an scalar of type double from the database without key prefixing
  double   get_exact_key_scalar_double(
     const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type float from the database without key prefixing
  float    get_exact_key_scalar_float(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type int64_t from the database without key prefixing
  int64_t  get_exact_key_scalar_int64(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type int32_t from the database without key prefixing
  int32_t  get_exact_key_scalar_int32(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type uint64_t from the database without key prefixing
  uint64_t get_exact_key_scalar_uint64(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Get an scalar of type uint32_t from the database without key prefixing
  uint32_t get_exact_key_scalar_uint32(
      const char* key      /*!< Identifier for this object in the database */
  );

  //! Check if a key exists in the database
  bool exact_key_exists(
      const char* key       /*!< Identifier to check for in the database */
  );

  //! Poll the database until a specified key exists.
  bool poll_exact_key(
      const char* key      /*!< Identifier for this object in the database */,
      int poll_frequency_ms=1000  /*!< How often to check the database in milliseconds */,
      int num_tries=-1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_exact_key_and_check_scalar_double(
      const char* key        /*!< Identifier for this object in the database */,
      double value           /*!< Scalar value against which to check */,
      int poll_frequency_ms = 1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_exact_key_and_check_scalar_float(
      const char* key        /*!< Identifier for this object in the database */,
      float value            /*!< Scalar value against which to check */,
      int poll_frequency_ms = 1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_exact_key_and_check_scalar_int64(
      const char* key        /*!< Identifier for this object in the database */,
      int64_t value          /*!< Scalar value against which to check */,
      int poll_frequency_ms = 1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_exact_key_and_check_scalar_int32(
      const char* key        /*!< Identifier for this object in the database */,
      int32_t value          /*!< Scalar value against which to check */,
      int poll_frequency_ms = 1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_exact_key_and_check_scalar_uint64(
      const char* key        /*!< Identifier for this object in the database */,
      uint64_t value          /*!< Scalar value against which to check */,
      int poll_frequency_ms = 1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

  //! Poll the database for a key and check its value
  bool poll_exact_key_and_check_scalar_uint32(
      const char* key        /*!< Identifier for this object in the database */,
      uint32_t value          /*!< Scalar value against which to check */,
      int poll_frequency_ms = 1000  /*!< How often to check the database in milliseconds */,
      int num_tries = -1     /*!< Number of times to check the database */
  );

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
  void set_data_source(const char* source_id);
  const char* query_get_prefix();

private:
  sw::redis::RedisCluster* redis_cluster;
  sw::redis::Redis* redis;
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
extern "C" void* initialize_c_client( bool cluster );
extern "C" void* initialize_fortran_client( bool cluster );
#else
void* initialize_c_client( int cluster );
void* initialize_fortran_client( int cluster );
#endif

#endif //SMARTSIM_CPP_CLIENT_H
