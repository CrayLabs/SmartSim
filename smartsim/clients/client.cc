#include "client.h"

SmartSimClient::SmartSimClient(bool fortran_array) :
    redis_cluster(_get_ssdb())
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  this->_set_prefixes_from_env();
  this->_fortran_array = fortran_array;
  return;
}

SmartSimClient::~SmartSimClient()
{
}

void SmartSimClient::_set_prefixes_from_env()
{
  if (const char* keyout_p = std::getenv("SSKEYOUT"))
    this->_put_key_prefix = keyout_p;
  else
    this->_put_key_prefix.clear();

  if (const char* keyin_p = std::getenv("SSKEYIN")){
    std::string keyin_env_string(keyin_p);
    // Parse a delimited list of input prefixes
    char parse_char[] = ";";
    int start_pos = 0;
    int end_pos = 0;
    int str_len = keyin_env_string.length();

    for (int i=0; i<str_len; i++) {
      if ( keyin_env_string[i] == parse_char[0]) {
        end_pos = i-1;
        this->_get_key_prefixes.push_back(keyin_env_string.substr(start_pos,end_pos));
        start_pos = i+1;
      }
      else if (i == str_len-1)
        this->_get_key_prefixes.push_back(keyin_env_string.substr(start_pos,str_len-1));
    }
  if ( this->_get_key_prefixes.size() == 1)
    this->set_data_source( this->_get_key_prefixes[0].c_str() );
  }
}

void SmartSimClient::set_data_source(const char *source_id)
{
  bool valid_prefix = false;
  int num_prefix = _get_key_prefixes.size();
  int i = 0;
  for (i=0; i<num_prefix; i++)
  {
    if ( this->_get_key_prefixes[i].compare(source_id))
    {
      valid_prefix = true;
      break;
    }
  }

  if (valid_prefix)
    this->_get_key_prefix = this->_get_key_prefixes[i];
  else
	  throw std::runtime_error("Client error: requested key" + std::string(source_id) + "has not been registered");
}

const char* SmartSimClient::query_get_prefix()
{
  return this->_get_key_prefix.c_str();
}

// Put and get routines that query the database for the exact key
void SmartSimClient::put_exact_key_array_double(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArrayDouble,double>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_exact_key_array_float(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArrayFloat,float>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_exact_key_array_int64(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArraySInt64,int64_t>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_exact_key_array_int32(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArraySInt32,int32_t>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_exact_key_array_uint64(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArrayUInt64,uint64_t>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_exact_key_array_uint32(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArrayUInt32,uint32_t>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::get_exact_key_array_double(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArrayDouble, double>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_exact_key_array_float(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArrayFloat, float>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_exact_key_array_int64(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArraySInt64,int64_t>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_exact_key_array_int32(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArraySInt32,int32_t>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_exact_key_array_uint64(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArrayUInt64,uint64_t>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_exact_key_array_uint32(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArrayUInt32,uint32_t>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::put_exact_key_scalar_double(const char* key, double value)
{
  this->_put_scalar<double>(&protob_scalar_double, key, value, false);
  return;
}

void SmartSimClient::put_exact_key_scalar_float(const char* key, float value)
{
  this->_put_scalar<float>(&protob_scalar_float, key, value, false);
  return;
}

void SmartSimClient::put_exact_key_scalar_int64(const char* key, int64_t value)
{
  this->_put_scalar<int64_t>(&protob_scalar_int64, key, value, false);
  return;
}

void SmartSimClient::put_exact_key_scalar_int32(const char* key, int32_t value)
{
  this->_put_scalar<int32_t>(&protob_scalar_int32, key, value, false);
  return;
}

void SmartSimClient::put_exact_key_scalar_uint64(const char* key, uint64_t value)
{
  this->_put_scalar<uint64_t>(&protob_scalar_uint64, key, value, false);
  return;
}

void SmartSimClient::put_exact_key_scalar_uint32(const char* key, uint32_t value)
{
  this->_put_scalar<uint32_t>(&protob_scalar_uint32, key, value, false);
  return;
}

double SmartSimClient::get_exact_key_scalar_double(const char* key)
{
  return this->_get_scalar<double>(&protob_scalar_double, key, false);
}

float SmartSimClient::get_exact_key_scalar_float(const char* key)
{
  return this->_get_scalar<float>(&protob_scalar_float, key, false);
}

int64_t SmartSimClient::get_exact_key_scalar_int64(const char* key)
{
  return this->_get_scalar<int64_t>(&protob_scalar_int64, key, false);
}

int32_t SmartSimClient::get_exact_key_scalar_int32(const char* key)
{
  return this->_get_scalar<int32_t>(&protob_scalar_int32, key, false);
}

uint64_t SmartSimClient::get_exact_key_scalar_uint64(const char* key)
{
  return this->_get_scalar<uint64_t>(&protob_scalar_uint64, key, false);
}

uint32_t SmartSimClient::get_exact_key_scalar_uint32(const char* key)
{
  return this->_get_scalar<uint32_t>(&protob_scalar_uint32, key, false);
}

// Put and get routines that potentially mangle the requested key with prefixing
void SmartSimClient::put_array_double(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArrayDouble,double>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_array_float(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArrayFloat,float>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_array_int64(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArraySInt64,int64_t>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_array_int32(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArraySInt32,int32_t>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_array_uint64(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArrayUInt64,uint64_t>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::put_array_uint32(const char* key, void* value, int* dims, int n_dims)
{
  this->_put_array<SmartSimProtobuf::ArrayUInt32,uint32_t>(key, value, dims, n_dims, false);
  return;
}

void SmartSimClient::get_array_double(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArrayDouble, double>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_array_float(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArrayFloat, float>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_array_int64(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArraySInt64,int64_t>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_array_int32(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArraySInt32,int32_t>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_array_uint64(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArrayUInt64,uint64_t>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::get_array_uint32(const char* key, void* result, int* dims, int n_dims)
{
  this->_get_array<SmartSimProtobuf::ArrayUInt32,uint32_t>(key, result, dims, n_dims, false);
  return;
}

void SmartSimClient::put_scalar_double(const char* key, double value)
{
  this->_put_scalar<double>(&protob_scalar_double, key, value, false);
  return;
}

void SmartSimClient::put_scalar_float(const char* key, float value)
{
  this->_put_scalar<float>(&protob_scalar_float, key, value, false);
  return;
}

void SmartSimClient::put_scalar_int64(const char* key, int64_t value)
{
  this->_put_scalar<int64_t>(&protob_scalar_int64, key, value, false);
  return;
}

void SmartSimClient::put_scalar_int32(const char* key, int32_t value)
{
  this->_put_scalar<int32_t>(&protob_scalar_int32, key, value, false);
  return;
}

void SmartSimClient::put_scalar_uint64(const char* key, uint64_t value)
{
  this->_put_scalar<uint64_t>(&protob_scalar_uint64, key, value, false);
  return;
}

void SmartSimClient::put_scalar_uint32(const char* key, uint32_t value)
{
  this->_put_scalar<uint32_t>(&protob_scalar_uint32, key, value, false);
  return;
}

double SmartSimClient::get_scalar_double(const char* key)
{
  return this->_get_scalar<double>(&protob_scalar_double, key, false);
}

float SmartSimClient::get_scalar_float(const char* key)
{
  return this->_get_scalar<float>(&protob_scalar_float, key, false);
}

int64_t SmartSimClient::get_scalar_int64(const char* key)
{
  return this->_get_scalar<int64_t>(&protob_scalar_int64, key, false);
}

int32_t SmartSimClient::get_scalar_int32(const char* key)
{
  return this->_get_scalar<int32_t>(&protob_scalar_int32, key, false);
}

uint64_t SmartSimClient::get_scalar_uint64(const char* key)
{
  return this->_get_scalar<uint64_t>(&protob_scalar_uint64, key, false);
}

uint32_t SmartSimClient::get_scalar_uint32(const char* key)
{
  return this->_get_scalar<uint32_t>(&protob_scalar_uint32, key, false);
}

// Routines for polling and checking scalars within the database by key
bool SmartSimClient::exact_key_exists(const char* key)
{
  return redis_cluster.exists(key);
}

bool SmartSimClient::poll_key(const char* key, int poll_frequency_ms, int num_tries)
{
  bool key_exists = false;

  while(!(num_tries==0)) {
    if(this->key_exists(key)) {
      key_exists = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
    if(num_tries>0)
      num_tries--;
  }

  if(key_exists)
    return true;
  else
    return false;
}

// These routines potentially modify keys by adding a prefix
bool SmartSimClient::key_exists(const char* key)
{
  return redis_cluster.exists(this->_build_get_key(key).c_str());
}

bool SmartSimClient::poll_key_and_check_scalar_double(const char* key, double value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<double>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_key_and_check_scalar_float(const char* key, float value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<float>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_key_and_check_scalar_int64(const char* key, int64_t value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<int64_t>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_key_and_check_scalar_int32(const char* key, int32_t value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<int32_t>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_key_and_check_scalar_uint64(const char* key, uint64_t value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<uint64_t>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_key_and_check_scalar_uint32(const char* key, uint32_t value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<uint32_t>(key, value, poll_frequency_ms, num_tries);
}
  // Poll for the exact key
bool SmartSimClient::poll_exact_key_and_check_scalar_double(const char* key, double value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<double>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_exact_key_and_check_scalar_float(const char* key, float value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<float>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_exact_key_and_check_scalar_int64(const char* key, int64_t value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<int64_t>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_exact_key_and_check_scalar_int32(const char* key, int32_t value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<int32_t>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_exact_key_and_check_scalar_uint64(const char* key, uint64_t value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<uint64_t>(key, value, poll_frequency_ms, num_tries);
}

bool SmartSimClient::poll_exact_key_and_check_scalar_uint32(const char* key, uint32_t value, int poll_frequency_ms, int num_tries)
{
  return this->_poll_key_and_check_scalar<uint32_t>(key, value, poll_frequency_ms, num_tries);
}

template <class T>
bool SmartSimClient::_poll_key_and_check_scalar(const char* key, T value, int poll_frequency_ms, int num_tries, bool add_prefix)
{
  bool matched_value = false;
  T current_value;

  std::string get_key = (add_prefix) ? this->_build_get_key(key) : key;

  while( !(num_tries==0) ) {
    if(this->key_exists(get_key.c_str())) {
      if(std::is_same<T, double>::value)
    	current_value = this->get_scalar_double(get_key.c_str());
      else if(std::is_same<T, float>::value)
	    current_value = this->get_scalar_float(get_key.c_str());
      else if(std::is_same<T, int64_t>::value)
    	current_value = this->get_scalar_int64(get_key.c_str());
      else if(std::is_same<T, int32_t>::value)
	    current_value = this->get_scalar_int32(get_key.c_str());
      else if(std::is_same<T, uint64_t>::value)
    	current_value = this->get_scalar_uint64(get_key.c_str());
      else if(std::is_same<T,uint32_t>::value)
    	current_value = this->get_scalar_uint32(get_key.c_str());
      else
    	throw std::runtime_error("Client Error: Unsupported scalar type.");

      if(value == current_value) {
        matched_value = true;
        num_tries = 0;
      }
    }

    if(!matched_value)
      std::this_thread::sleep_for(std::chrono::milliseconds(poll_frequency_ms));
    if(num_tries>0)
      num_tries--;
  }

  if(matched_value)
    return true;
  else
    return false;
}


// Low-level routines to serialize/deserialize via protobuf and set/get from database
template <class T>
void SmartSimClient::_put_scalar(google::protobuf::Message* pb_message, const char* key, T value, bool add_prefix)
{
  std::string put_key = (add_prefix) ? this->_build_put_key(key) : std::string(key);

  const google::protobuf::Reflection* refl = pb_message->GetReflection();
  const google::protobuf::FieldDescriptor* data_field = pb_message->GetDescriptor()->FindFieldByName("data");

  //Protobuf does not support general add function like they do for arrays, so a check on the type is needed
  if(std::is_same<T, double>::value)
    refl->SetDouble(pb_message, data_field, value);
  else if(std::is_same<T, float>::value)
    refl->SetFloat(pb_message, data_field, value);
  else if(std::is_same<T, int64_t>::value)
    refl->SetInt64(pb_message, data_field, value);
  else if(std::is_same<T, int32_t>::value)
    refl->SetInt32(pb_message, data_field, value);
  else if(std::is_same<T, uint64_t>::value)
    refl->SetUInt64(pb_message, data_field, value);
  else if(std::is_same<T,uint32_t>::value)
    refl->SetUInt32(pb_message, data_field, value);
  else
    throw std::runtime_error("Client Error: Unsupported scalar type.");

  std::string output = _serialize_protobuff(pb_message);
  _clear_protobuff(pb_message);
  _put_to_keydb(put_key.c_str(), output);

  return;
}

template<class T>
T SmartSimClient::_get_scalar(google::protobuf::Message* pb_message, const char* key, bool add_prefix)
{
  std::string get_key = (add_prefix) ? this->_build_get_key(key) : std::string(key);
  _put_keydb_value_into_protobuff(pb_message, get_key.c_str(), 1);

  const google::protobuf::Reflection* refl = pb_message->GetReflection();
  const google::protobuf::FieldDescriptor* data_field = pb_message->GetDescriptor()->FindFieldByName("data");

  T value;

  //Protobuf does not support general get function like they do for arrays, so a check on the type is needed
  if(std::is_same<T, double>::value)
    value = refl->GetDouble(*pb_message, data_field);
  else if(std::is_same<T, float>::value)
    value = refl->GetFloat(*pb_message, data_field);
  else if(std::is_same<T, int64_t>::value)
    value = refl->GetInt64(*pb_message, data_field);
  else if(std::is_same<T, int32_t>::value)
    value = refl->GetInt32(*pb_message, data_field);
  else if(std::is_same<T, uint64_t>::value)
    value = refl->GetUInt64(*pb_message, data_field);
  else if(std::is_same<T,uint32_t>::value)
    value = refl->GetUInt32(*pb_message, data_field);
  else
    throw std::runtime_error("Client Error: Unsupported scalar type.");

  _clear_protobuff(pb_message);

  return value;
}

template <class T, class U>
void SmartSimClient::_put_array(const char* key, void* value, int* dims, int n_dims, bool add_prefix)
{
  std::string buff;
  std::string put_key = (add_prefix) ? this->_build_put_key(key) : std::string(key);

  T* pb_message = new T();
  this->_serialize_array<U>(pb_message, buff, value, dims, n_dims);
  delete pb_message;
  this->_put_to_keydb(put_key.c_str(), buff);
  return;
}

template <class  T>
void SmartSimClient::_serialize_array(google::protobuf::Message* pb_message, std::string& buff, void* value, int* dims, int n_dims)
{
  const google::protobuf::Reflection* refl = pb_message->GetReflection();
  const google::protobuf::FieldDescriptor* dim_field = pb_message->GetDescriptor()->FindFieldByName("dimension");
  const google::protobuf::MutableRepeatedFieldRef<uint64_t> dimension = refl->GetMutableRepeatedFieldRef<uint64_t>(pb_message, dim_field);

  for(int i = 0; i < n_dims; i++)
    dimension.Add(dims[i]);

  // After saving the dims for later retrieval, point the pointer
  // dims to a new dynamically allocated array of ints of length 1
  // to indicate that a fotran array is contiguous in memory
  // and recursion is not necessary
  if(this->_fortran_array) {

      int n_values = 1;
      for(int i = 0; i < n_dims; i++)
	      n_values *= dims[i];

      dims = new int[1];
      dims[0] = n_values;

      n_dims = 1;
  }

  const google::protobuf::FieldDescriptor* data_field = pb_message->GetDescriptor()->FindFieldByName("data");
  const google::protobuf::MutableRepeatedFieldRef<T> data = refl->GetMutableRepeatedFieldRef<T>(pb_message, data_field);
  _add_array_values(data, value, dims, n_dims);

  if(this->_fortran_array)
    delete[] dims;

  buff = _serialize_protobuff(pb_message);

  return;
}

template <class T, class U>
void SmartSimClient::_get_array(const char* key, void* result, int* dims, int n_dims, bool add_prefix)
{

  T* pb_message = new T();

  if(n_dims<=0)
    return;

  int n_values = 1;
  for(int i = 0; i < n_dims; i++)
    n_values *= dims[i];

  std::string get_key = (add_prefix) ? this->_build_get_key(key) : std::string(key);
  _put_keydb_value_into_protobuff(pb_message, get_key.c_str(), n_values);

  // If it is a fortran array, reset dims to
  // point to dynamically allocated dimension of 1
  // so that _place_array can be used.
  if (this->_fortran_array) {
    n_dims = 1;
    dims = new int[1];
    dims[0] = n_values;
  }

  int proto_position = 0;

  const google::protobuf::Reflection* refl = pb_message->GetReflection();
  const google::protobuf::FieldDescriptor* data_field = pb_message->GetDescriptor()->FindFieldByName("data");
  const google::protobuf::MutableRepeatedFieldRef<U> data = refl->GetMutableRepeatedFieldRef<U>(pb_message, data_field);
  _place_array_values<U>(data, result, dims, n_dims, proto_position);

  if(this->_fortran_array)
    delete[] dims;

  delete pb_message;

  return;
}

template <class T>
void SmartSimClient::_add_array_values(const google::protobuf::MutableRepeatedFieldRef<T>& pb_repeated_field, void* value, int* dims, int n_dims)
{
  if(n_dims > 1) {
    T** current = (T**) value;
    for(int i = 0; i < dims[0]; i++) {
      _add_array_values(pb_repeated_field, *current, &dims[1], n_dims-1);
      current++;
    }
  }
  else {
    T* array = (T*)value;
    for(int i = 0; i < dims[0]; i++)
      pb_repeated_field.Add(array[i]);
  }
  return;
}

template <class T>
void SmartSimClient::_place_array_values(const google::protobuf::MutableRepeatedFieldRef<T>& pb_repeated_field, void* value, int* dims, int n_dims, int& proto_position)
{
  if(n_dims > 1) {
    T** current = (T**) value;
    for(int i = 0; i < dims[0]; i++) {
      _place_array_values(pb_repeated_field, *current, &dims[1], n_dims-1, proto_position);
      current++;
    }
  }
  else {
    T* array = (T*)value;
    for(int i = 0; i < dims[0]; i++)
      array[i] = pb_repeated_field.Get(proto_position++);
  }
  return;
}

void SmartSimClient::_put_to_keydb(const char* key, std::string& value)
{

  int n_trials = 5;
  bool success = false;

  while(n_trials > 0) {
    try {
      success = redis_cluster.set(key, value);
      n_trials = -1;
    }
    catch (sw::redis::IoError& e) {
      n_trials--;
      std::cout<<"WARNING: Caught redis IOError: "<<e.what()<<std::endl;
      std::cout<<"WARNING: Could not set key "<<key<<" in database. "<<n_trials<<" more trials will be made."<<std::endl;
    }
  }
  if(n_trials == 0)
    throw std::runtime_error("Could not set " + std::string(key) + " in database due to redis IOError.");

  if(!success)
    throw std::runtime_error("KeyDB failed to receive key: " + std::string(key));

  return;
}

std::string SmartSimClient::_build_put_key(const char* key)
{
  //This function builds the key that it will be used
  //for the put value.  The key is SSKEYOUT + _ + key
  std::string suffix(key);
  std::string out = (this->_put_key_prefix.empty()) ? std::string(key) : this->_put_key_prefix + '_' + suffix;
  return out.c_str();

}

std::string SmartSimClient::_build_get_key(const char* key)
{
  //This function builds the key that it will be used
  //for the put value.  The key is SSDATAIN + _ + key

  std::string suffix(key);
  std::string out = (this->_get_key_prefix.empty()) ? std::string(key) : this->_get_key_prefix + '_' + suffix;
  return out.c_str();
}

std::string SmartSimClient::_get_from_keydb(const char* key)
{

  int n_trials = 5;
  sw::redis::OptionalString value;

  while(n_trials > 0) {
    try {
      value = redis_cluster.get(key);
      n_trials = -1;
    }
    catch (sw::redis::IoError& e) {
      n_trials--;
      std::cout<<"WARNING: Caught redis IOError: "<<e.what()<<std::endl;
      std::cout<<"WARNING: Could not get key "<<key<<" from database. "<<n_trials<<" more trials will be made."<<std::endl;
    }
  }
  if(n_trials == 0)
    throw std::runtime_error("Could not retreive "+std::string(key)+" from database due to redis IOError.");

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

std::string SmartSimClient::_serialize_protobuff(google::protobuf::Message* pb_message)
{
  std::string buff;
  bool success = pb_message->SerializeToString(&buff);

  if(!success)
    throw std::runtime_error("Protobuf serialization failed");

  return buff;
}

void SmartSimClient::_put_keydb_value_into_protobuff(google::protobuf::Message* pb_message, const char* key, int n_values)
{
  std::string value = _get_from_keydb(key);

  pb_message->ParseFromString(value);

  return;
}

void SmartSimClient::_clear_protobuff(google::protobuf::Message* pb_message)
{
  const google::protobuf::Reflection* refl = pb_message->GetReflection();
  std::vector<const google::protobuf::FieldDescriptor*>  pb_message_fields;
  refl->ListFields(*pb_message, &pb_message_fields);

  std::vector<const google::protobuf::FieldDescriptor*>::iterator it;
  for (it = pb_message_fields.begin(); it != pb_message_fields.end(); it++)
    refl->ClearField(pb_message, *it);

  return;
}

extern "C" void* initialize_c_client() {
  SmartSimClient *s = new SmartSimClient(false);
  void* c_ptr = (void*)s;
  return c_ptr;
}

extern "C" void* initialize_fortran_client() {
  SmartSimClient *s = new SmartSimClient(true);
  void* c_ptr = (void*)s;
  return c_ptr;
}