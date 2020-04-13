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

void SmartSimClient::put_array_double(const char* key, void* value, int* dims, int n_dims, bool fortran_array)
{
  this->_put_array<SmartSimProtobuf::ArrayDouble,double>(key, value, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::put_array_float(const char* key, void* value, int* dims, int n_dims, bool fortran_array)
{
  this->_put_array<SmartSimProtobuf::ArrayFloat,float>(key, value, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::put_array_int64(const char* key, void* value, int* dims, int n_dims, bool fortran_array)
{
  this->_put_array<SmartSimProtobuf::ArraySInt64,int64_t>(key, value, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::put_array_int32(const char* key, void* value, int* dims, int n_dims, bool fortran_array)
{
  this->_put_array<SmartSimProtobuf::ArraySInt32,int32_t>(key, value, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::put_array_uint64(const char* key, void* value, int* dims, int n_dims, bool fortran_array)
{
  this->_put_array<SmartSimProtobuf::ArrayUInt64,uint64_t>(key, value, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::put_array_uint32(const char* key, void* value, int* dims, int n_dims, bool fortran_array)
{
  this->_put_array<SmartSimProtobuf::ArrayUInt32,uint32_t>(key, value, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::get_array_double(const char* key, void* result, int* dims, int n_dims, bool fortran_array)
{
  this->_get_array<SmartSimProtobuf::ArrayDouble, double>(key, result, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::get_array_float(const char* key, void* result, int* dims, int n_dims, bool fortran_array)
{
  this->_get_array<SmartSimProtobuf::ArrayFloat, float>(key, result, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::get_array_int64(const char* key, void* result, int* dims, int n_dims, bool fortran_array)
{
  this->_get_array<SmartSimProtobuf::ArraySInt64,int64_t>(key, result, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::get_array_int32(const char* key, void* result, int* dims, int n_dims, bool fortran_array)
{
  this->_get_array<SmartSimProtobuf::ArraySInt32,int32_t>(key, result, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::get_array_uint64(const char* key, void* result, int* dims, int n_dims, bool fortran_array)
{
  this->_get_array<SmartSimProtobuf::ArrayUInt64,uint64_t>(key, result, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::get_array_uint32(const char* key, void* result, int* dims, int n_dims, bool fortran_array)
{
  this->_get_array<SmartSimProtobuf::ArrayUInt32,uint32_t>(key, result, dims, n_dims, fortran_array);
  return;
}

void SmartSimClient::put_scalar_double(const char* key, double value)
{
  this->_put_scalar<double>(&protob_scalar_double, key, value);
  return;
}

void SmartSimClient::put_scalar_float(const char* key, float value)
{
  this->_put_scalar<float>(&protob_scalar_float, key, value);
  return;
}

void SmartSimClient::put_scalar_int64(const char* key, int64_t value)
{
  this->_put_scalar<int64_t>(&protob_scalar_int64, key, value);
  return;
}

void SmartSimClient::put_scalar_int32(const char* key, int32_t value)
{
  this->_put_scalar<int32_t>(&protob_scalar_int32, key, value);
  return;
}

void SmartSimClient::put_scalar_uint64(const char* key, uint64_t value)
{
  this->_put_scalar<uint64_t>(&protob_scalar_uint64, key, value);
  return;
}

void SmartSimClient::put_scalar_uint32(const char* key, uint32_t value)
{
  this->_put_scalar<uint32_t>(&protob_scalar_uint32, key, value);
  return;
}

double SmartSimClient::get_scalar_double(const char* key)
{
  return this->_get_scalar<double>(&protob_scalar_double, key);
}

float SmartSimClient::get_scalar_float(const char* key)
{
  return this->_get_scalar<float>(&protob_scalar_float, key);
}

int64_t SmartSimClient::get_scalar_int64(const char* key)
{
  return this->_get_scalar<int64_t>(&protob_scalar_int64, key);
}

int32_t SmartSimClient::get_scalar_int32(const char* key)
{
  return this->_get_scalar<int32_t>(&protob_scalar_int32, key);
}

uint64_t SmartSimClient::get_scalar_uint64(const char* key)
{
  return this->_get_scalar<uint64_t>(&protob_scalar_uint64, key);
}

uint32_t SmartSimClient::get_scalar_uint32(const char* key)
{
  return this->_get_scalar<uint64_t>(&protob_scalar_uint32, key);
}

template <class T>
void SmartSimClient::_put_scalar(google::protobuf::Message* pb_message, const char* key, T value)
{
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
  _put_to_keydb(key, output);

  return;  
}

template<class T>
T SmartSimClient::_get_scalar(google::protobuf::Message* pb_message, const char* key)
{
  _put_keydb_value_into_protobuff(pb_message, key, 1);
  
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
void SmartSimClient::_put_array(const char* key, void* value, int* dims, int n_dims, bool fortran_array)
{
  std::string buff;
  T* pb_message = new T();
  this->_serialize_array<U>(pb_message, buff, value, dims, n_dims, fortran_array);
  delete pb_message;
  this->_put_to_keydb(key, buff);
  return;
}

template <class  T>
void SmartSimClient::_serialize_array(google::protobuf::Message* pb_message, std::string& buff, void* value, int* dims, int n_dims, bool fortran_array)
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
  if(fortran_array) {
    
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

  if(fortran_array)
    delete[] dims;

  buff = _serialize_protobuff(pb_message);

  return;
}

template <class T, class U>
void SmartSimClient::_get_array(const char* key, void* result, int* dims, int n_dims, bool fortran_array)
{

  T* pb_message = new T();
  
  if(n_dims<=0)
    return;

  int n_values = 1;
  for(int i = 0; i < n_dims; i++)
    n_values *= dims[i];

  _put_keydb_value_into_protobuff(pb_message, key, n_values);

  // If it is a fortran array, reset dims to
  // point to dynamically allocated dimension of 1
  // so that _place_array can be used.
  if (fortran_array) {
    n_dims = 1;
    dims = new int[1];
    dims[0] = n_values;
  }

  int proto_position = 0;

  const google::protobuf::Reflection* refl = pb_message->GetReflection();
  const google::protobuf::FieldDescriptor* data_field = pb_message->GetDescriptor()->FindFieldByName("data");
  const google::protobuf::MutableRepeatedFieldRef<U> data = refl->GetMutableRepeatedFieldRef<U>(pb_message, data_field);
  _place_array_values<U>(data, result, dims, n_dims, proto_position);

  if(fortran_array)
    delete[] dims;

  delete pb_message;

  return;
}

bool SmartSimClient::exists(const char* key)
{
  std::string prefixed_key = _build_get_key(key);
  return redis_cluster.exists(prefixed_key.c_str());
}

bool SmartSimClient::poll_key(const char* key, int poll_frequency_ms, int num_tries)
{
  bool key_exists = false;
  
  while(!(num_tries==0)) {
    if(this->exists(key)) {
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

template <class T>
bool SmartSimClient::_poll_key_and_check_scalar(const char* key, T value, int poll_frequency_ms, int num_tries)
{
  bool matched_value = false;
  T current_value;

  while( !(num_tries==0) ) {
    if(this->exists(key)) {
      if(std::is_same<T, double>::value)
	current_value = this->get_scalar_double(key);
      else if(std::is_same<T, float>::value)
	current_value = this->get_scalar_float(key);
      else if(std::is_same<T, int64_t>::value)
	current_value = this->get_scalar_int64(key);
      else if(std::is_same<T, int32_t>::value)
	current_value = this->get_scalar_int32(key);
      else if(std::is_same<T, uint64_t>::value)
	current_value = this->get_scalar_uint64(key);
      else if(std::is_same<T,uint32_t>::value)
	current_value = this->get_scalar_uint32(key);
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

  //if(!(protob_double.data_size()==n_values))
  //  throw std::runtime_error("The protobuf array is not the same length as specified by n_values");
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

extern "C" void* GetObject() {
  SmartSimClient *s = new SmartSimClient();
  void* c_ptr = (void*)s;
  return c_ptr;
}

extern "C" void* ssc_constructor()
{
  return new SmartSimClient;
}

extern "C" void put_array_double_c(void* SmartSimClient_proto, const char *key, void *value, int **dimensions, int *ndims)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  s->put_array_double(key, value, *dimensions, *ndims, true);
}

extern "C" void get_array_double_c(void* SmartSimClient_proto, const char *key, void *value, int **dimensions, int *ndims)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  s->get_array_double(key, value, *dimensions, *ndims, true);
}

extern "C" void get_array_int64_c(void* SmartSimClient_proto, const char* key, void *value, int **dimensions, int *ndims)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  s->get_array_int64(key, value, *dimensions, *ndims, true);
}

extern "C" void put_array_int64_c(void* SmartSimClient_proto, const char* key, void *value, int **dimensions, int *ndims)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  s->put_array_int64(key, value, *dimensions, *ndims, true);
}

extern "C" void put_scalar_int64_c(void* SmartSimClient_proto, const char *key, int64_t value)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  s->put_scalar_int64(key, value);
}

extern "C" int64_t get_scalar_int64_c(void* SmartSimClient_proto, const char *key)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  return  s->get_scalar_int64(key);
}

extern "C" bool poll_key_and_check_scalar_int64_c(void *SmartSimClient_proto, const char* key, int64_t value, 
                                                  int poll_frequency_ms, int num_tries)
{
  SmartSimClient *s = (SmartSimClient *)SmartSimClient_proto;
  return s->poll_key_and_check_scalar_int64(key, value, poll_frequency_ms, num_tries);
}
