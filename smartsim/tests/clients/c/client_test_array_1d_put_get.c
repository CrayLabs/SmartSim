#include <mpi.h>
#include "c_client.h"
#include "c_client_test_utils.h"
#include <inttypes.h>
#include <stdlib.h>

int32_t* generate_random_1d_array_int32( int nsize )
{
   int32_t *array = (int32_t *) malloc(nsize * sizeof(int32_t));
   for (int i=0; i<nsize; i++)
       array[i] = safe_rand();
   return array;
}

uint32_t* generate_random_1d_array_uint32( int nsize )
{
   uint32_t *array = (uint32_t *) malloc(nsize * sizeof(uint32_t));
   for (int i=0; i<nsize; i++)
       array[i] = safe_rand();
   return array;
}

int64_t* generate_random_1d_array_int64( int nsize )
{
   int64_t *array = (int64_t *) malloc(nsize * sizeof(int64_t));
   for (int i=0; i<nsize; i++)
       array[i] = safe_rand();
   return array;
}

uint64_t* generate_random_1d_array_uint64( int nsize )
{
   uint64_t *array = (uint64_t *) malloc(nsize * sizeof(uint64_t));
   for (int i=0; i<nsize; i++)
       array[i] = safe_rand();
   return array;
}

float* generate_random_1d_array_float( int nsize )
{
   float *array = (float *) malloc(nsize * sizeof(float));
   for (int i=0; i<nsize; i++)
       array[i] = rand_float();
   return array;
}

double* generate_random_1d_array_double( int nsize )
{
   double *array = (double *) malloc(nsize * sizeof(double));
   for (int i=0; i<nsize; i++)
       array[i] = rand_double();
   return array;
}

int check_equal_1d_int32( int32_t *array1, int32_t *array2, int dim1 )
{
    for (int i=0;i<dim1;i++)
        if (array1[i] != array2[i])
            return 0;
    return 1;
}

int check_equal_1d_uint32( uint32_t *array1, uint32_t *array2, int dim1 )
{
    for (int i=0;i<dim1;i++)
        if (array1[i] != array2[i])
            return 0;
    return 1;
}

int check_equal_1d_int64( int64_t *array1, int64_t *array2, int dim1 )
{
    for (int i=0;i<dim1;i++)
        if (array1[i] != array2[i])
            return 0;
    return 1;
}

int check_equal_1d_uint64( uint64_t *array1, uint64_t *array2, int dim1 )
{
    for (int i=0;i<dim1;i++)
        if (array1[i] != array2[i])
            return 0;
    return 1;
}

int check_equal_1d_float( float *array1, float *array2, int dim1 )
{
    for (int i=0;i<dim1;i++)
        if (array1[i] != array2[i])
            return 0;
    return 1;
}

int check_equal_1d_double( double *array1, double *array2, int dim1 )
{
    for (int i=0;i<dim1;i++)
        if (array1[i] != array2[i])
            return 0;
    return 1;
}

int main(int argc, char* argv[]) {

  MPI_Init(NULL,NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  void *smartsim_client = initialize_c_client( 0 );

  int ndims = 1;
  int dim1 = 10;
  int *dims = (int *) malloc(ndims*sizeof(int));
  dims[0] = dim1;

  char array_int32_key[] = "array_int32";
  int32_t *true_array_int32 = generate_random_1d_array_int32( dim1 ) ;
  int32_t *recv_array_int32 = generate_random_1d_array_int32( dim1 );
  put_array_int32_c(smartsim_client, array_int32_key, true_array_int32, &dims, &ndims);
  get_array_int32_c(smartsim_client, array_int32_key, recv_array_int32, &dims, &ndims);
  test_result( check_equal_1d_int32(true_array_int32, recv_array_int32, dim1),
                                    "put_array_int32" );
  free(true_array_int32);
  free(recv_array_int32);

  char array_int64_key[] = "array_int64";
  int64_t *true_array_int64 = generate_random_1d_array_int64( dim1 ) ;
  int64_t *recv_array_int64 = generate_random_1d_array_int64( dim1 );
  put_array_int64_c(smartsim_client, array_int64_key, true_array_int64, &dims, &ndims);
  get_array_int64_c(smartsim_client, array_int64_key, recv_array_int64, &dims, &ndims);
  test_result( check_equal_1d_int64(true_array_int64, recv_array_int64, dim1),
                                    "put_array_int64" );
  free(true_array_int64);
  free(recv_array_int64);

  char array_uint32_key[] = "array_uint32";
  uint32_t *true_array_uint32 = generate_random_1d_array_uint32( dim1 ) ;
  uint32_t *recv_array_uint32 = generate_random_1d_array_uint32( dim1 );
  put_array_uint32_c(smartsim_client, array_uint32_key, true_array_uint32, &dims, &ndims);
  get_array_uint32_c(smartsim_client, array_uint32_key, recv_array_uint32, &dims, &ndims);
  test_result( check_equal_1d_uint32(true_array_uint32, recv_array_uint32, dim1),
                                    "put_array_uint32" );
  free(true_array_uint32);
  free(recv_array_uint32);

  char array_uint64_key[] = "array_uint64";
  uint64_t *true_array_uint64 = generate_random_1d_array_uint64( dim1 ) ;
  uint64_t *recv_array_uint64 = generate_random_1d_array_uint64( dim1 );
  put_array_uint64_c(smartsim_client, array_uint64_key, true_array_uint64, &dims, &ndims);
  get_array_uint64_c(smartsim_client, array_uint64_key, recv_array_uint64, &dims, &ndims);
  test_result( check_equal_1d_uint64(true_array_uint64, recv_array_uint64, dim1),
                                    "put_array_uint64" );
  free(true_array_uint64);
  free(recv_array_uint64);

  char array_float_key[] = "array_float";
  float *true_array_float = generate_random_1d_array_float( dim1 ) ;
  float *recv_array_float = generate_random_1d_array_float( dim1 );
  put_array_float_c(smartsim_client, array_float_key, true_array_float, &dims, &ndims);
  get_array_float_c(smartsim_client, array_float_key, recv_array_float, &dims, &ndims);
  test_result( check_equal_1d_float(true_array_float, recv_array_float, dim1),
                                    "put_array_float" );
  free(true_array_float);
  free(recv_array_float);

  char array_double_key[] = "array_double";
  double *true_array_double = generate_random_1d_array_double( dim1 ) ;
  double *recv_array_double = generate_random_1d_array_double( dim1 );
  put_array_double_c(smartsim_client, array_double_key, true_array_double, &dims, &ndims);
  get_array_double_c(smartsim_client, array_double_key, recv_array_double, &dims, &ndims);
  test_result( check_equal_1d_double(true_array_double, recv_array_double, dim1),
                                    "put_array_double" );
  free(true_array_double);
  free(recv_array_double);

  return 0;
}
