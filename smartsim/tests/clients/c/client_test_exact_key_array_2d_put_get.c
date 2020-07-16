#include <mpi.h>
#include "c_client.h"
#include "c_client_test_utils.h"
#include <inttypes.h>
#include <stdlib.h>

int32_t** generate_random_2d_array_int32( int dim1, int dim2 )
{
   int32_t **array = (int32_t **)malloc(dim1*sizeof(int32_t *));
   for (int i=0; i<dim1; i++)
       array[i] = (int32_t *)malloc(dim2*sizeof(int32_t));
   for (int i=0; i<dim1; i++)
      for (int j=0; j<dim2; j++)
         array[i][j] = safe_rand();
   return array;
}

uint32_t** generate_random_2d_array_uint32( int dim1, int dim2 )
{
   uint32_t **array = (uint32_t **)malloc(dim1*sizeof(uint32_t *));
   for (int i=0; i<dim1; i++)
       array[i] = (uint32_t *)malloc(dim2*sizeof(uint32_t));
   for (int i=0; i<dim1; i++)
      for (int j=0; j<dim2; j++)
        array[i][j] = safe_rand();
   return array;
}

int64_t** generate_random_2d_array_int64( int dim1, int dim2 )
{
   int64_t **array = (int64_t **)malloc(dim1*sizeof(int64_t *));
   for (int i=0; i<dim1; i++)
       array[i] = (int64_t *)malloc(dim2*sizeof(int64_t));
   for (int i=0; i<dim1; i++)
      for (int j=0; j<dim2; j++)
        array[i][j] = safe_rand();
    return array;
}

uint64_t** generate_random_2d_array_uint64( int dim1, int dim2 )
{
   uint64_t **array = (uint64_t **)malloc(dim1*sizeof(uint64_t *));
   for (int i=0; i<dim1; i++)
       array[i] = (uint64_t *)malloc(dim2*sizeof(uint64_t));
   for (int i=0; i<dim1; i++)
      for (int j=0; j<dim2; j++)
        array[i][j] = safe_rand();
    return array;
}

float** generate_random_2d_array_float( int dim1, int dim2 )
{
   float **array = (float **)malloc(dim1*sizeof(float *));
   for (int i=0; i<dim1; i++)
       array[i] = (float *)malloc(dim2*sizeof(float));
   for (int i=0; i<dim1; i++)
      for (int j=0; j<dim2; j++)
        array[i][j] = rand_float();
    return array;
}

double** generate_random_2d_array_double( int dim1, int dim2 )
{
   double **array = (double **)malloc(dim1*sizeof(double *));
   for (int i=0; i<dim1; i++)
       array[i] = (double *)malloc(dim2*sizeof(double));
   for (int i=0; i<dim1; i++)
      for (int j=0; j<dim2; j++)
        array[i][j] = rand_double();
    return array;
}

int check_equal_2d_int32( int32_t **array1, int32_t **array2, int dim1, int dim2 )
{
    for (int i=0;i<dim1;i++)
      for (int j=0; j<dim2; j++)
        if (array1[i][j] != array2[i][j])
            return 0;
    return 1;
}

int check_equal_2d_uint32( uint32_t **array1, uint32_t **array2, int dim1, int dim2 )
{
    for (int i=0;i<dim1;i++)
      for (int j=0; j<dim2; j++)
        if (array1[i][j] != array2[i][j])
            return 0;
    return 1;
}

int check_equal_2d_int64( int64_t **array1, int64_t **array2, int dim1, int dim2 )
{
    for (int i=0;i<dim1;i++)
      for (int j=0; j<dim2; j++)
        if (array1[i][j] != array2[i][j])
            return 0;
    return 1;
}

int check_equal_2d_uint64( uint64_t **array1, uint64_t **array2, int dim1, int dim2 )
{
    for (int i=0;i<dim1;i++)
      for (int j=0; j<dim2; j++)
        if (array1[i][j] != array2[i][j])
            return 0;
    return 1;
}

int check_equal_2d_float( float **array1, float **array2, int dim1, int dim2 )
{
    for (int i=0;i<dim1;i++)
      for (int j=0; j<dim2; j++)
        if (array1[i][j] != array2[i][j])
            return 0;
    return 1;
}

int check_equal_2d_double( double **array1, double **array2, int dim1, int dim2 )
{
    for (int i=0;i<dim1;i++)
      for (int j=0; j<dim2; j++)
        if (array1[i][j] != array2[i][j])
            return 0;
    return 1;
}

void free_2d_int32( int32_t **array, int dim1 ){
   for (int i=0; i<dim1; i++)
       free(array[i]);
   free(array);
}

void free_2d_int64( int64_t **array, int dim1 ){
   for (int i=0; i<dim1; i++)
       free(array[i]);
   free(array);
}

void free_2d_uint32( uint32_t **array, int dim1 ){
   for (int i=0; i<dim1; i++)
       free(array[i]);
    free(array);
}

void free_2d_uint64( uint64_t **array, int dim1 ){
   for (int i=0; i<dim1; i++)
       free(array[i]);
   free(array);
}

void free_2d_float( float **array, int dim1 ){
   for (int i=0; i<dim1; i++)
       free(array[i]);
   free(array);
}

void free_2d_double( double **array, int dim1 ){
   for (int i=0; i<dim1; i++)
       free(array[i]);
   free(array);
}

int main(int argc, char* argv[]) {

  MPI_Init(NULL,NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  void *smartsim_client = initialize_c_client( 0 );

  int ndims = 2;
  int dim1 = 10;
  int dim2 = 5;
  int *dims = (int *) malloc(ndims*sizeof(int));
  dims[0] = dim1;
  dims[1] = dim2;

  char array_int32_key[] = "array_int32";
  int32_t **true_array_int32 = generate_random_2d_array_int32( dim1, dim2 ) ;
  int32_t **recv_array_int32 = generate_random_2d_array_int32( dim1, dim2 );
  put_exact_key_array_int32_c(smartsim_client, array_int32_key, true_array_int32, &dims, &ndims);
  get_exact_key_array_int32_c(smartsim_client, array_int32_key, recv_array_int32, &dims, &ndims);
  test_result( check_equal_2d_int32(true_array_int32, recv_array_int32, dim1, dim2),
                                    "put_exact_key_array_int32" );
  free_2d_int32(true_array_int32, dim1);
  free_2d_int32(recv_array_int32, dim1);

  char array_int64_key[] = "array_int64";
  int64_t **true_array_int64 = generate_random_2d_array_int64( dim1, dim2 ) ;
  int64_t **recv_array_int64 = generate_random_2d_array_int64( dim1, dim2 );
  put_exact_key_array_int64_c(smartsim_client, array_int64_key, true_array_int64, &dims, &ndims);
  get_exact_key_array_int64_c(smartsim_client, array_int64_key, recv_array_int64, &dims, &ndims);
  test_result( check_equal_2d_int64(true_array_int64, recv_array_int64, dim1, dim2),
                                    "put_exact_key_array_int64" );
  free_2d_int64(true_array_int64, dim1);
  free_2d_int64(recv_array_int64, dim1);

  char array_uint32_key[] = "array_uint32";
  uint32_t **true_array_uint32 = generate_random_2d_array_uint32( dim1, dim2 ) ;
  uint32_t **recv_array_uint32 = generate_random_2d_array_uint32( dim1, dim2 );
  put_exact_key_array_uint32_c(smartsim_client, array_uint32_key, true_array_uint32, &dims, &ndims);
  get_exact_key_array_uint32_c(smartsim_client, array_uint32_key, recv_array_uint32, &dims, &ndims);
  test_result( check_equal_2d_uint32(true_array_uint32, recv_array_uint32, dim1, dim2),
                                    "put_exact_key_array_uint32" );
  free_2d_uint32(true_array_uint32, dim1);
  free_2d_uint32(recv_array_uint32, dim1);

  char array_uint64_key[] = "array_uint64";
  uint64_t **true_array_uint64 = generate_random_2d_array_uint64( dim1, dim2 ) ;
  uint64_t **recv_array_uint64 = generate_random_2d_array_uint64( dim1, dim2 );
  put_exact_key_array_uint64_c(smartsim_client, array_uint64_key, true_array_uint64, &dims, &ndims);
  get_exact_key_array_uint64_c(smartsim_client, array_uint64_key, recv_array_uint64, &dims, &ndims);
  test_result( check_equal_2d_uint64(true_array_uint64, recv_array_uint64, dim1, dim2),
                                    "put_exact_key_array_uint64" );
  free_2d_uint64(true_array_uint64, dim1);
  free_2d_uint64(recv_array_uint64, dim1);

  char array_float_key[] = "array_float";
  float **true_array_float = generate_random_2d_array_float( dim1, dim2 ) ;
  float **recv_array_float = generate_random_2d_array_float( dim1, dim2 );
  put_exact_key_array_float_c(smartsim_client, array_float_key, true_array_float, &dims, &ndims);
  get_exact_key_array_float_c(smartsim_client, array_float_key, recv_array_float, &dims, &ndims);
  test_result( check_equal_2d_float(true_array_float, recv_array_float, dim1, dim2),
                                    "put_exact_key_array_float" );
  free_2d_float(true_array_float, dim1);
  free_2d_float(recv_array_float, dim1);

  char array_double_key[] = "array_double";
  double **true_array_double = generate_random_2d_array_double( dim1, dim2 ) ;
  double **recv_array_double = generate_random_2d_array_double( dim1, dim2 );
  put_exact_key_array_double_c(smartsim_client, array_double_key, true_array_double, &dims, &ndims);
  get_exact_key_array_double_c(smartsim_client, array_double_key, recv_array_double, &dims, &ndims);
  test_result( check_equal_2d_double(true_array_double, recv_array_double, dim1, dim2),
                                    "put_exact_key_array_double" );
  free_2d_double(true_array_double, dim1);
  free_2d_double(recv_array_double, dim1);

  return 0;
}
