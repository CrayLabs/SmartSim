#ifndef SMARTSIM_TEST_UTILS_H
#define SMARTSIM_TEST_UTILS_H

#include <typeinfo>
#include <random>

template <typename T>
T** allocate_2D_array(int dim_1, int dim_2)
{
  /* This function allocates a 2D array and
     and returns a pointer to that 2D array.
  */
  T **array = (T **)malloc(dim_1*sizeof(T *));
  for (int i=0; i<dim_1; i++)
    array[i] = (T *)malloc(dim_2*sizeof(T));

  return array;
}

template <typename T>
T*** allocate_3D_array(int dim_1, int dim_2, int dim_3)
{
  /* This function allocates a 3D array and returns
     a pointer to that 3D array.
  */
  T*** array = (T***)malloc(dim_1*sizeof(T**));
  for (int i=0; i<dim_1; i++) {
    array[i] = (T**)malloc(dim_2*sizeof(T*));
    for(int j=0; j<dim_2; j++){
      array[i][j] = (T*)malloc(dim_3 * sizeof(T));
    }
  }
  return array;
}

template <typename T>
T**** allocate_4D_array(int dim_1, int dim_2,
                        int dim_3, int dim_4)
{
  /* This function allocates a 4D array and returns
  a pointer to that 4D array.  This is not coded
  recursively to avoid propagating bugs.
  */
  T**** array = (T****)malloc(dim_1*sizeof(T***));
  for(int i=0; i<dim_1; i++) {
    array[i] = (T***)malloc(dim_2*sizeof(T**));
    for(int j=0; j<dim_2; j++) {
      array[i][j] = (T**)malloc(dim_3*sizeof(T*));
      for(int k=0; k<dim_4; k++) {
        array[i][j][k] = (T*)malloc(dim_4 * sizeof(T));
      }
    }
  }
  return array;
}

template <typename T>
void free_1D_array(T* array)
{
  /* This function frees memory associated with
     pointer.
  */
  free(array);
}

template <typename T>
void free_2D_array(T** array, int dim_1)
{
  /*  This function frees memory of dynamically
      allocated 2D array.
  */
  for(int i=0; i<dim_1; i++)
       free(array[i]);
  free(array);
}

template <typename T>
void free_3D_array(T*** array, int dim_1, int dim_2)
{
  /* This function frees memory of dynamically
     allocated 3D array.
  */
  for(int i=0; i<dim_1; i++)
    free_2D_array(array[i], dim_2);
  free(array);
}

template <typename T>
void free_4D_array(T**** array, int dim_1,
                   int dim_2, int dim_3)
{
  for(int i=0; i<dim_1; i++)
    free_3D_array(array[i], dim_2, dim_3);
  return;
}

template <typename T, typename U>
bool is_equal_1D_array(T* a, U* b, int dim_1)
{
  /* This function compares two arrays to
     make sure their values are identical.
  */
  for(int i=0; i<dim_1; i++)
      if(!(a[i] == b[i]))
	return false;
  return true;
}

template <typename T, typename U>
bool is_equal_2D_array(T** a, U** b, int dim_1, int dim_2)
{
  /* This function compares two 2D arrays to
     check if they are identical.
  */
  for(int i=0; i<dim_1; i++)
    for(int j=0; j<dim_2; j++)
      if(!(a[i][j] == b[i][j]))
	return false;
  return true;
}

template <typename T, typename U>
bool is_equal_3D_array(T*** a, U*** b, int dim_1, int dim_2, int dim_3)
{
  /* This function compares two 3D arrays to
     check if they are identical.
  */
  for(int i=0; i<dim_1; i++)
    for(int j=0; j<dim_2; j++)
      for(int k=0; k<dim_3; k++)
	if(!(a[i][j][k] == b[i][j][k]))
	  return false;
  return true;
}

template <typename T>
void set_1D_array_floating_point_values(T* a, int dim_1)
{
  /* This function fills a 1D array with random
     floating point values.
  */
  std::default_random_engine generator(rand());
  std::uniform_real_distribution<T> distribution;
  for(int i=0; i<dim_1; i++)
    //a[i] = distribution(generator);
    a[i] = 2.0*rand()/RAND_MAX - 1.0;
}

template <typename T>
void set_2D_array_floating_point_values(T** a, int dim_1, int dim_2)
{
  /* This function fills a 2D array with random
     floating point values.
  */
  for(int i = 0; i < dim_1; i++) {
    set_1D_array_floating_point_values<T>(a[i], dim_2);
  }
}

template <typename T>
void set_3D_array_floating_point_values(T*** a, int dim_1, int dim_2, int dim_3)
{
  /* This function fills a 3D array with random floating
     point values.
  */
  for(int i = 0; i < dim_1; i++)
    set_2D_array_floating_point_values<T>(a[i], dim_2, dim_3);
}

template <typename T>
void set_1D_array_integral_values(T* a, int dim_1)
{
  /* This function fills a 1D array with random
     integral values.
  */
  std::default_random_engine generator(rand());
  T t_min = std::numeric_limits<T>::min();
  T t_max = std::numeric_limits<T>::max();
  std::uniform_int_distribution<T> distribution(t_min, t_max);
  for(int i=0; i<dim_1; i++)
    a[i] = distribution(generator);
}

template <typename T>
void set_2D_array_integral_values(T** a, int dim_1, int dim_2)
{
  /* This function fills a 2D array with random
     integral values.
  */
  for(int i = 0; i < dim_1; i++) {
    set_1D_array_integral_values<T>(a[i], dim_2);
  }

}

template <typename T>
void set_3D_array_integral_values(T*** a, int dim_1, int dim_2, int dim_3)
{
  /* This function fills a 3D array with random
     integral values.
  */
  for(int i = 0; i < dim_1; i++)
    set_2D_array_integral_values<T>(a[i], dim_2, dim_3);
}

template <typename T>
T get_integral_scalar()
{
  /* This function returns a random integral
     scalar value.
  */
  std::default_random_engine generator;
  std::uniform_int_distribution<T> distribution;
  return distribution(generator);
}

template <typename T>
T get_floating_point_scalar()
{
  /* This function returns a random floating
     point value.
  */
  std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution;
  return distribution(generator);
}

#endif //SMARTSIM_TEST_UTILS_H
