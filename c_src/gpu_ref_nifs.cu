#include "erl_nif.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <dlfcn.h>

#define MX_ROWS(matrix) (((uint32_t*)matrix)[0])
#define MX_COLS(matrix) (((uint32_t*)matrix)[1])
#define MX_SET_ROWS(matrix, rows) ((uint32_t*)matrix)[0] = rows
#define MX_SET_COLS(matrix, cols) ((uint32_t*)matrix)[1] = cols
#define MX_LENGTH(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)

ErlNifResourceType *ARRAY_TYPE;


static int
load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  ARRAY_TYPE =
  enif_open_resource_type(env, NULL, "gpu_ref", NULL, ERL_NIF_RT_CREATE  , NULL);
  return 0;
}


static ERL_NIF_TERM create_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary  matrix_el;
  float         *matrix;
  float         *dev_matrix;
  
  if (!enif_inspect_binary(env, argv[0], &matrix_el)) return enif_make_badarg(env);

  matrix = (float *) matrix_el.data;
  uint64_t data_size = sizeof(float)*MX_LENGTH(matrix);
  
  matrix +=2; 

  cudaMalloc( (void**)&dev_matrix, data_size);
  cudaMemcpy( dev_matrix, matrix, data_size, cudaMemcpyHostToDevice );


  float **gpu_res = (float**)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}

static ERL_NIF_TERM new_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  float         *dev_matrix;
  int data_size;
  
  if (!enif_get_int(env, argv[0], &data_size)) {
      return enif_make_badarg(env);
  }
 
  data_size = data_size * sizeof(float);
  cudaMalloc( (void**)&dev_matrix, data_size);
  

  float **gpu_res = (float**)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}


static ERL_NIF_TERM get_matrex_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  int nrow;
  int ncol;
  ERL_NIF_TERM  result;
  float **array_res;
  
  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **) &array_res)) {
    return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[1], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[2], &ncol)) {
      return enif_make_badarg(env);
  }
  
  float *dev_array = *array_res;

  int result_size = sizeof(float) * (nrow*ncol+2);
  float *result_data = (float *) enif_make_new_binary(env, result_size, &result);

  float *ptr_matrix ;
  ptr_matrix = result_data;
  ptr_matrix +=2;

  cudaMemcpy( dev_array, ptr_matrix, result_size, cudaMemcpyHostToDevice );

  MX_SET_ROWS(result_data, nrow);
  MX_SET_COLS(result_data, ncol);
  
  return result;
}


static ErlNifFunc nif_funcs[] = {
  {"create_ref_nif", 1, create_ref_nif},
  {"new_ref_nif", 1, new_ref_nif},
  {"get_matrex_nif", 3, get_matrex_nif}
};

ERL_NIF_INIT(Elixir.GPURef, nif_funcs, load, NULL, NULL, NULL)
