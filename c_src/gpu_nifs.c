#include "erl_nif.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define MX_ROWS(matrix) (((uint32_t*)matrix)[0])
#define MX_COLS(matrix) (((uint32_t*)matrix)[1])
#define MX_SET_ROWS(matrix, rows) ((uint32_t*)matrix)[0] = rows
#define MX_SET_COLS(matrix, cols) ((uint32_t*)matrix)[1] = cols
#define MX_LENGTH(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)

ErlNifResourceType *ARRAY_TYPE;


static int
load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  ARRAY_TYPE =
  enif_open_resource_type(env, NULL, "gpu_ref", NULL, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER , NULL);
  return 0;
}


static ERL_NIF_TERM create_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary  matrix_el;
  float         *matrix;
  float         *dev_matrix;
  
  if (!enif_inspect_binary(env, argv[0], &matrix_el)) return enif_make_badarg(env);

  matrix = (float *) matrix_el.data;
  uint64_t data_size = MX_LENGTH(matrix);
  cudaMalloc( (void**)&dev_matrix, data_size);
  cudaMemcpy( dev_matrix, matrix, data_size, cudaMemcpyHostToDevice );


  float **gpu_res = enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}

static ErlNifFunc nif_funcs[] = {
  {"create_ref_nif", 1, create_ref_nif}
};

ERL_NIF_INIT(Elixir.GPU, nif_funcs, load, NULL, NULL, NULL)
