#include "erl_nif.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MX_LENGTH(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)

static ERL_NIF_TERM print_matrex_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary  matrix_el;
  float        *matrix_data;
  float         *matrix;
  if (!enif_inspect_binary(env, argv[0], &matrix_el)) return enif_make_badarg(env);

  matrix = (float *) matrix_el.data;

  uint64_t data_size = MX_LENGTH(matrix);
  
  for (uint64_t index = 2; index < data_size; index += 1) {
    printf("%f\n", matrix[index])  ;
  }
    return enif_make_int(env, 0);
 }


// Let's define the array of ErlNifFunc beforehand:
static ErlNifFunc nif_funcs[] = {
  // {erl_function_name, erl_function_arity, c_function}
  {"print_matrex_nif", 1, print_matrex_nif}
};

ERL_NIF_INIT(Elixir.TesteGPU, nif_funcs, NULL, NULL, NULL, NULL)
