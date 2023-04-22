#include "erl_nif.h"

ErlNifResourceType *ARRAY_TYPE;


static int
load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
  //printf("ok");
  //fflush();
  //return 0;
  ARRAY_TYPE =
  enif_open_resource_type(env, NULL, "array", NULL, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER , NULL);
  return 0;
}

static ERL_NIF_TERM
create_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  // Let's allocate the memory for a db_conn_t * pointer
printf("ok");
  int size;
  // Fill a and b with the values of the first two args
  if (!enif_get_int(env, argv[0], &size)) {
      return enif_make_badarg(env);
  }
  float **array_res = enif_alloc_resource(ARRAY_TYPE, sizeof(float *));

  // Let's create conn and let the resource point to it
  float *array = (float*) malloc(size * sizeof(float));
  *array_res = array;
  for (int index = 0; index < size; index += 1) {
    array[index]=index  ;
  }
  // We can now make the Erlang term that holds the resource...
  ERL_NIF_TERM term = enif_make_resource(env, array_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(array_res);

  return term;
}


static ERL_NIF_TERM
print_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {

  
  float **array_res;
  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void *) &array_res)) {
    return enif_make_badarg(env);
  }
  int size;
  // Fill a and b with the values of the first two args
  if (!enif_get_int(env, argv[1], &size)) {
      return enif_make_badarg(env);
  }

  float *array = *array_res;

  // We can now run our query
  for (int index = 0; index < size; index += 1) {
    printf("[%d]=%f\n", index,array[index])  ;
  }
    return enif_make_int(env, 0);
  
}

static ErlNifFunc nif_funcs[] = {
  {"create_array_nif", 1, create_array_nif},
  {"print_array_nif", 2, print_array_nif}
};

ERL_NIF_INIT(Elixir.TesteArray, nif_funcs, load, NULL, NULL, NULL)

