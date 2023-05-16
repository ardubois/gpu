#include "erl_nif.h"

__global__
void aadd_vectors(float *result, float *a, float *b, int n)
{
int index = (threadIdx.x + (blockIdx.x * blockDim.x));
int stride = (blockDim.x * gridDim.x);
for( int i = index; i<n; i+=stride){
	result[i] = (a[i] + b[i]);
}

}

extern "C" void aadd_vectors_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type)
  {

    ERL_NIF_TERM list;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail;

    int blocks,threads;
    float **array_res;

    enif_get_int(env, argv[1], &blocks);
    enif_get_int(env, argv[2], &threads);
    list= argv[3];

  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg1 = *array_res;
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg2 = *array_res;
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg3 = *array_res;
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  int arg4;
  enif_get_int(env, head, &arg4);
  list = tail;

   aadd_vectors<<<blocks, threads>>>(arg1,arg2,arg3,arg4);
}
