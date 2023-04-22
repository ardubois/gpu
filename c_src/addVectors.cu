#include "erl_nif.h"

__global__
void addVectors(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
  
}


extern "C" void addVectors_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type)
{
//printf("ok kernel\n");
fflush(stdout);
ERL_NIF_TERM list;
ERL_NIF_TERM head;
ERL_NIF_TERM tail;    

int blocks,threads;
float **array_res;

enif_get_int(env, argv[1], &blocks);
enif_get_int(env, argv[2], &threads);
list= argv[3];


enif_get_list_cell(env,list,&head,&tail);
//printf("term type %d\n",enif_term_type(env,head));
int resp=enif_get_resource(env, head, type, (void **) &array_res);
float *arg1 = *array_res;
list = tail;
//printf("Pointer to matrix: %p\n",arg1);
//printf("resposta get resource:%d\n",resp);


enif_get_list_cell(env,list,&head,&tail);
enif_get_resource(env, head, type, (void **) &array_res);
float *arg2 = *array_res;
list = tail;
//printf("Pointer to matrix: %p\n",arg2);
enif_get_list_cell(env,list,&head,&tail);
enif_get_resource(env, head, type, (void **) &array_res);
float *arg3 = *array_res;
list = tail;
//printf("Pointer to matrix: %p\n",arg3);
//printf("Alo kernel");
//fflush(stdout);
enif_get_list_cell(env,list,&head,&tail);
int arg4;
enif_get_int(env, head, &arg4);
list = tail;
//printf("Arg 4 %d\n",arg4);

//printf("yo\n");
addVectors<<<blocks, threads>>>(arg1, arg2, arg3, arg4);
//addVectors<<<blocks, threads>>>();
}

void print2()
{ printf("Alo mundo2\n");
  
}
extern "C" void print()
{ printf("Alo mundo\n");
  print2();
}
