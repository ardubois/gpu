#include "erl_nif.h"

__global__
void mm(float *a, float *b, float *c, int m, int n, int k)
{
int row = ((blockIdx.y * blockDim.y) + threadIdx.y);
int col = ((blockIdx.x * blockDim.x) + threadIdx.x);
int sum = 0;
if(((col < k) && (row < m)))
{
for( int i = 0; i<n; i+=1){
	sum = (sum + (a[((row * n) + i)] * b[((i * k) + col)]));
}

	c[((row * k) + col)] = sum;
}

}

extern "C" void mm_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type)
  {

    ERL_NIF_TERM list;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail;
    float **array_res;

    const ERL_NIF_TERM *tuple_blocks;
    const ERL_NIF_TERM *tuple_threads;
    int arity;

    if (!enif_get_tuple(env, argv[1], &arity, &tuple_blocks)) {
      printf ("spawn: blocks argument is not a tuple");
    }

    if (!enif_get_tuple(env, argv[2], &arity, &tuple_threads)) {
      printf ("spawn:threads argument is not a tuple");
    }
    int b1,b2,b3,t1,t2,t3;

    enif_get_int(env,tuple_blocks[0],&b1);
    enif_get_int(env,tuple_blocks[1],&b2);
    enif_get_int(env,tuple_blocks[2],&b3);
    enif_get_int(env,tuple_threads[0],&t1);
    enif_get_int(env,tuple_threads[1],&t2);
    enif_get_int(env,tuple_threads[2],&t3);

    dim3 blocks(b1,b2,b3);
    dim3 threads(t1,t2,t3);

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

  enif_get_list_cell(env,list,&head,&tail);
  int arg5;
  enif_get_int(env, head, &arg5);
  list = tail;

  enif_get_list_cell(env,list,&head,&tail);
  int arg6;
  enif_get_int(env, head, &arg6);
  list = tail;

   mm<<<blocks, threads>>>(arg1,arg2,arg3,arg4,arg5,arg6);
}
