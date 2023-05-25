#include "erl_nif.h"

__global__
void julia(float *ptr)
{
int dim = 1000;
int x = blockIdx.x;
int y = blockIdx.y;
int offset = (x + (y * gridDim.x));
int juliaValue = 2;
float scale = 1.5;
float jx = ((scale * ((dim / 2) - x)) / (dim / 2));
float jy = ((scale * ((dim / 2) - y)) / (dim / 2));
float cr = (- 0.8);
float ci = 0.156;
float ar = jx;
float ai = jy;
for( int i = 0; i<200; i++){
	ar = (((ar * ar) - (ai * ai)) + cr);
	ai = (((ai * ar) + (ar * ai)) + ci);
if((((ar * ar) + (ai * ai)) > 1000))
{
	juliaValue = 0;
}

if((juliaValue != 0))
{
	juliaValue = 1;
}

}

	ptr[((offset * 4) + 0)] = (255 * juliaValue);
	ptr[((offset * 4) + 1)] = 0;
	ptr[((offset * 4) + 2)] = 0;
	ptr[((offset * 4) + 3)] = 255;
}

extern "C" void julia_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type)
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

   julia<<<blocks, threads>>>(arg1);
    cudaError_t error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)
     { char message[200];
       strcpy(message,"Error kernel call: ");
       strcat(message, cudaGetErrorString(error_gpu));
       enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
     }
}
