defmodule GPU.Backend do


  def gen_c_kernel(kname,nargs,types) do
    gen_header(kname) <> gen_args(nargs,types) <> gen_kernel_call(kname,nargs)
  end

  def gen_header(fname) do
  "extern \"C\" void #{fname}_call(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type)
  {

    ERL_NIF_TERM list;
    ERL_NIF_TERM head;
    ERL_NIF_TERM tail;

    int blocks,threads;
    float **array_res;

    enif_get_int(env, argv[1], &blocks);
    enif_get_int(env, argv[2], &threads);
    list= argv[3];

"
  end
  def gen_kernel_call(kernelname,nargs) do
"   #{kernelname}<<<blocks, threads>>>" <> gen_call_args(nargs) <> ";
}
"
  end
  def gen_call_args(nargs) do
    "(" <> gen_call_args_(nargs-1) <>"arg#{nargs})"
  end
  def gen_call_args_(0) do
    ""
  end
  def gen_call_args_(n) do
    args = gen_call_args_(n-1)
    args <> "arg#{n},"
  end
  def gen_args(0,_l) do
    ""
  end
  def gen_args(n,[]) do
    args = gen_args(n-1,[])
    arg = gen_arg_matrix(n)
    args <> arg
  end
  def gen_args(n,[:matrex|t]) do
    args = gen_args(n-1,t)
    arg = gen_arg_matrix(n)
    args <> arg
  end
  def gen_args(n,[:int|t]) do
    args = gen_args(n-1,t)
    arg = gen_arg_int(n)
    args <> arg
  end
  def gen_args(n,[:float|t]) do
    args = gen_args(n-1,t)
    arg = gen_arg_float(n)
    args <> arg
  end
  def gen_args(n,[:double|t]) do
    args = gen_args(n-1,t)
    arg = gen_arg_double(n)
    args <> arg
  end
  def gen_arg_matrix(narg) do
"  enif_get_list_cell(env,list,&head,&tail);
  enif_get_resource(env, head, type, (void **) &array_res);
  float *arg#{narg} = *array_res;
  list = tail;

"
  end
  def gen_arg_int(narg) do
"  enif_get_list_cell(env,list,&head,&tail);
  int arg#{narg};
  enif_get_int(env, head, &arg4);
  list = tail;

"
  end
  def gen_arg_float(narg) do
"  enif_get_list_cell(env,list,&head,&tail);
  double darg#{narg};
  float arg#{narg};
  enif_get_double(env, head, &darg4);
  arg#{narg} = (float) darg#{narg};
  list = tail;

"
  end
  def gen_arg_double(narg) do
"  enif_get_list_cell(env,list,&head,&tail);
  double arg#{narg};
  enif_get_double(env, head, &darg4);
  list = tail;

"
  end
end
