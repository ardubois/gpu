defmodule GPU.CudaBackend do
  def gen_kernel(name,para,body) do
    "__global__\nvoid #{name}(#{para})\n{\n#{body}\n}"
  end
  def gen_body(body) do
    case body do
      {:__block__, _, _code} ->
        gen_block body
      {:do, {:__block__,pos, code}} ->
        gen_block {:__block__, pos,code}
      {:do, exp} ->
        gen_command exp
      {_,_,_} ->
        gen_command body
    end
  end
  defp gen_block({:__block__, _, code}) do
    code
        |>Enum.map(&gen_command/1)
        |>Enum.join("\n")
  end
  defp gen_header_for(header) do
    case header do
      {:in, _,[{var,_,nil},{:range,_,[n]}]} ->
            "for( int #{var} = 0; #{var}<#{gen_exp n}; #{var}++)"
      {:in, _,[{var,_,nil},{:range,_,[argr1,argr2]}]} ->
            "for( int #{var} = #{gen_exp argr1}; #{var}<#{gen_exp argr2}; #{var}++)"
      {:in, _,[{var,_,nil},{:range,_,[argr1,argr2,step]}]} ->
            "for( int #{var} = #{gen_exp argr1}; #{var}<#{gen_exp argr2}; #{var}+=#{gen_exp step})"
    end
  end
  defp gen_command(code) do
  #  if check_atrib_last code do
   #    gen_atrib_last code
   # else
      case code do
       {:for,_,[param,[body]]} ->
          header = gen_header_for(param)
          body = gen_body(body)
          header <> "{\n" <> body <> "\n}\n"
        {:=, _, [arg, exp]} ->
          a = gen_exp arg
          e = gen_exp exp
          "\t#{a} = #{e}\;"
        {:if, _, if_com} ->
            genIf(if_com)
        {:var, _ , [{var,_,[{:=, _, [{type,_,nil}, exp]}]}]} ->
          IO.puts "aqui"
          gexp = gen_exp exp
          "#{to_string type} #{to_string var} = #{gexp};"
        {:var, _ , [{var,_,[{:=, _, [type, exp]}]}]} ->
            gexp = gen_exp exp
            "#{to_string type} #{to_string var} = #{gexp};"
        {:var, _ , [{var,_,[{type,_,_}]}]} ->
           "#{to_string type} #{to_string var};"
        {:var, _ , [{var,_,[type]}]} ->
           "#{to_string type} #{to_string var};"
        {fun, _, args} ->
          nargs=args
          |> Enum.map(&gen_exp/1)
          |> Enum.join(", ")
          "#{fun}(#{nargs})\;"
        number when is_integer(number) or is_float(number) -> to_string(number)
      end
    end
    defp gen_exp(exp) do
      case exp do
         {{:., _, [Access, :get]}, _, [arg1,arg2]} ->
          name = gen_exp arg1
          index = gen_exp arg2
          "#{name}[#{index}]"
        {{:., _, [{struct, _, nil}, field]},_,[]} ->
          "#{to_string struct}.#{to_string(field)}"
        {{:., _, [{:__aliases__, _, [struct]}, field]}, _, []} ->
          "#{to_string struct}.#{to_string(field)}"
        {op, _, args} when op in [:+, :-, :/, :*, :<=, :<, :>, :>=, :&&, :||, :!] ->
          case args do
            [a1] ->
              "(#{to_string(op)} #{gen_exp a1})"
            [a1,a2] ->
              "(#{gen_exp a1} #{to_string(op)} #{gen_exp a2})"
            end
        {var, _, nil} when is_atom(var) -> to_string(var)
        {fun, _, args} ->
          nargs=args
          |> Enum.map(&gen_exp/1)
          |> Enum.join(", ")
          "#{fun}(#{nargs})"
        number when is_integer(number) or is_float(number) -> to_string(number)
      end

    end
    defp genIf([bexp, [do: then]]) do
        gen_then([bexp, [do: then]])
    end
    defp genIf([bexp, [do: thenbranch, else: elsebranch]]) do
         gen_then([bexp, [do: thenbranch]])
         <>
         "else{\n" <>
         (gen_body elsebranch) <>
         "\n}\n"
    end
    defp gen_then([bexp, [do: then]]) do
      "if(#{gen_exp bexp})\n" <>
      "{\n" <>
      (gen_body then) <>
      "\n}\n"
    end

#######
  def gen_kernel_call(kname,nargs,types) do
    gen_header(kname) <> gen_args(nargs,types) <> gen_call(kname,nargs)
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
  def gen_call(kernelname,nargs) do
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
