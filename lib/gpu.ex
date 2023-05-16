defmodule GPU do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/gpu_nifs', 0)
      #IO.puts("ok")
  end
  defp gen_para(p,:matrex) do
    "float *#{p}"
  end
  defp gen_para(p,:float) do
    "float #{p}"
  end
  defp gen_para(p,:int) do
    "int #{p}"
  end
  defmacro kernel(header, do: body) do
      {fname, _, para} = header
     {param_list,types} = if is_list(List.last(para)) do
        types = List.last(para)
        param_list = para
          |> List.delete_at(length(para)-1)
          |> Enum.map(fn({p, _, _}) -> p end)
          |> Enum.zip(types)
          |> Enum.map(fn({p,t}) -> gen_para(p,t) end)
          |> Enum.join(", ")
        {param_list,types}
      else
      types = List.duplicate(:matrex,length(para))
      param_list = para
      |> Enum.map(fn({p, _, _}) -> p end)
      |> Enum.zip(types)
      |> Enum.map(fn({p,t}) -> gen_para(p,t) end)
      |> Enum.join(", ")
      {param_list,types}
     end
     cuda_body = GPU.CudaBackend.gen_body(body)
     k = GPU.CudaBackend.gen_kernel(fname,param_list,cuda_body)
     accessfunc = GPU.CudaBackend.gen_kernel_call(fname,length(types),Enum.reverse(types))
     file = File.open!("c_src/#{fname}.cu", [:write])
     IO.write(file, "#include \"erl_nif.h\"\n\n" <> k <> "\n\n" <> accessfunc)
     File.close(file)
     #IO.puts k
     #IO.puts accessfunc
     quote do
        def unquote(header)do
          IO.puts "hello"
        end
      end
  end
  def create_ref_nif(_matrex) do
    raise "NIF create_ref_nif/1 not implemented"
end
def create_ref(%Matrex{data: matrix} = a) do
  ref=create_ref_nif(matrix)
  {ref, Matrex.size(a)}
end
def new_ref_nif(_matrex) do
  raise "NIF new_ref_nif/1 not implemented"
end
def synchronize_nif() do
  raise "NIF new_ref_nif/1 not implemented"
end
def synchronize() do
  synchronize_nif()
end
def new_ref(size) do
ref=new_ref_nif(size)
{ref, {1,size}}
end
def get_matrex_nif(_ref,_rows,_cols) do
raise "NIF get_matrex_nif/1 not implemented"
end
def get_matrex({ref,{rows,cols}}) do
%Matrex{data: get_matrex_nif(ref,rows,cols)}
end

def load_kernel_nif(_matrex) do
  raise "NIF new_ref_nif/1 not implemented"
end
def kernel(s) do
  s
end
def build(kernelname) do
  {result, _errcode} = System.cmd("nvcc",
      ["--shared",
        "--compiler-options",
        "'-fPIC'",
        "-o",
        "priv/#{kernelname}.so",
        "c_src/#{kernelname}.cu"
      ], stderr_to_stdout: true)
  IO.puts(result)
  GPU.load_kernel_nif(kernelname)
end
def build_kernel(kernel,kernelname,nargs,types) do
  accessfunc = GPU.Backend.gen_c_kernel(kernelname,nargs,Enum.reverse(types))
  file = File.open!("c_src/#{kernelname}.cu", [:write])
  IO.write(file, "#include \"erl_nif.h\"\n\n" <> kernel <> "\n\n" <> accessfunc)
  File.close(file)
  {result, _errcode} = System.cmd("nvcc",
      ["--shared",
        "--compiler-options",
        "'-fPIC'",
        "-o",
        "priv/#{kernelname}.so",
        "c_src/#{kernelname}.cu"
      ], stderr_to_stdout: true)
  IO.puts(result)
  GPU.load_kernel_nif(kernelname)
end
def spawn_nif(_k,_t,_b,_l) do
  raise "NIF spawn_nif/1 not implemented"
end
def spawn(k,t,b,l) do
  spawn_nif(k,t,b,Enum.map(l,&get_ref/1))
end
def get_ref({ref,{_rows,_cols}}) do
  ref
end
def get_ref(e) do
  e
end
end
