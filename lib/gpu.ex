defmodule GPU do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/gpu_nifs', 0)
      #IO.puts("ok")
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
