
k = GPU.kernel "__global__
void add_vectors(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }

}
"
defmodule MyKernel do
  import GPU
#@spec myk(integer,integer,integer,integer)::none
kernel aadd_vectors(result, a, b, n, [:matrex,:matrex,:matrex,:int]) do
  var index int = threadIdx.x + blockIdx.x * blockDim.x;
  var stride int = blockDim.x * gridDim.x;
  for i in range(index,n,stride) do
         result[i] = a[i] + b[i]
  end
end
end


list = [[1,2,3,4,5]]

mat = Matrex.new(list)
#IO.inspect(mat)
#IO.puts("yeah1")
ref1=GPU.create_ref(mat)
ref2=GPU.create_ref(mat)
ref3=GPU.new_ref(5)
#IO.puts("yeah2")
#matrex = GPU.get_matrex(ref)
#IO.inspect(matrex)
#TesteGPU.printMatrex(mat)
#IO.gets("asdf")
kernel=GPU.build('aadd_vectors')
IO.puts("teste ok")
GPU.spawn(kernel,10,10,[ref3,ref1,ref2,666])

#IO.puts GPU.Backend.gen_c_kernel('addVectors',4,[])
