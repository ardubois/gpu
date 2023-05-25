

defmodule MyKernel do
  import GPU
#@spec myk(integer,integer,integer,integer)::none
kernel add_vectors(result, a, b, n, [:matrex,:matrex,:matrex,:int]) do
  var index int = threadIdx.x + blockIdx.x * blockDim.x;
  var stride int = blockDim.x * gridDim.x;
  for i in range(index,n,stride) do
         result[i] = a[i] + b[i]
  end
end
end

n = 1000

list = [Enum.to_list(1..n)]

vet1 = Matrex.new(list)
vet2 = Matrex.new(list)

IO.inspect vet1

ref1=GPU.create_ref(vet1)
ref2=GPU.create_ref(vet2)
ref3=GPU.new_ref(n)

kernel=GPU.build('add_vectors')

threadsPerBlock = 128;
numberOfBlocks = div(n + threadsPerBlock - 1, threadsPerBlock)

prev = System.monotonic_time()
GPU.spawn(kernel,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref3,ref1,ref2,5])
GPU.synchronize()
next = System.monotonic_time()
IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

result = GPU.get_matrex(ref3)
IO.inspect result


prev = System.monotonic_time()
eresult = Matrex.add(vet1,vet2)
next = System.monotonic_time()
IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"
