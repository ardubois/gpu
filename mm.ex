

defmodule MyKernel do
  import GPU
kernel mm(a,b,c,m,n,k, [:matrex,:matrex,:matrex,:int,:int,:int]) do
  var row int = blockIdx.y * blockDim.y + threadIdx.y
  var col int = blockIdx.x * blockDim.x + threadIdx.x
  var sum int = 0
  if(col < k && row < m) do
    for i in range(0,n,1) do
      sum = sum + a[row * n + i] * b[i * k + col]
    end
    c[row * k + col] = sum
  end

end
end

m = 1000
n = 1000
k = 1000

list = [Enum.to_list(1..(m*k))]
mat = Matrex.new(list)

a=GPU.create_ref(mat)
b=GPU.create_ref(mat)
c=GPU.new_ref(m*k)

ker=GPU.build('mm')

block_size = 16
grid_rows = trunc ((m + block_size - 1) / block_size)
grid_cols = trunc ((k + block_size - 1) / block_size)



GPU.spawn(ker,{grid_rows,grid_cols,1},{block_size,block_size,1},[a,b,c,m,n,k])
GPU.synchronize()
result = GPU.get_matrex(c)
IO.inspect result
#IO.puts GPU.Backend.gen_c_kernel('addVectors',4,[])
