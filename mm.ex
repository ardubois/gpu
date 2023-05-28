

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


mat = Matrex.fill(1,m*k,1)

f = fn _ -> Enum.random(1..10) end

mat1 = Matrex.apply(mat,f)
mat2 = Matrex.apply(mat,f)
a=GPU.create_ref(mat1)
b=GPU.create_ref(mat2)
c=GPU.new_ref(m*k)

ker=GPU.build('mm')

block_size = 16
grid_rows = trunc ((m + block_size - 1) / block_size)
grid_cols = trunc ((k + block_size - 1) / block_size)

prev = System.monotonic_time()

GPU.spawn(ker,{grid_rows,grid_cols,1},{block_size,block_size,1},[a,b,c,m,n,k])
GPU.synchronize()

next = System.monotonic_time()
IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

result = GPU.get_matrex(c)
IO.inspect result
#IO.puts GPU.Backend.gen_c_kernel('addVectors',4,[])

amat = Matrex.reshape(mat1,m,k)
bmat = Matrex.reshape(mat2,m,k)

prev = System.monotonic_time()
cmat = Matrex.dot(amat,bmat)
next = System.monotonic_time()
IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

rmat = Matrex.reshape(result,m,k)

fmat = Matrex.subtract(cmat,rmat)


IO.puts "this value must be zero: #{Matrex.sum(fmat)}"
