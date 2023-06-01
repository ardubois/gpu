defmodule Julia do
  import GPU
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/julia_nifs', 0)
  end
  def gen_bmp_nif(_string,_dim,_mat) do
      raise "gen_bmp_nif not implemented"
  end
  def gen_bmp(string,dim,%Matrex{data: matrix} = _a) do
    gen_bmp_nif(string,dim,matrix)
  end
  kernel ripple_kernel(ptr,dim,ticks,[:matrex,:int,:int]) do
    var x int = threadIdx.x + blockIdx.x * blockDim.x;
    var y int = threadIdx.y + blockIdx.y * blockDim.y;
    var offset int = x + y * blockDim.x * gridDim.x;

    var fx float = 0.5 *  x - dim/15;
    var fy float  = 0.5 *  y - dim/15;
    var d float = sqrtf( fx * fx + fy * fy );
    var grey float = floor(128.0 + 127.0 *cos(d/10.0 - ticks/7.0) /(d/10.0 + 1.0));
    ptr[offset*4 + 0] = grey;
    ptr[offset*4 + 1] = grey;
    ptr[offset*4 + 2] = grey;
    ptr[offset*4 + 3] = 255;
  end
  def gen_pixel(ptr,dim,ticks,{y,x}) do
    offset = x + (y * dim)
    fx = 0.5 *  x - dim/15;#x - dim/2;
    fy = 0.5 *  y - dim/15;#y - dim/2;
    d  = :math.sqrt( fx * fx + fy * fy );
    grey = floor(128.0 + 127.0 * :math.cos(d/10.0 - ticks/7.0) /(d/10.0 + 1.0));

    ptr
    |> Matrex.set(1, offset*4 + 1, grey)
    |> Matrex.set(1, offset*4 + 2, grey)
    |> Matrex.set(1, offset*4 + 3, grey)
    |> Matrex.set(1, offset*4 + 4, 255)
  end
  def ripple_seq(ptr,dim, ticks, [{y,x}]) do
    gen_pixel(ptr,dim, ticks, {y,x})
  end
  def ripple_seq(ptr, dim,ticks ,[{y,x}|tail]) do
    narray = gen_pixel(ptr,dim,ticks, {y,x})
    ripple_seq(narray,dim,ticks, tail)
  end
end

dim =300

mat = Matrex.fill(1,dim*dim*4,0)

ref=GPU.create_ref(mat)

ker=GPU.build('ripple_kernel')
n_threads = 128
n_blocks = floor ((dim+(n_threads-1))/n_threads)


prev = System.monotonic_time()
GPU.spawn(ker,{n_blocks,n_threads,1},{n_blocks,n_threads,1},[ref,dim,10])
GPU.synchronize()
next = System.monotonic_time()
IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

image = GPU.get_matrex(ref)

Julia.gen_bmp('ripplegpu.bmp',dim,image)

indices = for i <- Enum.to_list(0..(dim-1)), j<-Enum.to_list(0..(dim-1)), do: {i,j}

prev = System.monotonic_time()
imageseq = Julia.ripple_seq(mat,dim,10,indices)
next = System.monotonic_time()
IO.puts "time cpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

Julia.gen_bmp('rippleseq.bmp',dim,imageseq)
