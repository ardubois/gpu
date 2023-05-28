defmodule Julia do
  import GPU
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/julia_nifs', 0)
  end
  def gen_bmp_nif(string,dim,(%Matrex{data: matrix} = a)) do
      raise "gen_bmp_nif not implemented"
  end
  def gen_bmp(string,dim,%Matrex{data: matrix} = a) do
    gen_bmp_nif(string,dim,matrix)
  end
kernel julia_kernel(ptr,dim,[:matrex,:int]) do
  var x int = blockIdx.x
  var y int = blockIdx.y
  var offset int = x + y * gridDim.x
#####
  var juliaValue int = 2
  var scale float = 0.1
  var jx float = scale * (dim - x)/dim
  var jy float = scale * (dim - y)/dim

  var cr float = -0.8
  var ci float = 0.156
  var ar float = jx
  var ai float = jy
  for i in range(0,200) do
      ar = (ar*ar - ai*ai) + cr
      ai = (ai*ar + ar*ai) + ci
      if ((ar * ar)+(ai * ai ) > 1000) do
        juliaValue = 0
        break
      end
  end
  if (juliaValue != 0) do
    juliaValue = 1
  end
#####
  ptr[offset*4 + 0] = 255 * juliaValue;
  ptr[offset*4 + 1] = 0;
  ptr[offset*4 + 2] = 0;
  ptr[offset*4 + 3] = 255;

end
def julia_seq(dim, [{y,x}], array) do
  gen_pixel(dim,{y,x},array)
end
def julia_seq(dim,[{y,x}|tail], array) do
  narray = gen_pixel(dim,{y,x},array)
  julia_seq(dim,tail,narray)
end
def gen_pixel(dim,{y,x}, array) do
  offset = x + (y * dim)

  juliaValue = julia( x, y, dim )
  array
  |> Matrex.set(1, offset*4 + 1, 255 * juliaValue)
  |> Matrex.set(1, offset*4 + 2, 0 )
  |> Matrex.set(1, offset*4 + 3, 0)
  |> Matrex.set(1, offset*4 + 4, 255)
end
def julia(x,y,dim) do
  scale = 0.1
  jx = scale * (dim - x)/dim
  jy = scale * (dim - y)/dim
  cr  = -0.8
  ci  = 0.156
  ar  = jx
  ai  = jy
  test_julia(200,cr,ci,ar,ai,1000)
end
def test_julia(0,_cr,_ci,_ar,_ai,_number) do
  1
end
def test_julia(n,cr,ci,ar,ai,number) do
  nar = ((ar * ar) - (ai*ai)) + cr
  nai = ((ai * ar) + (ar*ai)) + ci
  if (nar*nar + nai*nai > number) do
    0
  else
    test_julia(n-1,cr,ci,nar,nai,number)
  end
end
end


dim =400

mat = Matrex.fill(1,dim*dim*4,0)

ref=GPU.create_ref(mat)

ker=GPU.build('julia_kernel')

prev = System.monotonic_time()
GPU.spawn(ker,{dim,1,1},{dim,1,1},[ref,dim])
GPU.synchronize()
next = System.monotonic_time()
IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

image = GPU.get_matrex(ref)

Julia.gen_bmp('juliacuda.bmp',dim,image)

indices = for i <- Enum.to_list(0..(dim-1)), j<-Enum.to_list(0..(dim-1)), do: {i,j}


imageseq = Julia.julia_seq(dim,indices,mat)

Julia.gen_bmp('julia.bmp',dim,imageseq)
