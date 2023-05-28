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
kernel julia(ptr,dim,[:matrex,:int]) do
  var x int = blockIdx.x;
  var y int = blockIdx.y;
  var offset int = x + y * gridDim.x;
#####
  var juliaValue int = 2
  var scale float = 1.5;
  var jx float = scale * (dim/2 - x)/(dim/2);
  var jy float = scale * (dim/2 - y)/(dim/2);

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
end


dim =400

mat = Matrex.fill(1,dim*dim*4,0)

ref=GPU.create_ref(mat)

ker=GPU.build('julia')

prev = System.monotonic_time()
GPU.spawn(ker,{dim,1,1},{dim,1,1},[ref,dim])
GPU.synchronize()
next = System.monotonic_time()
IO.puts "time gpu #{System.convert_time_unit(next-prev,:native,:millisecond)}"

image = GPU.get_matrex(ref)

Julia.gen_bmp('julia.bmp',dim,image)
