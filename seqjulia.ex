defmodule Julia do
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
  def julia_reduce(dim, [{y,x}], array) do
    kernel(dim,{y,x},array)
  end
  def julia_reduce(dim,[{y,x}|tail], array) do
    narray = kernel(dim,{y,x},array)
    julia_reduce(dim,tail,narray)
  end
  def kernel(dim,{y,x}, array) do
    offset = x + (y * dim);

    juliaValue = julia( x, y, dim );
    array
    |> Matrex.set(1, offset*4 + 1, 255 * juliaValue)
    |> Matrex.set(1, offset*4 + 2, 0 )
    |> Matrex.set(1, offset*4 + 3, 0)
    |> Matrex.set(1, offset*4 + 4, 255)
  end
  def julia(x,y,dim) do
    scale = 0.1;
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



indices = for i <- Enum.to_list(0..(dim-1)), j<-Enum.to_list(0..(dim-1)), do: {i,j}


image = Julia.julia_reduce(dim,indices,mat)


#mat = Matrex.fill(1,dim*dim*4,100)
Julia.gen_bmp('julia.bmp',dim,image)
