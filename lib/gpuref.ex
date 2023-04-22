defmodule GPURef do
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./priv/gpu_ref_nifs', 0)
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
end
