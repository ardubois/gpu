defmodule TesteGPU do
#import Matrex
@on_load :load_nifs
def load_nifs do
    :erlang.load_nif('./c_src/printMatrec', 0)
end

  def print_matrex_nif(_a) do
    raise "NIF print_matrex_nif/1 not implemented"
  end
  def printMatrex(%Matrex{data: matrix} = _a) do
            print_matrex_nif(matrix)
  end
  def kernel(str) do
    IO.inspect(str)
  end
end
