defmodule TesteArray do
  #import Matrex
  @on_load :load_nifs
  def load_nifs do
      :erlang.load_nif('./c_src/array', 0)
      #IO.puts("ok")
  end

    def print_array_nif(_array,_size) do
      raise "NIF print_matrex_nif/1 not implemented"
    end
    def create_array_nif(_size) do
      raise "NIF print_matrex_nif/1 not implemented"
    end

  end
