#mat = Matrex.random(5)

list = [[1,2,3,4,5]]

mat = Matrex.new(list)
#IO.inspect(mat)
#IO.puts("yeah1")
ref1=GPU.create_ref(mat)
ref2=GPU.create_ref(mat)
ref3=GPU.new_ref(5)
#IO.puts("yeah2")
#matrex = GPU.get_matrex(ref)
#IO.inspect(matrex)
#TesteGPU.printMatrex(mat)
#IO.gets("asdf")
kernel=GPU.load_kernel_nif('addVectors')
IO.puts("teste ok")
GPU.spawn(kernel,10,10,[ref3,ref1,ref2,666])
#array = TesteArray.create_array_nif(100)
#TesteArray.print_array_nif(array,100)

#TesteGPU.kernel "asdf
#asdf
#asdf
#     fasdf
#     asdf
#     asdf
#asdf "
