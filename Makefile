all: priv/gpu_nifs.so priv/gpu_ref_nifs.so

priv/gpu_nifs.so: c_src/gpu_nifs.cu
	nvcc --shared -g --compiler-options '-fPIC' -o priv/gpu_nifs.so c_src/gpu_nifs.cu

julia: c_src/julia_nifs.cu 
	nvcc --shared -g --compiler-options '-fPIC' -o priv/julia_nifs.so c_src/julia_nifs.cu

priv/gpu_ref_nifs.so: c_src/gpu_ref_nifs.cu
	nvcc --shared -g --compiler-options '-fPIC' -o priv/gpu_ref_nifs.so c_src/gpu_ref_nifs.cu
clean:
	rm priv/gpu_ref_nifs.so priv/gpu_nifs.so
