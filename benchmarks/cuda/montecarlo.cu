#include <stdio.h>

__global__ void gpu_monte_carlo(float *estimate, float *x, float*y, int points_per_thread){
	unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int points_in_circle = 0;
	float p1, p2;


	for(int i = 0; i < points_per_thread; i++) {
		p1 =  x[points_per_thread*tid+i];
		p2 =  y[points_per_thread*tid+i];
		points_in_circle += (p1*p1 + p2*p2 <= 1.0f); 
	}
	estimate[tid] = 4.0 * points_in_circle /  points_per_thread; 
}

int main ()
{
   int blocks = 10;
   int threads = 10;
   int points_per_thread = 2;
   int size = blocks*threads*points_per_thread;
   cudaError_t mc_error;


   float x[size];
   float y[size];
   float estimate[blocks*threads];
   

   float *dev_x, *dev_y, *dev_estimate;

   for (int i = 0; i< size; i++)
    {
      x[i]= rand() / (float) RAND_MAX;
      y[i]= rand() / (float) RAND_MAX;
    }

    ////////
    cudaMalloc( (void**)&dev_x, size*sizeof(float) );
    mc_error = cudaGetLastError();
    if(mc_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(mc_error));
    ////////
    cudaMalloc( (void**)&dev_y, size*sizeof(float) );
    mc_error = cudaGetLastError();
    if(mc_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(mc_error));
    /////////////////////
    cudaMalloc( (void**)&dev_estimate, blocks*threads*sizeof(float) );
    mc_error = cudaGetLastError();
    if(mc_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(mc_error));
    //////////////////////

    cudaMemcpy( dev_x, x, size*sizeof(float), cudaMemcpyHostToDevice );
    mc_error = cudaGetLastError();
    if(mc_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(mc_error));
    cudaMemcpy( dev_y, y, size*sizeof(float), cudaMemcpyHostToDevice );
    mc_error = cudaGetLastError();
    if(mc_error != cudaSuccess) printf("Error 5: %s\n", cudaGetErrorString(mc_error));

    gpu_monte_carlo<<<blocks, threads>>>(dev_estimate, dev_x, dev_y, points_per_thread);
    mc_error = cudaGetLastError();
    if(mc_error != cudaSuccess) printf("Error 6: %s\n", cudaGetErrorString(mc_error));

	 cudaMemcpy(estimate, dev_estimate, blocks * threads * sizeof(float), cudaMemcpyDeviceToHost); // return results 
    mc_error = cudaGetLastError();
    if(mc_error != cudaSuccess) printf("Error 7: %s\n", cudaGetErrorString(mc_error));

	float pi_gpu=0;
	for(int i = 0; i < blocks * threads; i++) {
		pi_gpu += estimate[i];
	}

	pi_gpu /= (blocks * threads);
	printf("Pi estimate dev: %f",pi_gpu);
  
  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_estimate);
//free(x);
  //#free(y);
  //free(estimate);
}

