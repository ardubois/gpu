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
   int blocks = 100;
   int threads = 128;
   int points_per_thread = 5;
   int size = blocks*threads*points_per_thread;


   float x[size];
   float y[size];
   float estimate[blocks*threads];
   float dev_estimate[blocks*threads];

   float *dev_x, *dev_y;

   for (int i = 0; i< size; i++)
    {
      x[i]= rand() / (float) RAND_MAX;
      y[i]= rand() / (float) RAND_MAX;
    }

    cudaMalloc( (void**)&dev_x, size*sizeof(float) );
    cudaMalloc( (void**)&dev_y, size*sizeof(float) );
    cudaMalloc( (void**)&dev_estimate, blocks*threads*sizeof(float) );

    cudaMemcpy( dev_x, x, size*sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_y, y, size*sizeof(float), cudaMemcpyHostToDevice );

    gpu_monte_carlo<<<blocks, threads>>>(dev_estimate, dev_x, dev_y, points_per_thread);

	cudaMemcpy(estimate, dev_estimate, blocks * threads * sizeof(float), cudaMemcpyDeviceToHost); // return results 

	float pi_gpu=0;
	for(int i = 0; i < blocks * threads; i++) {
		pi_gpu += dev_estimate[i];
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

