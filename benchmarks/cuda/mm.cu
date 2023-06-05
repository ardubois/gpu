#include <stdio.h>
__global__ void gpu_mm(float *a,float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

void cpu_mm(float *h_a, float *h_b, float *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

void checkElementsAre(float *gpu, float *cpu, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(gpu[i] != cpu[i])
    {
      printf("FAIL: gpu[%d] - %0.0f does not equal cpu = %0.0f\n", i, gpu[i], cpu[i]);
      exit(1);
    }
  }
  printf("SUCCESS! All values computed correctly.\n");
}

int main(int argc, char const *argv[])
{
    int m = 1000;
    int block_size = 128;

    float *a = (float*) malloc(m*m*sizeof(float));
    float *b = (float*) malloc(m*m*sizeof(float));
    float *c = (float*) malloc(m*m*sizeof(float));
    float *cpu_result = (float*) malloc(m*m*sizeof(float));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            a[i * m + j] = 2.0;
        }
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            b[i * m + j] = 3.0;
        }
    }

    float *d_a, *d_b, *d_c;


    cudaMalloc((void **) &d_a, sizeof(float)*m*m);
    cudaMalloc((void **) &d_b, sizeof(float)*m*m);
    cudaMalloc((void **) &d_c, sizeof(float)*m*m);

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_a, a, sizeof(float)*m*m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float)*m*m, cudaMemcpyHostToDevice);

    int grid_rows = (m + block_size - 1) / block_size;
    int grid_cols = (m + block_size - 1) / block_size;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(block_size, block_size);
   
    gpu_mm<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, m,m,m);    

    cudaMemcpy(c, d_c, sizeof(float)*m*m, cudaMemcpyDeviceToHost);

    cpu_mm(a,b,cpu_result,m,m,m);

    checkElementsAre(c,cpu_result,m*m);

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}
    