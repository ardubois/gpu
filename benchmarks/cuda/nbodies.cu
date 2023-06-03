#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



typedef struct { float x, y, z, vx, vy, vz; } Body;

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__
void bodyForce(Body *p, float dt, int n,float softening) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + softening;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

__global__
void gpu_bodyForce(float *p, float dt, int n, float softening) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {

    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j] - p[i];
      float dy = p[j+1] - p[i+1];
      float dz = p[j+2] - p[i+2];
      float distSqr = dx*dx + dy*dy + dz*dz + softening;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; 
      Fy += dy * invDist3; 
      Fz += dz * invDist3;
    }

    p[i+3]+= dt*Fx; 
    p[i+4] += dt*Fy; 
    p[i+5] += dt*Fz;
  }
}

void cpu_bodyForce(float *p, float dt, int n,float softening) {
  for(int i = 0; i<n; i++)
  {

    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j] - p[i];
      float dy = p[j+1] - p[i+1];
      float dz = p[j+2] - p[i+2];
      float distSqr = dx*dx + dy*dy + dz*dz + softening;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; 
      Fy += dy * invDist3; 
      Fz += dz * invDist3;
    }

    p[i+3]+= dt*Fx; 
    p[i+4] += dt*Fy; 
    p[i+5] += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  int block_size =  128;
  float softening = 0.000000001;
  cudaError_t nb_error;
  
  const float dt = 0.01; // time step
  

  int bytes = nBodies*sizeof(Body);
  float *h_buf = (float*)malloc(bytes);
  float *d_resp = (float*)malloc(bytes);
 

  randomizeBodies(h_buf, 6*nBodies); // Init pos / vel data

  float *d_buf;

  ///////////////////
  cudaMalloc(&d_buf, bytes);
  nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 1: %s\n", cudaGetErrorString(nb_error));
  //////// 
  

  int nBlocks = (nBodies + block_size - 1) / block_size;
  
  
  clock_t begin = clock();
 
  //////////////////////////////// 
  cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice);
   nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 2: %s\n", cudaGetErrorString(nb_error));
  //////// 
 

   ////////////////////
    gpu_bodyForce<<<nBlocks, block_size>>>(d_buf, dt, nBodies,softening); // compute interbody forces
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 3: %s\n", cudaGetErrorString(nb_error));
  //////// 
 


   /////////////////////////////
   cudaMemcpy(d_resp, d_buf, bytes, cudaMemcpyDeviceToHost);
    nb_error = cudaGetLastError();
    if(nb_error != cudaSuccess) printf("Error 4: %s\n", cudaGetErrorString(nb_error));
  //////// 
 
    float *p = d_resp;
    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i] += p[i+3]*dt;
      p[i+1] += p[i+4]*dt;
      p[i+2] += p[i+5]*dt;
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("GPU elapsed time is %f seconds\n", time_spent);

    begin = clock();
    cpu_bodyForce(h_buf,dt,nBodies,softening);
    p = h_buf;
    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i] += p[i+3]*dt;
      p[i+1] += p[i+4]*dt;
      p[i+2] += p[i+5]*dt;
    }
  
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("CPU elapsed time is %f seconds\n", time_spent);
    free(h_buf);
    free(d_resp);
    cudaFree(d_buf);
}