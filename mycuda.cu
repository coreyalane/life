#include <stdio.h>
#include <cuda.h>
 
struct some_data {
	some_data(float a_, int b_) {
		a = a_;
		b = b_;
	}
	
	float a;
	int b;

	__device__ float other_func() {
		return a * (float)b;
	}
};

// Kernel that executes on the CUDA device
__global__ void square_array(float *a, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<N) a[idx] = a[idx] * a[idx];
}

__global__ void square_array(some_data* a, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N) {
    //a[idx].a = a[idx].a * a[idx].a;
    //a[idx].b = a[idx].b * a[idx].b;
    a[idx].a = a[idx].other_func();
  }
}
 
// main routine that executes on the host
int main(void)
{
  some_data *a_h;  // Pointer to host & device arrays
  const int N = 10;  // Number of elements in arrays
  size_t size = N * sizeof(some_data);
  cudaMallocManaged((void **) &a_h, size);   // Allocate array on device
  cudaDeviceSynchronize();
  // Initialize host array and copy it to CUDA device
  for (int i=0; i<N; i++){
     //a_h[i].a = (float)i;
     //a_h[i].b = i+2;
     a_h[i] = some_data((float)i, i + 2);
  }
  // Do calculation on device:
  int block_size = 4;
  int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
  square_array <<< n_blocks, block_size >>> (a_h, N);
  cudaDeviceSynchronize();
  // Print results
  for (int i=0; i<N; i++){
    printf("%d %f %d\n", i, a_h[i].a, a_h[i].b);
  }
  // Cleanup
  cudaFree(a_h);
}
