#include "etudes/add.h"
#include <iostream>

#define M 512

__global__ void add_(int N, float *x, float *y)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
    y[i] = x[i] + y[i];
}

void add()
{
  const int N = 10;
  float *x = new float[N];
  float *y = new float[N];

  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  for (int i = 0; i < N; ++i)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  add_<<<2, 2>>>(N, x, y);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; ++i) std::cout << x[i] << " " << y[i] << "\n";

  cudaFree(x);
  cudaFree(y);
}