#include <iostream>
#include "etudes/add.h"

int main(void)
{
  const int N = 1 << 20;
  float *x, *y;

  x = new float[N];
  y = new float[N];

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  add<<<1, 1>>>(N, x, y);
  cudaDeviceSynchronize();

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = std::max(maxError, std::abs(y[i]-3.0f));
  std::cout << "\n\nMax error: " << maxError << std::endl;

  cudaFree(x);
  cudaFree(y);

  return 0;
}