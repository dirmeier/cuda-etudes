/**
 *  Gaussian Jordan elimination can be found here:
 * https://en.wikipedia.org/wiki/Gaussian_elimination
 *  @author: Simon Dirmeier <simon.dirmeier @ web.de>
 */

#include <iostream>
#include <cmath>
#include <random>

#include <curand.h>
#include "etudes/gauss_jordan.h"

static const int N_THREADS = 512;

int col_arg(const int i, const int n, const int ncol, float* ab)
{
  int c = 0;
  while ((i + c) < n && ab[(i + c) * ncol + i] == 0)
    c++;
  return i + c;
}

__global__ void swap(
  const int n, const int i, const int j, const int ncol, float* ab)
{
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  if (k <= n)
  {
    float tmp        = ab[j * ncol + k];
    ab[j * ncol + k] = ab[i * ncol + k];
    ab[i * ncol + k] = tmp;
  }
}

__device__ void substract_(
  const int i, const int j, const int k, const int ncol, float pro, float* ab)
{
  ab[j * ncol + k] -= ab[i * ncol + k] * pro;
}

__global__ void substract(const int i, const int n, const int ncol, float* ab)
{
  const int j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j < n)
  {
    if (i != j)
    {
      float pro = ab[j * ncol + i] / ab[i * ncol + i];
      for (int k = 0; k <= n; k++)
        substract_(i, j, k, ncol, pro, ab);
    }
  }
}

void gauss_jordan_(const int n, const int n_blocks, float* ab, float* x)
{
  for (int i = 0; i < n; ++i)
  {
    // form upper triangular matrix
    if (ab[i * (n + 1) + i] == 0)
    {
      int i_max = col_arg(i, n, n + 1, ab);
      if (i_max == n)
        return;
      swap<<<n_blocks, N_THREADS>>>(n, i, i_max, n + 1, ab);
      cudaDeviceSynchronize();
    }
    // row transformation
    substract<<<n_blocks, N_THREADS>>>(i, n, n + 1, ab);
    cudaDeviceSynchronize();
  }

  for (int i = 0; i < n; ++i)
    x[i] = ab[i * (n + 1) + n] / ab[i * (n + 1) + i];
}

void gauss_jordan()
{
  std::cout << "\nsolving system of linear equations:\n";

  const int N = 3;

  float* A = new float[N * N];
  float* b = new float[N];
  float* x = new float[N];
  float* Ab;

  cudaMallocManaged(&Ab, (N * N + N) * sizeof(float));

  std::mt19937 mt_rand(23);
  std::normal_distribution<> rnorm(0, 1);
  for (int i = 0; i < N; ++i)
  {
    b[i] = rnorm(mt_rand);
    for (int j = 0; j < N; ++j)
    {
      A[i * N + j] = rnorm(mt_rand);
    }
  }

  for (int i = 0; i < N; ++i)
  {
    Ab[i * (N + 1) + N] = b[i];
    for (int j = 0; j < N; ++j)
    {
      Ab[i * (N + 1) + j] = A[i * N + j];
    }
  }

  const int N_BLOCKS = (N + 1 + N_THREADS - 1) / N_THREADS;
  std::cout << "\tusing n_blocks=" << N_BLOCKS
            << ", n_threads_per_block=" << N_THREADS << "\n";

  gauss_jordan_(N, N_BLOCKS, Ab, x);

  delete[] A;
  delete[] b;
  delete[] x;
  cudaFree(Ab);
}
