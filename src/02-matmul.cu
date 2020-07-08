#include <iostream>
#include <cmath>
#include <cassert>
#include "etudes/matmul.h"

static const int N_THREADS = 5;

__global__ void matmul_(int n_row, int n_col, int width, int* p, int* x, int* y)
{
  const int row = threadIdx.y + blockIdx.y * blockDim.y;
  const int col = threadIdx.x + blockIdx.x * blockDim.x;

  if (col < n_col && row < n_row)
  {
    int sum = 0;
    for (int i = 0; i < width; ++i)
    {
      sum += x[row * width + i] * y[i * n_col + col];
    }
    p[row * n_col + col] = sum;
  }
}

void matmul()
{
  std::cout << "\nmultiply two matrices:\n";

  const int N = 10;
  const int P = 5;
  const int M = 5;
  const int Q = 15;

  int* x;
  int* y;
  int* p;
  assert(P == M);

  cudaMallocManaged(&x, N * P * sizeof(int));
  cudaMallocManaged(&y, M * Q * sizeof(int));
  cudaMallocManaged(&p, N * Q * sizeof(int));

  for (int i = 0; i < N * P; ++i)
  {
    x[i] = i;
  }
  for (int i = 0; i < M * Q; ++i)
  {
    y[i] = i;
  }

  const int N_BLOCKS = (N * Q + N_THREADS - 1) / N_THREADS;
  dim3 dim_blocks(N_BLOCKS, N_BLOCKS, 1);
  dim3 dim_threads(N_THREADS, N_THREADS, 1);

  std::cout << "\tusing n_blocks=" << N_BLOCKS
            << ", n_threads_per_block=" << N_THREADS << "/" << N_THREADS
            << "\n";

  matmul_<<<dim_blocks, dim_threads>>>(N, Q, P, p, x, y);
  cudaDeviceSynchronize();

  cudaFree(x);
  cudaFree(y);
  cudaFree(p);
}
