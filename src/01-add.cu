#include <iostream>
#include <cmath>
#include "etudes/add.h"

#define M 512
#define L 7

__global__ void add_(int NML, int* x, int* y)
{
  // we rather iterate over some indexes in contrast to how one usually sees
  // it in tutorials otherwise we only use a thread for a single computation
  // which is a waste also when a pointer is longer than MAX_BLOCKS *
  // MAX_THREADS_PER_BLOCK we need to do it this way tl;dr: this is better
  const int idx = (threadIdx.x * L) + (blockIdx.x * blockDim.x * L);
  for (int i = idx; i < idx + L; ++i)
  {
    if (i < NML)
    {
      y[i] = x[i] + y[i];
    }
  }
}

void add()
{
  std::cout << "\nadd two vectors:\n";

  const int NML = 27329;  // random prime number
  int* x;
  int* y;

  cudaMallocManaged(&x, NML * sizeof(int));
  cudaMallocManaged(&y, NML * sizeof(int));

  for (int i = 0; i < NML; ++i)
  {
    x[i] = 1;
    y[i] = 2;
  }

  const int N = (NML + M * L - 1) / (M * L);
  std::cout << "\tusing n_blocks=" << N << ", n_threads_per_block=" << M
            << ", NML=" << NML << ", N*M*L=" << (N * M * L) << "\n";

  add_<<<N, M>>>(NML, x, y);
  cudaDeviceSynchronize();

  for (int i = 0; i < NML; ++i)
  {
    if (y[i] != 3)
      std::cout << "\nincorrect idx: " << i << "\n";
  }

  cudaFree(x);
  cudaFree(y);
}
