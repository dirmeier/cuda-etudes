#include <iostream>
#include <cmath>
#include <random>

#include <curand.h>
#include "etudes/gauss_jordan.h"

static const int N_THREADS = 5;

//__global__ void gauss_jordan_(const int n, float* ab) {
//  const int row = threadIdx.y + blockIdx.y * blockDim.y;
//  const int col = threadIdx.x + blockIdx.x * blockDim.x;
//
////  int k = 0, c;
////  for (int i = 0; i < n; i++) {
////    if (a[i][i] == 0) {
////      c = 1;
////      while ((i + c) < n && a[i + c][i] == 0)
////        c++;
////      if ((i + c) == n) {
////        flag = 1;
////        break;
////      }
////      for (j = i, k = 0; k <= n; k++)
////        swap(a[j][k], a[j + c][k]);
////    }
////
////    for (int j = 0; j < n; j++) {
////
////      // Excluding all i == j
////      if (i != j) {
////
////        // Converting Matrix to reduced row
////        // echelon form(diagonal matrix)
////        float pro = a[j][i] / a[i][i];
////
////        for (k = 0; k <= n; k++)
////          a[j][k] = a[j][k] - (a[i][k]) * pro;
////      }
////    }
////  }
//}

void gauss_jordan() {
  std::cout << "\nsolving system of linear equations:\n";

  const int N = 3;

  float* A = new float[N * N];
  float* b = new float[N];
  float* Ab = new float[N * N + N];

  cudaMallocManaged(&Ab, (N * N + N) * sizeof(float));

  std::mt19937 mt_rand(23);
  std::normal_distribution<> rnorm(0, 1);
  for (int i = 0; i < N; ++i) {
    b[i] = rnorm(mt_rand);
    for (int j = 0; j < N; ++j) {
      A[i * N + j] = rnorm(mt_rand);
    }
  }

  for (int i = 0; i < N; ++i) {
    Ab[i * (N + 1) + N] = b[i];
    for (int j = 0; j < N; ++j) {
      Ab[i * (N + 1) + j] = A[i * N + j];
    }
  }

  const int N_BLOCKS = ((N * (N + 1)) + N_THREADS - 1) / N_THREADS;
  dim3 dim_blocks(N_BLOCKS, N_BLOCKS, 1);
  dim3 dim_threads(N_THREADS, N_THREADS, 1);

  std::cout << "\tusing n_blocks=" << N_BLOCKS
            << ", n_threads_per_block=" << N_THREADS << "/" << N_THREADS
            << "\n";

//  gauss_jordan_ << < dim_blocks, dim_threads >> > (N, N + 1, Ab);
  cudaDeviceSynchronize();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      std::cout << A[i * N + j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";

  for (int j = 0; j < N; ++j) {
    std::cout << b[j] << " ";
  }
  std::cout << "\n";
  std::cout << "\n";

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < (N + 1); ++j) {
      std::cout << Ab[i * (N + 1) + j] << " ";
    }
    std::cout << "\n";
  }

  delete[] A;
  delete[] b;
  cudaFree(Ab);
}
