#include "etudes/add.h"
#include <iostream>

#define NUM_BLOCKS 4096
#define NUM_THREADS_PER_BLOCK 512

__global__ void add_(int n, float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

void add()
{
    const int N = 1 << 20;
    float *x, *y;

    x = new float[N];
    y = new float[N];

    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    add_<<<1, 1>>>(N, x, y);
    cudaDeviceSynchronize();

    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = std::max(maxError, std::abs(y[i] - 3.0f));
    std::cout << "\n\nMax error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
}