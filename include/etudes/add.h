#ifndef CUDA_ETUDES_ADD_H
#define CUDA_ETUDES_ADD_H

#define NUM_BLOCKS 4096
#define NUM_THREADS_PER_BLOCK 512

__global__ void add(int n, float *x, float *y);

#endif