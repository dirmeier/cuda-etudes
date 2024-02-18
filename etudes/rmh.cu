/**
 *  Random-walk Metropolis sampler
 */

#include <curand.h>
#include <curand_kernel.h>

#include <cmath>
#include <iostream>
#include <random>
#include <utility>

#include "rmh.h"


__device__ void _generate_initial_state(float* initial_state, int dimension, curandState_t prng_state)
{
    for (int i = 0; i < dimension; i++) {
        initial_state[i] = curand_normal(&prng_state);
    }
}

__device__ void _propose_next_state(float* state, float* previous_state, int dimension, curandState_t prng_state)
{
    for (int i = 0; i < dimension; i++) {
        state[i] = previous_state[i] + curand_normal(&prng_state);
    }
}

__device__ float _pdf(float* x, int dim)
{
    float lp = 0.0;
    for (int i = 0; i < dim; i++) {
        lp += -(x[i] * x[i]) / 2;
    }
    return exp(lp);
}

__device__ float _mh_ratio(float* proposed_state, float* last_state, int dimension)
{
    float mh_ratio =  _pdf(proposed_state, dimension) / _pdf(last_state, dimension);
    return fminf(1.0, mh_ratio);
}

__global__ void metropolis_hastings(int n_chains, int n_samples, int n_dimension, float*** samples)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_chains) return;

    curandState_t prng_state;
    curand_init(index, 0, 0, &prng_state);

    float* proposal = (float*) malloc(n_dimension * sizeof(float));
    _generate_initial_state(proposal, n_dimension, prng_state);
    for (int j = 0; j < n_dimension; ++j) {
        samples[index][0][j] = proposal[j];
    }

    for (int i = 1; i < n_samples; ++i) {
        _propose_next_state(proposal, samples[index][i - 1], n_dimension, prng_state);
        float accept_prob = _mh_ratio(proposal, samples[index][i - 1], n_dimension);
        float randr = curand_uniform(&prng_state);
        if (randr <= accept_prob) {
            for (int j = 0; j < n_dimension; ++j) {
                samples[index][i][j] = proposal[j];
            }
        }
        else {
            for (int j = 0; j < n_dimension; ++j) {
                samples[index][i][j] = samples[index][i - 1][j] ;
            }
        }
    }
}

void rmh()
{
    const int n_chains = 1000;
    const int N_THREADS = 32 * 4;
    const int N_BLOCKS = (n_chains + 1 + N_THREADS - 1) / N_THREADS;
    std::cout << "\nsampling of " << n_chains << " chains using random walk metropolis\n";
    std::cout << "\tusing n_blocks=" << N_BLOCKS << ", n_threads_per_block=" << N_THREADS << "\n";

    const int dimension = 10;
    const int n_samples = 1000;
    float*** samples;

    cudaMallocManaged(&samples, n_chains * sizeof(float**));
    for (int i = 0; i < n_chains; i++) {
           cudaMallocManaged(&samples[i], n_samples * sizeof(float*));
        for (int j = 0; j < n_samples; j++) {
            cudaMallocManaged(&samples[i][j], dimension * sizeof(float));
        }
    }
    metropolis_hastings<<<N_BLOCKS, N_THREADS>>>(n_chains, n_samples, dimension, samples);
    cudaDeviceSynchronize();
    cudaFree(samples);
}
