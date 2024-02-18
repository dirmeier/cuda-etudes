/**
 *  Random-walk Metropolis Hastings sampler
 */

#include <curand.h>
#include <curand_kernel.h>
#include <cmath>
#include <utility>
#include <iostream>
#include <random>

#include "rmh.h"

static const int N_THREADS = 512;
__device__
    void _generate_initial_state(float* initial_state, int dimension, curandState_t prng_state) {
    for(int i=0; i < dimension; i++) {
        initial_state[i] = 0.0;
    }
}

__device__
    void _propose_next_state(float* current_state, float* next_state, int dimension, curandState_t prng_state)  {
    for (int i = 0; i < dimension; i++) {
        next_state[i] = current_state[i] + curand_normal(&prng_state);
    }
}

__device__
float distribution_function(float* x, int dim) {
    float exponent = 0.0;
    for(int i=0; i< dim; i++) {
        exponent += -(x[i]*x[i])/2;
    }
    float probability = exp(exponent);
    return probability;
}

__device__
    float _compute_acceptance_ratio(float current_state[], float proposed_state[], int dimension) {
    return distribution_function(proposed_state, dimension)/distribution_function(current_state, dimension);
}

__global__
    void metropolis_hastings(int num_samples, int dimension, float** samples) {
    float* current_state = (float*) malloc(dimension * sizeof(float));


    int index = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t prng_state;
    curand_init(index, 0,0, &prng_state);
    _generate_initial_state(current_state, dimension, prng_state);

    for(int i=index; i<num_samples; i++) {
        _propose_next_state(current_state, samples[i], dimension, prng_state);
        float acceptance_ratio = fminf(1.0, _compute_acceptance_ratio(current_state, samples[i], dimension));
        float test_probability = curand(&prng_state) / ((float) RAND_MAX);
        if(acceptance_ratio < test_probability) { //Reject, copy state forward
            for(int j=0; j<dimension; j++) {
                //samples[(i * num_samples) + j] = current_state[j][0];
                //samples[(i * num_samples) + (j + 1)] = current_state[j][1];
            }
        }
        current_state = samples[i];
    }
}

    void rmh()
{
    const int n_chains = 1;
    const int N_BLOCKS = (n_chains + 1 + N_THREADS - 1) / N_THREADS;
    std::cout << "\nsampling of" << n_chains << "chains using random walk metropolis\n";
    std::cout << "\tusing n_blocks=" << N_BLOCKS << ", n_threads_per_block=" << N_THREADS << "\n";


    const int dimension = 2;
    const int n_samples = 10;
    float** samples;

    cudaMallocManaged(&samples, n_chains * n_samples * sizeof(float*));
    for(int i = 0; i <   n_chains; i++) {
        for(int j = 0; j < n_samples; j++) {
            cudaMallocManaged(&samples[(i * n_samples) + j], dimension * sizeof(float));
        }
    }

    metropolis_hastings<<<N_BLOCKS, N_THREADS>>>(n_samples, dimension, samples);
    cudaDeviceSynchronize();
    cudaFree(samples);
}
